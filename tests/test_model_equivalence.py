"""The refactored scheduler LM against the research implementation, on a tiny random model."""

import importlib
import sys

import pytest
import torch

from neural_scheduler.model import (
    ChatGLMConfig,
    ChatGLMForConditionalGeneration,
    ChatGLMForConditionalGenerationPredDuration,
    compute_channel_ids,
)

pytestmark = pytest.mark.equivalence

ORIG_MODEL_DIR = "ntpp/model/chatGLM_dual_timing_only"
TINY = dict(
    num_layers=3,
    padded_vocab_size=120,
    hidden_size=64,
    ffn_hidden_size=96,
    kv_channels=16,
    num_attention_heads=4,
    seq_length=256,
    multi_query_attention=True,
    multi_query_group_num=2,
    original_rope=True,
    use_cache=True,
    torch_dtype="float32",
    num_channel=3,
    num_duration_classes=32,
)


@pytest.fixture(scope="module")
def orig(ntpp_orig):
    """The original model modules, imported with their directory on sys.path."""
    model_dir = str(ntpp_orig / ORIG_MODEL_DIR)
    sys.path.insert(0, model_dir)
    try:
        modules = {
            "config": importlib.import_module("configuration_chatglm_timing"),
            "duration": importlib.import_module("modeling_chatglm_timing_duration_pred_stable"),
            "plain": importlib.import_module("modeling_chatglm_timing"),
        }
    finally:
        sys.path.remove(model_dir)
    return modules


def build_pair(orig, with_duration, attn_implementation=None):
    """Original and refactored models with identical float32 weights."""
    torch.manual_seed(0)
    kwargs = dict(TINY, use_duration_prediction=with_duration)
    old_config = orig["config"].ChatGLMConfig(**kwargs)
    new_config = ChatGLMConfig(**kwargs)
    if attn_implementation is not None:
        old_config._attn_implementation = attn_implementation
        new_config._attn_implementation = attn_implementation
    if with_duration:
        old = orig["duration"].ChatGLMForConditionalGenerationPredDuration(old_config, empty_init=False)
        new = ChatGLMForConditionalGenerationPredDuration(new_config, empty_init=False)
    else:
        old = orig["plain"].ChatGLMForConditionalGeneration(old_config, empty_init=False)
        new = ChatGLMForConditionalGeneration(new_config, empty_init=False)
    assert old_config._attn_implementation == new_config._attn_implementation
    assert list(old.state_dict().keys()) == list(new.state_dict().keys())
    old = old.float()
    # Some parameters (RMSNorm weights) are allocated uninitialised; give every one a value.
    with torch.no_grad():
        for parameter in old.parameters():
            parameter.normal_(mean=0.0, std=0.2)
    # Cast before copying: the channel embedding is created in half precision.
    new = new.float()
    new.load_state_dict(old.state_dict(), strict=True)
    return old.eval(), new.eval()


def make_batch(pad, with_duration, seed=1):
    """Two samples with different script lengths; ``pad`` right-padding tokens on the shorter one."""
    g = torch.Generator().manual_seed(seed)
    starts = [7, 11]
    lengths = [7 + 20, 11 + 20 + pad]
    seq = max(lengths)
    input_ids = torch.randint(0, 100, (2, seq), generator=g)
    attention_mask = torch.zeros(2, seq, dtype=torch.long)
    labels = torch.full((2, seq), -100)
    durations = torch.full((2, seq), -100)
    for b, (start, length) in enumerate(zip(starts, lengths)):
        attention_mask[b, :length] = 1
        labels[b, start:length] = input_ids[b, start:length]
        durations[b, start:length] = torch.randint(0, 60, (length - start,), generator=g)
    batch = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        generation_start_index=torch.tensor(starts),
    )
    if with_duration:
        batch["remaining_durations"] = durations
        batch["duration_loss_weight"] = 0.5
    return batch


def assert_same_outputs(old_out, new_out, with_duration):
    assert torch.equal(old_out.logits, new_out.logits)
    assert torch.equal(old_out.loss, new_out.loss)
    if with_duration:
        assert torch.equal(old_out.duration_logits, new_out.duration_logits)
        assert torch.equal(old_out.token_loss, new_out.token_loss)
        assert torch.equal(old_out.duration_loss, new_out.duration_loss)
        assert torch.equal(old_out.duration_accuracy, new_out.duration_accuracy)


def test_default_attention_implementation_matches(orig):
    old, new = build_pair(orig, with_duration=True)
    assert old.config._attn_implementation == new.config._attn_implementation


@pytest.mark.parametrize("with_duration", [True, False])
@pytest.mark.parametrize("attn", [None, "eager", "sdpa"])
def test_unpadded_batch(orig, with_duration, attn):
    old, new = build_pair(orig, with_duration, attn)
    batch = make_batch(pad=0, with_duration=with_duration)
    # Same length for both samples: no padding anywhere.
    batch["input_ids"] = batch["input_ids"][:, :27]
    batch["attention_mask"] = torch.ones(2, 27, dtype=torch.long)
    batch["labels"] = batch["labels"][:, :27]
    if with_duration:
        batch["remaining_durations"] = batch["remaining_durations"][:, :27]
    with torch.no_grad():
        assert_same_outputs(old(**batch), new(**batch), with_duration)


@pytest.mark.parametrize("with_duration", [True, False])
def test_single_sequence_with_explicit_channel_ids(orig, with_duration):
    """The inference call pattern: batch of one, channel ids given, no cache."""
    old, new = build_pair(orig, with_duration)
    g = torch.Generator().manual_seed(3)
    input_ids = torch.randint(0, 100, (1, 41), generator=g)
    channel_ids = compute_channel_ids(41, torch.tensor([9]), input_ids.device)
    kwargs = dict(input_ids=input_ids, channel_ids=channel_ids, attention_mask=torch.ones_like(input_ids),
                  use_cache=False)
    with torch.no_grad():
        old_out, new_out = old(**kwargs), new(**kwargs)
    assert torch.equal(old_out.logits, new_out.logits)
    if with_duration:
        assert torch.equal(old_out.duration_logits, new_out.duration_logits)


@pytest.mark.parametrize("with_duration", [True, False])
def test_cached_decoding_step_by_step(orig, with_duration):
    """Prefill then single-token steps with the KV cache give the same logits in both."""
    old, new = build_pair(orig, with_duration)
    g = torch.Generator().manual_seed(4)
    input_ids = torch.randint(0, 100, (1, 30), generator=g)
    channel_ids = compute_channel_ids(30, torch.tensor([8]), input_ids.device)
    position_ids = torch.arange(30).unsqueeze(0)

    def run(model):
        logits = []
        with torch.no_grad():
            out = model(input_ids=input_ids[:, :20], channel_ids=channel_ids[:, :20],
                        position_ids=position_ids[:, :20], attention_mask=torch.ones(1, 20, dtype=torch.long),
                        use_cache=True)
            logits.append(out.logits)
            past = out.past_key_values
            for t in range(20, 30):
                out = model(input_ids=input_ids[:, t:t + 1], channel_ids=channel_ids[:, :t + 1],
                            position_ids=position_ids[:, t:t + 1],
                            attention_mask=torch.ones(1, t + 1, dtype=torch.long),
                            past_key_values=past, use_cache=True)
                logits.append(out.logits)
                past = out.past_key_values
        return torch.cat(logits, dim=1)

    assert torch.equal(run(old), run(new))


@pytest.mark.parametrize("with_duration", [True, False])
@pytest.mark.parametrize("attn", ["eager", "sdpa"])
def test_padded_batch(orig, with_duration, attn):
    """Right-padded batch, including the rotary-position behaviour under padding.

    The research code's mask for the non-flash attention paths was left
    half-disabled; the release restores the stock GLM mask. To compare
    everything else exactly, the original is given the same mask function.
    """
    old, new = build_pair(orig, with_duration, attn)
    old.transformer.get_masks = new.transformer.get_masks
    batch = make_batch(pad=5, with_duration=with_duration)
    with torch.no_grad():
        assert_same_outputs(old(**batch), new(**batch), with_duration)


def test_compute_channel_ids(orig):
    for seq_length, starts in [(12, [0, 5, 12]), (31, [7]), (8, [20])]:
        start = torch.tensor(starts)
        expected = orig["duration"].compute_channel_ids(seq_length, start, start.device)
        assert torch.equal(compute_channel_ids(seq_length, start, start.device), expected)
    assert torch.equal(
        compute_channel_ids(9, 4, torch.device("cpu")),
        orig["duration"].compute_channel_ids(9, 4, torch.device("cpu")),
    )


def test_duration_head_init_matches_stable(orig):
    """Same RNG state, same constructor draws: the heads start identical."""
    kwargs = dict(TINY, use_duration_prediction=True)
    torch.manual_seed(5)
    old = orig["duration"].DurationPredictionHead(orig["config"].ChatGLMConfig(**kwargs))
    torch.manual_seed(5)
    from neural_scheduler.model import DurationPredictionHead
    new = DurationPredictionHead(ChatGLMConfig(**kwargs))
    for (k1, v1), (k2, v2) in zip(old.state_dict().items(), new.state_dict().items()):
        assert k1 == k2 and torch.equal(v1, v2)
