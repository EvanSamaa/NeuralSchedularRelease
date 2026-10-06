"""Scheduler LM behaviour that can be checked without the research tree or real weights."""

import pytest
import torch

from neural_scheduler.constants import CHANNEL_DIALOGUE_A, CHANNEL_DIALOGUE_B, CHANNEL_SCRIPT
from neural_scheduler.model import (
    CausalLMOutputWithPastAndDuration,
    ChatGLMConfig,
    ChatGLMForConditionalGeneration,
    ChatGLMForConditionalGenerationPredDuration,
    compute_channel_ids,
    load_scheduler,
    scheduler_model_class,
)

TINY = dict(
    num_layers=2,
    padded_vocab_size=90,
    hidden_size=32,
    ffn_hidden_size=48,
    kv_channels=8,
    num_attention_heads=4,
    seq_length=128,
    multi_query_attention=True,
    multi_query_group_num=2,
    use_cache=True,
    torch_dtype="float32",
)


def tiny_model(with_duration=True, seed=0):
    torch.manual_seed(seed)
    config = ChatGLMConfig(**TINY, use_duration_prediction=with_duration)
    model = scheduler_model_class(config)(config, empty_init=False).float()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.2)
    return model.eval()


def test_compute_channel_ids_layout():
    ids = compute_channel_ids(8, torch.tensor([3, 0]), torch.device("cpu"))
    S, A, B = CHANNEL_SCRIPT, CHANNEL_DIALOGUE_A, CHANNEL_DIALOGUE_B
    assert ids.tolist() == [[S, S, S, A, B, A, B, A], [A, B, A, B, A, B, A, B]]
    assert compute_channel_ids(4, 1, torch.device("cpu")).tolist() == [[S, A, B, A]]


def test_config_selects_class_and_drops_auto_map():
    config = ChatGLMConfig(**TINY, use_duration_prediction=True,
                           auto_map={"AutoModel": "THUDM/glm-4-voice-9b--modeling_chatglm.X"})
    assert not hasattr(config, "auto_map")
    assert scheduler_model_class(config) is ChatGLMForConditionalGenerationPredDuration
    config.use_duration_prediction = False
    assert scheduler_model_class(config) is ChatGLMForConditionalGeneration
    assert config.num_channel == 3 and config.num_duration_classes == 32


def test_state_dict_layout():
    plain = set(tiny_model(with_duration=False).state_dict())
    duration = set(tiny_model(with_duration=True).state_dict())
    assert duration - plain == {
        "duration_head.duration_predictor.0.weight",
        "duration_head.duration_predictor.0.bias",
        "duration_head.duration_predictor.3.weight",
        "duration_head.duration_predictor.3.bias",
    }
    assert "transformer.channel_emb.channel_embedding.weight" in plain
    assert tiny_model().transformer.channel_emb.channel_embedding.weight.shape == (3, TINY["kv_channels"])


def test_forward_outputs_and_losses():
    model = tiny_model()
    g = torch.Generator().manual_seed(1)
    input_ids = torch.randint(0, 80, (2, 24), generator=g)
    start = torch.tensor([6, 6])
    labels = input_ids.clone()
    labels[:, :6] = -100
    durations = torch.full((2, 24), -100)
    durations[:, 6:] = torch.randint(0, 50, (2, 18), generator=g)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=labels,
                    generation_start_index=start, remaining_durations=durations, duration_loss_weight=0.25)
    assert isinstance(out, CausalLMOutputWithPastAndDuration)
    assert out.logits.shape == (2, 24, 90) and out.duration_logits.shape == (2, 24, 32)
    assert out.duration_logits.dtype == torch.float32
    assert torch.allclose(out.loss, out.token_loss + 0.25 * out.duration_loss)

    # Labels above the last class are clamped to it.
    clamped = torch.where(durations == -100, durations, durations.clamp(max=31))
    with torch.no_grad():
        out_clamped = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=labels,
                            generation_start_index=start, remaining_durations=clamped)
    assert torch.equal(out.duration_loss, out_clamped.duration_loss)


def test_model_without_head_rejects_durations_and_has_no_duration_outputs():
    model = tiny_model(with_duration=False)
    input_ids = torch.randint(0, 80, (1, 10))
    out = model(input_ids=input_ids, generation_start_index=torch.tensor([4]), labels=input_ids)
    assert not hasattr(out, "duration_logits") and out.loss is not None
    with pytest.raises(ValueError):
        model(input_ids=input_ids, generation_start_index=torch.tensor([4]),
              remaining_durations=torch.zeros(1, 10, dtype=torch.long))


def test_channel_ids_are_required():
    with pytest.raises(ValueError):
        tiny_model()(input_ids=torch.randint(0, 80, (1, 10)))


def test_generate_is_disabled():
    with pytest.raises(NotImplementedError):
        tiny_model().generate(torch.zeros(1, 3, dtype=torch.long))


def test_causality():
    """Changing a later token leaves earlier logits untouched."""
    model = tiny_model()
    input_ids = torch.randint(0, 80, (1, 20))
    changed = input_ids.clone()
    changed[0, 15] = (changed[0, 15] + 1) % 80
    kwargs = dict(generation_start_index=torch.tensor([5]), use_cache=False)
    with torch.no_grad():
        a = model(input_ids=input_ids, **kwargs).logits
        b = model(input_ids=changed, **kwargs).logits
    assert torch.equal(a[:, :15], b[:, :15])
    assert not torch.equal(a[:, 15:], b[:, 15:])


@pytest.mark.parametrize("attn", ["eager", "sdpa"])
def test_right_padding(attn):
    """Padding is masked from attention, but shifts the rotary positions of the padded row.

    A row without padding reproduces its single-sequence logits. A padded row
    does not: its pad positions carry channel id 0 and are counted as script
    tokens when dialogue positions are computed. This pins the behaviour the
    released checkpoint was trained with.
    """
    torch.manual_seed(0)
    config = ChatGLMConfig(**TINY, use_duration_prediction=True)
    config._attn_implementation = attn
    model = ChatGLMForConditionalGenerationPredDuration(config, empty_init=False).float()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.2)
    model.eval()

    lengths, starts = [20, 26], [5, 7]
    input_ids = torch.randint(0, 80, (2, 26))
    attention_mask = torch.zeros(2, 26, dtype=torch.long)
    for row, length in enumerate(lengths):
        attention_mask[row, :length] = 1
    with torch.no_grad():
        batched = model(input_ids=input_ids, attention_mask=attention_mask,
                        generation_start_index=torch.tensor(starts)).logits
        solo = [
            model(input_ids=input_ids[row:row + 1, :length], attention_mask=torch.ones(1, length, dtype=torch.long),
                  generation_start_index=torch.tensor([starts[row]])).logits[0]
            for row, length in enumerate(lengths)
        ]
    assert torch.allclose(batched[1], solo[1], atol=1e-5)
    # Script positions of the padded row are unaffected, its dialogue positions are shifted.
    assert torch.allclose(batched[0, :starts[0]], solo[0][:starts[0]], atol=1e-5)
    assert not torch.allclose(batched[0, starts[0]:lengths[0]], solo[0][starts[0]:], atol=1e-3)

    # Explicit channel ids that are non-zero on the padding avoid the shift.
    channel_ids = compute_channel_ids(26, torch.tensor(starts), input_ids.device)
    with torch.no_grad():
        fixed = model(input_ids=input_ids, attention_mask=attention_mask, channel_ids=channel_ids).logits
    assert torch.allclose(fixed[0, :lengths[0]], solo[0], atol=1e-5)


def test_load_scheduler_roundtrip(tmp_path):
    """A checkpoint saved from the model loads back to identical weights and outputs."""
    snapshot = _glm_tokenizer_files()
    # The real tokenizer needs room for its ids, so this tiny model has the full vocabulary.
    torch.manual_seed(2)
    config = ChatGLMConfig(**{**TINY, "padded_vocab_size": 168960}, use_duration_prediction=True)
    model = ChatGLMForConditionalGenerationPredDuration(config, empty_init=False).float()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.2)
    model.eval()
    model.save_pretrained(tmp_path)
    for name in snapshot:
        (tmp_path / name.name).write_bytes(name.read_bytes())

    loaded, tokenizer = load_scheduler(tmp_path, device="cpu")
    assert isinstance(loaded, ChatGLMForConditionalGenerationPredDuration)
    assert not loaded.training
    assert next(loaded.parameters()).dtype == torch.float32
    assert tokenizer.convert_tokens_to_ids("<|start-of-script-A|>") == 168736
    assert tokenizer.convert_tokens_to_ids("<|end-turn-49|>") == 168841
    for (k1, v1), (k2, v2) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert k1 == k2 and torch.equal(v1, v2)

    input_ids = torch.randint(0, 168842, (1, 16))
    kwargs = dict(input_ids=input_ids, generation_start_index=torch.tensor([4]), use_cache=False)
    with torch.no_grad():
        assert torch.equal(model(**kwargs).logits, loaded(**kwargs).logits)


def _glm_tokenizer_files():
    """Tokenizer files of GLM-4-Voice from the local Hugging Face cache, or skip."""
    from pathlib import Path

    try:
        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download("THUDM/glm-4-voice-9b", local_files_only=True,
                                          allow_patterns=["token*", "special_tokens_map.json", "added_tokens.json"]))
    except Exception:
        pytest.skip("THUDM/glm-4-voice-9b tokenizer is not in the local Hugging Face cache")
    files = [p for p in snapshot.iterdir() if p.name.startswith("token") or p.name in
             ("special_tokens_map.json", "added_tokens.json")]
    if not any(p.name == "tokenizer.model" for p in files):
        pytest.skip("THUDM/glm-4-voice-9b tokenizer is not in the local Hugging Face cache")
    return files
