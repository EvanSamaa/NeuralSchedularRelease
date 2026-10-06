"""End-to-end run of the trainer on CPU with a tiny model: train, checkpoint, continue."""

import json
import sys
from pathlib import Path

import pytest
import torch

from neural_scheduler.constants import SILENCE_TOKEN_ID
from neural_scheduler.model import ChatGLMConfig, ChatGLMForConditionalGenerationPredDuration, load_scheduler
from neural_scheduler.vocab import FIRST_TURN_TOKEN_ID


def glm_tokenizer_files():
    try:
        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download("THUDM/glm-4-voice-9b", local_files_only=True,
                                          allow_patterns=["token*", "special_tokens_map.json", "added_tokens.json"]))
    except Exception:
        pytest.skip("THUDM/glm-4-voice-9b tokenizer is not in the local Hugging Face cache")
    files = [p for p in snapshot.iterdir()
             if p.name.startswith("token") or p.name in ("special_tokens_map.json", "added_tokens.json")]
    if not any(p.name == "tokenizer.model" for p in files):
        pytest.skip("THUDM/glm-4-voice-9b tokenizer is not in the local Hugging Face cache")
    return files


def write_records(path, count, seed):
    """Windows shaped like the preprocessing output: script, then interleaved dialogue."""
    g = torch.Generator().manual_seed(seed)
    with open(path, "w") as f:
        for i in range(count):
            script = [168736, SILENCE_TOKEN_ID, FIRST_TURN_TOKEN_ID] + \
                (152353 + torch.randint(0, 16000, (4 + i % 3,), generator=g)).tolist() + \
                [FIRST_TURN_TOKEN_ID + 1, 168737, 168738, 168739, 168740]
            frames = 6 + i % 4
            dialogue = [SILENCE_TOKEN_ID] * (2 * frames)
            record = {
                "input_ids": script + dialogue + [168741],
                "generation_start_index": len(script),
                "generation_end_index": len(script) + len(dialogue),
                "remaining_durations": [(frames - 1 - k // 2) for k in range(2 * frames)],
                "window_idx": i,
            }
            f.write(json.dumps(record) + "\n")


def run_training(monkeypatch, argv):
    from neural_scheduler.train import train as train_module

    monkeypatch.setattr(sys, "argv", ["train"] + [str(a) for a in argv])
    train_module.train()


def test_train_checkpoint_and_continue(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    tokenizer_files = glm_tokenizer_files()

    init_dir = tmp_path / "init"
    torch.manual_seed(0)
    # A config as the stock model ships it: no scheduler fields.
    config = ChatGLMConfig(num_layers=2, padded_vocab_size=168960, hidden_size=16, ffn_hidden_size=24, kv_channels=4,
                           num_attention_heads=4, multi_query_attention=True, multi_query_group_num=2,
                           torch_dtype="float32", use_cache=True, use_duration_prediction=True)
    model = ChatGLMForConditionalGenerationPredDuration(config, empty_init=False).float()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.05)
    model.save_pretrained(init_dir)
    for source in tokenizer_files:
        (init_dir / source.name).write_bytes(source.read_bytes())

    write_records(tmp_path / "train.jsonl", 12, seed=1)
    write_records(tmp_path / "val.jsonl", 4, seed=2)

    common = [
        "--attn_implementation", "sdpa", "--torch_dtype", "float32",
        "--train_file", tmp_path / "train.jsonl", "--validation_file", tmp_path / "val.jsonl",
        "--cache_dir", tmp_path / "cache", "--preprocessing_num_workers", 1, "--dataloader_num_workers", 0,
        "--use_duration_prediction", "--num_duration_classes", 32, "--duration_loss_weight", 0.5,
        "--do_train", "--do_eval", "--use_cpu", "--report_to", "none",
        "--per_device_train_batch_size", 2, "--per_device_eval_batch_size", 2, "--learning_rate", 1e-3,
        "--max_steps", 4, "--save_strategy", "steps", "--save_steps", 2, "--eval_strategy", "steps",
        "--eval_steps", 2, "--logging_steps", 1, "--no_remove_unused_columns", "--seed", 42,
    ]
    run_dir = tmp_path / "run"
    run_training(monkeypatch, ["--model_name_or_path", init_dir, "--output_dir", run_dir] + common)

    checkpoint = run_dir / "checkpoint-4"
    assert (checkpoint / "model.safetensors").is_file()
    saved_config = json.loads((checkpoint / "config.json").read_text())
    assert saved_config["architectures"] == ["ChatGLMForConditionalGenerationPredDuration"]
    assert saved_config["use_duration_prediction"] is True and saved_config["num_channel"] == 3
    assert "auto_map" not in saved_config
    eval_metrics = json.loads((run_dir / "eval_results.json").read_text())
    assert torch.isfinite(torch.tensor(eval_metrics["eval_loss"])) and eval_metrics["eval_samples"] == 4

    trained, tokenizer = load_scheduler(checkpoint, device="cpu")
    assert len(tokenizer) == 168842
    changed = [k for k, v in trained.state_dict().items() if not torch.equal(v, model.state_dict()[k])]
    assert "transformer.channel_emb.channel_embedding.weight" in changed
    assert "duration_head.duration_predictor.3.weight" in changed

    # --init_from starts a new run from the trained weights, not from stock GLM-4-Voice.
    continued_dir = tmp_path / "continued"
    args = [a for a in common]
    args[args.index("--max_steps") + 1] = 1
    args[args.index("--learning_rate") + 1] = 0.0
    args[args.index("--save_steps") + 1] = 1
    run_training(monkeypatch, ["--init_from", checkpoint, "--output_dir", continued_dir] + args)
    continued, _ = load_scheduler(continued_dir / "checkpoint-1", device="cpu")
    for key, value in trained.state_dict().items():
        assert torch.equal(value, continued.state_dict()[key]), key
