"""Helpers shared by the backchannel tests (no tests of its own).

* loaders for the original research modules, with the imports they cannot
  satisfy on a CPU test machine replaced by stubs;
* tiny randomly initialised RoBERTa / HuBERT / GPT-2 checkpoints;
* synthetic transcripts, labelled interactions and classification data.
"""

import ast
import importlib
import json
import os
import random
import sys
import types
from pathlib import Path

import numpy as np
import pytest

BC_CLASSES = ["Neutral", "Positive", "Emotional", "Anticipation", "Confirmation", "Repetition"]
ORIG_PREPROCESS = "BackchannelPrediction/preprocess_scripts"
ORIG_CLASSIFIER = "BackchannelPrediction/naive_roberta_classifier"
ORIG_STEP1_JOINT = (
    "ntpp/model/chatGLM_dual_timing_only/experiments/"
    "inference_pipeline_bc_only_full_duplex_bench/step_1_no_tts_classifier_voiceline_given_with_audio.py"
)
ORIG_STEP1_TEXT = (
    "ntpp/model/chatGLM_dual_timing_only/experiments/"
    "inference_pipeline_10_examples/step_1_no_tts_classifier_voiceline_given.py"
)


def import_original(ntpp_orig: Path, subdir: str, module: str, stubs=()):
    """Import ``module`` from ``ntpp_orig/subdir``. ``stubs`` are module names
    replaced by empty modules for the duration of the import."""
    path = str(ntpp_orig / subdir)
    added = []
    for name in stubs:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
            added.append(name)
    sys.path.insert(0, path)
    try:
        return importlib.import_module(module)
    finally:
        sys.path.remove(path)
        for name in added:
            del sys.modules[name]


def import_original_extract(ntpp_orig: Path):
    """The labelling script. Its labeller import needs a newer ``transformers``
    and is stubbed; ``pandas`` is stubbed when it cannot be imported."""
    labeler_stub = types.ModuleType("qwen_process_one_example_test_batch")
    labeler_stub.QwenOmniBatchLabeler = object
    sys.modules["qwen_process_one_example_test_batch"] = labeler_stub
    pandas_stubbed = False
    try:
        import pandas  # noqa: F401
    except ImportError:
        pandas_stub = types.ModuleType("pandas")
        pandas_stub.DataFrame = object
        sys.modules["pandas"] = pandas_stub
        pandas_stubbed = True
    try:
        return import_original(ntpp_orig, ORIG_PREPROCESS, "extract_backchannel_dataset_sharded_v2")
    finally:
        del sys.modules["qwen_process_one_example_test_batch"]
        if pandas_stubbed:
            del sys.modules["pandas"]


def extract_definitions(source_path: Path, names, namespace):
    """Execute the top-level functions / classes ``names`` of a script in
    ``namespace`` without importing the script."""
    tree = ast.parse(source_path.read_text())
    wanted = [n for n in tree.body
              if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in wanted} == set(names), f"missing definitions in {source_path}"
    module = ast.Module(body=wanted, type_ignores=[])
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace


# --- synthetic transcripts ---------------------------------------------------

WORDS = ["yeah", "so", "I", "think", "the", "dog", "ran", "home", "okay", "right", "wow", "mm-hmm",
         "really", "that's", "what", "we", "did", "then", "no", "way"]


def make_transcript(rng: random.Random, n_sentences: int, t0: float = 0.0):
    """A list of sentences in the Seamless Interaction transcript format."""
    sentences = []
    t = t0
    for _ in range(n_sentences):
        t += rng.choice([0.05, 0.2, 0.4, 1.0, 2.5, 4.0])
        n_words = rng.choice([1, 1, 2, 3, 6, 10])
        words = []
        for _ in range(n_words):
            dur = rng.choice([0.1, 0.2, 0.35])
            gap = rng.choice([0.0, 0.05, 0.3, 0.9, 1.2])
            start = round(t + gap, 3)
            end = round(start + dur, 3)
            t = end
            if rng.random() < 0.05:
                words.append({"word": rng.choice(WORDS), "start": None, "end": None})
            else:
                words.append({"word": rng.choice(WORDS), "start": start, "end": end})
        timed = [w for w in words if w["start"] is not None]
        if not timed:
            continue
        sentences.append({
            "words": words,
            "start": timed[0]["start"],
            "end": timed[-1]["end"],
            "transcript": " ".join(w["word"] for w in words),
        })
    return sentences


class FakeLabeler:
    """Stands in for the Qwen labeller: deterministic, sometimes invalid, answers."""

    ANSWERS = (
        ['{"classification": "%s", "word": "yeah"}' % c for c in BC_CLASSES]
        + ['{"classification": "NotBackchannel", "word": "the dog"}',
           '{"classification": "NotBackchannel", "word": ""}',
           "not json", "", "{'classification': 'Neutral', 'word': 'ok'}",
           '{"classification": "Bogus", "word": "x"}']
    )

    def __init__(self):
        self.calls = []

    def run_batch_labeling(self, dataset_pairs, batch_size=16):
        self.calls.append((json.dumps(dataset_pairs), batch_size))
        out = []
        for pair in dataset_pairs:
            key = sum(map(ord, pair[0]["transcript"] + "|" + pair[1]["transcript"]))
            out.append(self.ANSWERS[key % len(self.ANSWERS)])
        return out


# --- synthetic labelled interactions ----------------------------------------

def make_interaction(rng: random.Random, interaction_id: int, n_turns: int, wav_dir: str, split: str):
    """An interaction record in the labelling output format."""
    turns = []
    t = 0.0
    for turn_id in range(n_turns):
        main = rng.choice([0, 1])
        turn_start = t + rng.uniform(0.0, 1.5)
        cursor = turn_start
        for _ in range(rng.choice([1, 1, 2, 3])):
            dur = rng.uniform(0.2, 4.0)
            turns.append({"text": " ".join(rng.choices(WORDS, k=rng.randint(1, 8))),
                          "start": round(cursor, 3), "end": round(cursor + dur, 3),
                          "speaker_id": main, "turn_id": turn_id, "label": "Speech"})
            cursor += dur + rng.uniform(0.0, 0.6)
        for _ in range(rng.choice([0, 0, 0, 1, 1, 2])):
            start = rng.uniform(turn_start, cursor)
            dur = rng.choice([0.1, 0.25, 0.32, 0.33, 0.6, 1.1])
            turns.append({"text": rng.choice(WORDS), "start": round(start, 3),
                          "end": round(start + dur, 3), "speaker_id": 1 - main, "turn_id": turn_id,
                          "label": "backchannel-" + rng.choice(BC_CLASSES)})
        if rng.random() < 0.2:
            start = rng.uniform(turn_start, cursor)
            turns.append({"text": "but listen to me now", "start": round(start, 3),
                          "end": round(start + 1.7, 3), "speaker_id": 1 - main, "turn_id": turn_id,
                          "label": "Speech"})
        t = cursor
    turns.sort(key=lambda x: x["start"])
    return {
        "interaction_id": interaction_id,
        "identifier_left": f"V00_S{interaction_id:04d}_I0001_P0001",
        "identifier_right": f"V00_S{interaction_id:04d}_I0001_P0002",
        "wav_left": os.path.join(wav_dir, f"{interaction_id}_left.wav"),
        "wav_right": os.path.join(wav_dir, f"{interaction_id}_right.wav"),
        "transcript_left": "left.json", "transcript_right": "right.json",
        "split": split, "dataset_type": "naturalistic", "sr": 16000,
        "num_turns": n_turns, "num_sentences": len(turns),
        "num_backchannels": sum(t["label"] != "Speech" for t in turns),
        "turns": turns,
    }


def make_interactions(seed: int, n: int, wav_dir: str = "/data/wavs", write_wavs: bool = False,
                      n_turns: int = 14):
    rng = random.Random(seed)
    interactions = [
        make_interaction(rng, i, n_turns, wav_dir, "train" if i % 3 else "test") for i in range(n)
    ]
    if write_wavs:
        import soundfile as sf

        os.makedirs(wav_dir, exist_ok=True)
        nprng = np.random.default_rng(seed)
        for inter in interactions:
            duration = max(t["end"] for t in inter["turns"]) + 0.5
            for i, key in enumerate(("wav_left", "wav_right")):
                # one file at 16 kHz mono, the other at 8 kHz stereo, to cover resampling
                sr, channels = (16000, 1) if i == 0 else (8000, 2)
                audio = 0.1 * nprng.standard_normal((int(duration * sr), channels)).astype(np.float32)
                sf.write(inter[key], audio.squeeze(), sr)
    return interactions


# --- tiny pretrained models ---------------------------------------------------

@pytest.fixture(scope="session")
def tiny_models(tmp_path_factory):
    """Tiny random RoBERTa, GPT-2 and HuBERT checkpoints (with real tokenizers
    for the two text models). Skips when the tokenizers are not cached locally."""
    import torch
    from transformers import (
        GPT2Config, GPT2Model, GPT2Tokenizer, HubertConfig, HubertModel,
        RobertaConfig, RobertaModel, RobertaTokenizer,
    )

    try:
        roberta_tok = RobertaTokenizer.from_pretrained("roberta-base")
        gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
    except Exception as e:  # not cached and no network
        pytest.skip(f"roberta-base / gpt2 tokenizers unavailable: {e}")

    root = tmp_path_factory.mktemp("tiny_models")
    torch.manual_seed(1234)

    roberta = RobertaModel(RobertaConfig(
        vocab_size=len(roberta_tok), hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=64, max_position_embeddings=514, type_vocab_size=1, pad_token_id=1,
        bos_token_id=0, eos_token_id=2,
    ))
    roberta.save_pretrained(root / "roberta")
    roberta_tok.save_pretrained(root / "roberta")

    gpt2 = GPT2Model(GPT2Config(
        vocab_size=len(gpt2_tok), n_embd=32, n_layer=2, n_head=4, n_positions=1024,
    ))
    gpt2.save_pretrained(root / "gpt2")
    gpt2_tok.save_pretrained(root / "gpt2")

    hubert = HubertModel(HubertConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64,
        conv_dim=(16, 16, 16), conv_stride=(5, 4, 2), conv_kernel=(10, 4, 2),
        num_feat_extract_layers=3, num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=4,
        feat_extract_norm="group", do_stable_layer_norm=False, mask_time_prob=0.0,
        layerdrop=0.0,
    ))
    hubert.save_pretrained(root / "hubert")

    return {"roberta": str(root / "roberta"), "gpt2": str(root / "gpt2"), "hubert": str(root / "hubert")}


def write_classification_data(tmp_dir, seed=0, n_interactions=6):
    """Run the release prep on synthetic interactions with real wav files.
    Returns ``{split: jsonl_path}``."""
    from neural_scheduler.backchannel.data import prep

    wav_dir = os.path.join(tmp_dir, "wavs")
    interactions = make_interactions(seed, n_interactions, wav_dir=wav_dir, write_wavs=True, n_turns=10)
    pos, neg, _ = prep.generate_examples(interactions)
    out_dir = os.path.join(tmp_dir, "cls")
    prep.balance_and_write(pos, neg, out_dir, neg_ratio=4.0, seed=0)
    return {s: os.path.join(out_dir, f"{s}.jsonl") for s in ("train", "test")}
