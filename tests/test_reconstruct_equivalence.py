"""Audio reconstruction must copy exactly the samples the original step 4 copies."""

import json
import random

import numpy as np
import pytest
import soundfile as sf

from neural_scheduler.constants import SAMPLES_PER_TOKEN, SILENCE_CODE
from neural_scheduler.pipeline import reconstruct

from .test_scheduler_helpers import ORIG_RECONSTRUCT, load_module

pytestmark = pytest.mark.equivalence


@pytest.fixture(scope="module")
def orig(ntpp_orig):
    return load_module(ntpp_orig / ORIG_RECONSTRUCT, "orig_reconstruct")


def write_source_map(directory, rng: random.Random, suffix: str):
    """Random source-map files for one channel; returns the script's codec ids."""
    num_raw = rng.randint(20, 60)
    # A few repeated codes so that order-preserving matching matters.
    raw_tokens = [rng.choice([SILENCE_CODE] + list(range(12))) for _ in range(num_raw)]
    # The audio may end inside the last token (zero-padded slice) or before it.
    num_samples = num_raw * SAMPLES_PER_TOKEN - rng.choice([0, 0, 400, SAMPLES_PER_TOKEN + 10])
    audio = np.random.default_rng(rng.randint(0, 10**6)).uniform(-0.5, 0.5, num_samples).astype(np.float32)
    sf.write(str(directory / f"concatenated_audio_{suffix}.wav"), audio, 16000, subtype="FLOAT")

    script_tokens, source_map = [], []
    for raw_idx, code in enumerate(raw_tokens):
        if rng.random() < 0.15:
            script_tokens.append(999999)  # a marker
            source_map.append(None)
        if code != SILENCE_CODE and rng.random() < 0.9:
            script_tokens.append(152353 + code)
            source_map.append(raw_idx)
    for name, data in (("raw_tokens", raw_tokens), ("script_tokens", script_tokens),
                       ("script_source_map", source_map)):
        (directory / f"{name}_{suffix}.json").write_text(json.dumps(data))
    return [raw_tokens[i] for i in source_map if i is not None]


def make_schedule(rng: random.Random, script_codes):
    """The script's codes with silence inserted, plus out-of-order and unknown tokens."""
    schedule = []
    for code in script_codes:
        schedule.extend([SILENCE_CODE] * rng.choice([0, 0, 1, 3]))
        schedule.append(code)
    for _ in range(rng.randint(0, 4)):
        schedule.insert(rng.randint(0, len(schedule)), rng.choice(script_codes + [5000]) if script_codes else 5000)
    return schedule


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("output_sr", [16000, 22050])
def test_reconstruction_matches_original(seed, output_sr, tmp_path, orig):
    rng = random.Random(seed)
    cache = tmp_path / "cache"
    source_map_dir = cache / "source_map"
    source_map_dir.mkdir(parents=True)
    schedule_a = make_schedule(rng, write_source_map(source_map_dir, rng, "a"))
    schedule_b = make_schedule(rng, write_source_map(source_map_dir, rng, "b"))

    new_a, new_b = reconstruct.reconstruct_dialogue_from_cache(
        schedule_a, schedule_b, str(cache), output_dir=str(tmp_path / "new"), output_sr=output_sr)
    orig_a, orig_b = orig.reconstruct_dialogue_from_cache(
        schedule_a, schedule_b, str(cache), output_dir=str(tmp_path / "orig"), output_sr=output_sr)

    np.testing.assert_array_equal(new_a, orig_a)
    np.testing.assert_array_equal(new_b, orig_b)
    for name in ("channel_a.wav", "channel_b.wav", "channel_sum.wav"):
        new_audio, new_sr = sf.read(str(tmp_path / "new" / name))
        orig_audio, orig_sr = sf.read(str(tmp_path / "orig" / name))
        assert new_sr == orig_sr == output_sr
        np.testing.assert_array_equal(new_audio, orig_audio)


def test_missing_source_map_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        reconstruct.reconstruct_dialogue_from_cache([1], [2], str(tmp_path))
