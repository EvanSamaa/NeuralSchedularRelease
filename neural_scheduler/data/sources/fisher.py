"""Fisher English conversations: two-channel telephone audio at 8 kHz.

Two interchangeable sources are provided:

* :class:`FisherLMDBSource` reads an LMDB with one record per conversation.
  This is the path the released Fisher token data was produced with.
* :class:`FisherDirectorySource` reads stereo audio files from a directory, so
  the pipeline can be run from raw audio without building an LMDB.

Both resample to 16 kHz with ``torchaudio.transforms.Resample`` and index
conversations in sorted file-name order, so ``source_file`` indices agree.

LMDB record format (key = file name, e.g. ``fe_03_00001.wav``)::

    4 bytes   big-endian length N of the metadata
    N bytes   UTF-8 JSON {"sr": int, "shape": [channels, samples], "dtype": "float32"}
    rest      raw PCM, channels-first

Run ``python -m neural_scheduler.data.sources.fisher --help`` to build an LMDB
from a directory of stereo files.
"""

import argparse
import json
import os
import struct
from typing import Dict, List

import numpy as np
import torch

from ...constants import SPEECH_TOKENIZER_SR

AUDIO_EXTENSIONS = (".wav", ".flac", ".sph")
_DTYPES = {"float32": np.float32, "int16": np.int16, "float64": np.float64}


def resample_wav_tensor(wav: torch.Tensor, input_sr: int, output_sr: int) -> torch.Tensor:
    """Resample ``[channels, time]`` audio with torchaudio's default sinc resampler."""
    if input_sr == output_sr:
        return wav
    import torchaudio.transforms as T

    return T.Resample(orig_freq=input_sr, new_freq=output_sr)(wav)


def _split_channels(wav: torch.Tensor, sr: int) -> Dict:
    if wav.shape[0] < 2:
        left = wav[0]
        right = left.clone()
    else:
        left, right = wav[0], wav[1]
    return {"left": left, "right": right, "sr": sr}


def list_audio_files(audio_dir: str) -> List[str]:
    """Sorted audio file names directly inside ``audio_dir``."""
    return sorted(f for f in os.listdir(audio_dir) if f.lower().endswith(AUDIO_EXTENSIONS))


def read_stereo_file(path: str):
    """Read an audio file as float32 ``[channels, samples]`` at its native rate."""
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return np.ascontiguousarray(data.T), sr


class FisherLMDBSource:
    """Conversations stored in an LMDB (see the module docstring for the format)."""

    def __init__(self, lmdb_path: str, keys_path: str, output_sr: int = SPEECH_TOKENIZER_SR):
        self.lmdb_path = lmdb_path
        self.output_sr = output_sr
        with open(keys_path, "r") as f:
            self.keys = [line.strip() for line in f]
        self._env = None
        self._txn = None

    def __len__(self) -> int:
        return len(self.keys)

    def name(self, idx: int) -> str:
        return self.keys[idx]

    def _read_raw(self, idx: int):
        if self._env is None:
            import lmdb

            self._env = lmdb.open(self.lmdb_path, readonly=True, lock=False, meminit=False)
            self._txn = self._env.begin(write=False)
        key = self.keys[idx].encode("ascii")
        value = self._txn.get(key)
        if value is None:
            raise KeyError(f"Key {self.keys[idx]} not found in LMDB at index {idx}.")
        metadata_len = struct.unpack(">I", value[:4])[0]
        metadata = json.loads(value[4:4 + metadata_len].decode("utf-8"))
        dtype = _DTYPES.get(metadata["dtype"], np.float32)
        wav = np.frombuffer(value[4 + metadata_len:], dtype=dtype).reshape(metadata["shape"])
        return torch.from_numpy(wav), metadata["sr"]

    def load(self, idx: int) -> Dict:
        """Return ``{"left", "right", "sr"}`` for conversation ``idx``, resampled to ``output_sr``."""
        wav, sr = self._read_raw(idx)
        wav = resample_wav_tensor(wav, sr, self.output_sr)
        return _split_channels(wav, self.output_sr)


class FisherDirectorySource:
    """Conversations read from stereo audio files in one directory."""

    def __init__(self, audio_dir: str, output_sr: int = SPEECH_TOKENIZER_SR):
        self.audio_dir = audio_dir
        self.output_sr = output_sr
        self.keys = list_audio_files(audio_dir)

    def __len__(self) -> int:
        return len(self.keys)

    def name(self, idx: int) -> str:
        return self.keys[idx]

    def load(self, idx: int) -> Dict:
        """Return ``{"left", "right", "sr"}`` for conversation ``idx``, resampled to ``output_sr``."""
        data, sr = read_stereo_file(os.path.join(self.audio_dir, self.keys[idx]))
        wav = resample_wav_tensor(torch.from_numpy(data), sr, self.output_sr)
        return _split_channels(wav, self.output_sr)


def build_fisher_lmdb(audio_dir: str, lmdb_path: str, keys_path: str, map_size_gb: float = 400.0) -> int:
    """Write every audio file of ``audio_dir`` into an LMDB plus a key list.

    Audio is stored at its native sample rate as float32. Returns the number of
    conversations written.
    """
    import lmdb

    names = list_audio_files(audio_dir)
    env = lmdb.open(lmdb_path, map_size=int(map_size_gb * (1 << 30)))
    for name in names:
        data, sr = read_stereo_file(os.path.join(audio_dir, name))
        metadata = json.dumps({"sr": sr, "shape": list(data.shape), "dtype": "float32"}).encode("utf-8")
        with env.begin(write=True) as txn:
            txn.put(name.encode("ascii"), struct.pack(">I", len(metadata)) + metadata + data.tobytes())
    env.close()
    with open(keys_path, "w") as f:
        for name in names:
            f.write(name + "\n")
    return len(names)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the Fisher LMDB from a directory of stereo audio files.")
    parser.add_argument("--audio_dir", required=True, help="Directory with one stereo file per conversation")
    parser.add_argument("--lmdb_path", required=True)
    parser.add_argument("--keys_path", required=True)
    parser.add_argument("--map_size_gb", type=float, default=400.0)
    args = parser.parse_args(argv)
    count = build_fisher_lmdb(args.audio_dir, args.lmdb_path, args.keys_path, args.map_size_gb)
    print(f"Wrote {count} conversations to {args.lmdb_path}")


if __name__ == "__main__":
    main()
