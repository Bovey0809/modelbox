#!/usr/bin/env python3
"""Precompute the sliding mel-spectrogram windows from a video and save them
as a binary file consumable by the hit_audio_source flowunit.

The binary layout (little-endian) is:
  bytes  0..3   : magic "MELS"
  bytes  4..7   : uint32 count        (# windows)
  bytes  8..11  : uint32 mel_h        (128)
  bytes 12..15  : uint32 mel_w        (48 for the bundled tennis model)
  bytes 16..    : count * mel_h * mel_w * float32, row-major

The values are the exact (img/255 - 0.5)/0.5 normalization used at training
time, so they can be fed straight into hit_sound_infer_coreml.

Run this once per video, then pass the produced .bin path to the
hit_audio_source flowunit's `mel_bin_path` option.

Example:
  python precompute_mels.py \\
    --video /Users/houbowei/Desktop/tennis/2026-01-04-action-06.mp4 \\
    --output /tmp/hit_sound_mels.bin
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import av
import librosa
import numpy as np


def load_audio_mono(video_path: str, target_sr: int) -> np.ndarray:
    container = av.open(video_path)
    audio = next((s for s in container.streams if s.type == "audio"), None)
    if audio is None:
        container.close()
        raise RuntimeError(f"no audio stream in {video_path}")
    resampler = av.AudioResampler(format="flt", layout="mono", rate=target_sr)
    parts = []
    for frame in container.decode(audio=0):
        for out in resampler.resample(frame):
            parts.append(out.to_ndarray().reshape(-1).astype(np.float32))
    for out in resampler.resample(None):
        parts.append(out.to_ndarray().reshape(-1).astype(np.float32))
    container.close()
    if not parts:
        raise RuntimeError("no audio decoded")
    return np.concatenate(parts).astype(np.float32, copy=False)


def cut(audio: np.ndarray, sr: int, t: float, length_s: float) -> np.ndarray:
    s = int(round(t * sr))
    L = int(round(length_s * sr))
    e = s + L
    pad_l = max(0, -s); s = max(0, s)
    pad_r = max(0, e - len(audio)); e = min(len(audio), e)
    seg = audio[s:e]
    if pad_l or pad_r:
        seg = np.pad(seg, (pad_l, pad_r))
    if len(seg) < L:
        seg = np.pad(seg, (0, L - len(seg)))
    return seg[:L]


def mel_gray(audio_segment: np.ndarray, sr: int,
             n_fft: int, hop: int, n_mels: int) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=audio_segment, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    s_min = float(np.min(S_db))
    s_max = float(np.max(S_db))
    if s_max - s_min < 1e-8:
        norm = np.zeros_like(S_db, dtype=np.float32)
    else:
        norm = (S_db - s_min) / (s_max - s_min)
    return (norm * 255.0).astype(np.uint8)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--target_sr", type=int, default=16000)
    p.add_argument("--window_sec", type=float, default=0.5)
    p.add_argument("--step_sec", type=float, default=0.02)
    p.add_argument("--n_fft", type=int, default=512)
    p.add_argument("--hop_length", type=int, default=170)
    p.add_argument("--n_mels", type=int, default=128)
    p.add_argument("--meta", default="",
                   help="optional sibling json with all params + window count")
    args = p.parse_args()

    audio = load_audio_mono(args.video, target_sr=args.target_sr)
    duration = len(audio) / args.target_sr
    times = np.arange(0.0, max(0.0, duration - args.window_sec) + 1e-9,
                      args.step_sec)
    if len(times) == 0:
        print(f"refusing to write empty mels.bin (duration={duration:.3f}s)",
              file=sys.stderr)
        return 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine output mel shape from the first window.
    seg0 = cut(audio, args.target_sr, float(times[0]), args.window_sec)
    img0 = mel_gray(seg0, args.target_sr, args.n_fft, args.hop_length, args.n_mels)
    h, w = img0.shape
    count = len(times)
    print(f"writing {count} windows of shape (1,1,{h},{w}) to {out_path}")

    with open(out_path, "wb") as f:
        f.write(b"MELS")
        f.write(struct.pack("<III", count, h, w))
        # First window
        arr = img0.astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        f.write(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
        # Remaining
        for t in times[1:]:
            seg = cut(audio, args.target_sr, float(t), args.window_sec)
            img = mel_gray(seg, args.target_sr, args.n_fft, args.hop_length,
                           args.n_mels)
            arr = img.astype(np.float32) / 255.0
            arr = (arr - 0.5) / 0.5
            f.write(np.ascontiguousarray(arr, dtype=np.float32).tobytes())

    if args.meta:
        Path(args.meta).write_text(json.dumps({
            "video": args.video,
            "output": str(out_path),
            "count": count,
            "mel_h": h,
            "mel_w": w,
            "target_sr": args.target_sr,
            "window_sec": args.window_sec,
            "step_sec": args.step_sec,
            "n_fft": args.n_fft,
            "hop_length": args.hop_length,
            "n_mels": args.n_mels,
            "duration_s": float(duration),
        }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
