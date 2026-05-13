#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
# Helpers for the hit_sound_detect flowunit: audio loading, mel-spectrogram
# image, sliding-window postprocess. Logic ported from
# models/tennis_audio_seg/{infer_helpers,postprocess}.py.
#
"""Self-contained helpers so the flowunit has no path-sensitive imports."""

from __future__ import annotations

import collections
from typing import Dict, List, Optional, Tuple

import av
import cv2
import librosa
import numpy as np


def load_audio_mono(video_path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    """Decode mono audio from a video file at target_sr."""
    container = av.open(video_path)
    audio_stream = next((s for s in container.streams if s.type == "audio"), None)
    if audio_stream is None:
        container.close()
        raise RuntimeError(f"No audio stream found in {video_path}")
    resampler = av.AudioResampler(format="flt", layout="mono", rate=target_sr)
    samples: List[np.ndarray] = []
    for frame in container.decode(audio=0):
        for out in resampler.resample(frame):
            samples.append(out.to_ndarray().reshape(-1).astype(np.float32))
    # flush
    for out in resampler.resample(None):
        samples.append(out.to_ndarray().reshape(-1).astype(np.float32))
    container.close()
    if not samples:
        raise RuntimeError("Failed to decode audio.")
    return np.concatenate(samples, axis=0).astype(np.float32, copy=False), target_sr


def mel_gray_image(audio_segment: np.ndarray, sr: int, n_fft: int, hop: int, n_mels: int) -> np.ndarray:
    """Same normalization as training: mel power → db → min-max → uint8."""
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


def cut_audio(audio: np.ndarray, sr: int, start_t: float, length_s: float) -> np.ndarray:
    """Slice with zero-padding at boundaries."""
    start_idx = int(round(start_t * sr))
    length = int(round(length_s * sr))
    end_idx = start_idx + length
    pad_left = 0
    if start_idx < 0:
        pad_left = -start_idx
        start_idx = 0
    pad_right = 0
    if end_idx > len(audio):
        pad_right = end_idx - len(audio)
        end_idx = len(audio)
    seg = audio[start_idx:end_idx]
    if pad_left or pad_right:
        seg = np.pad(seg, (pad_left, pad_right), mode="constant")
    if len(seg) < length:
        seg = np.pad(seg, (0, length - len(seg)), mode="constant")
    return seg[:length]


class OutputSmoother:
    """Majority-vote smoothing over a fixed window."""

    def __init__(self, window_size: int = 5):
        self.window_size = max(1, int(window_size))
        self.buf: collections.deque = collections.deque(maxlen=self.window_size)

    def smooth(self, x: int) -> int:
        self.buf.append(int(x))
        return 1 if sum(self.buf) * 2 > len(self.buf) else 0


class DurationFilter:
    """Suppresses positive runs longer than max_duration seconds."""

    def __init__(self, max_duration: float = 1.0):
        self.max_duration = float(max_duration)
        self.run_start: Optional[float] = None

    def filter(self, t: float, x: int) -> int:
        if x == 1:
            if self.run_start is None:
                self.run_start = t
            if t - self.run_start > self.max_duration:
                return 0
            return 1
        self.run_start = None
        return 0


class BufferedRallyDetector:
    """Rally state with a buffer: a hit extends the rally for up to
    max_inter_hit_time, and the state is finalized after buffer_time seconds
    of silence."""

    def __init__(self, max_inter_hit_time: float = 3.0, buffer_time: float = 4.0, step: float = 0.02):
        if buffer_time <= max_inter_hit_time:
            raise ValueError("buffer_time must be > max_inter_hit_time")
        self.max_inter_hit_time = float(max_inter_hit_time)
        self.buffer_time = float(buffer_time)
        self.step = float(step)
        self.history: List[Tuple[float, int]] = []  # (t, hit)
        self.state_history: List[Tuple[float, int]] = []
        self.last_hit_t: Optional[float] = None
        self.in_rally = False

    def process(self, t: float, hit: int) -> None:
        if hit == 1:
            self.last_hit_t = t
            self.in_rally = True
        elif self.in_rally and self.last_hit_t is not None and (t - self.last_hit_t) > self.max_inter_hit_time:
            self.in_rally = False
        self.history.append((t, int(hit)))
        self.state_history.append((t, 1 if self.in_rally else 0))

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.state_history:
            return np.array([]), np.array([])
        ts = np.array([t for t, _ in self.state_history])
        states = np.array([s for _, s in self.state_history], dtype=np.int32)
        return ts, states


def merge_positive_windows(windows: List[Tuple[float, float]], join_gap: float) -> List[Tuple[float, float]]:
    """Merge overlapping or near-adjacent (start, end) windows."""
    if not windows:
        return []
    windows = sorted(windows)
    merged: List[Tuple[float, float]] = [windows[0]]
    for s, e in windows[1:]:
        ms, me = merged[-1]
        if s - me <= join_gap + 1e-9:
            merged[-1] = (ms, max(me, e))
        else:
            merged.append((s, e))
    return merged


def write_overlay_video(
    video_path: str,
    output_path: str,
    selected_intervals: List[Tuple[float, float]],
) -> None:
    """Reads frames from video_path, overlays 'kickball'/'no' label, writes mp4."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for overlay: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open writer: {output_path}")
    interval_idx = 0
    frame_idx = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t = frame_idx / fps if fps > 1e-3 else 0.0
            while interval_idx < len(selected_intervals) and t > selected_intervals[interval_idx][1]:
                interval_idx += 1
            in_iv = False
            if interval_idx < len(selected_intervals):
                s, e = selected_intervals[interval_idx]
                in_iv = s <= t <= e
            cv2.putText(
                frame,
                "kickball" if in_iv else "no",
                (10, 35),
                font,
                1.0,
                (0, 200, 0) if in_iv else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()


def get_audio_start_offset(video_path: str) -> float:
    """Best-effort audio stream start_time (seconds). 0 on any failure."""
    try:
        container = av.open(video_path)
        for s in container.streams:
            if s.type == "audio":
                offset = float(s.start_time * s.time_base) if s.start_time is not None else 0.0
                container.close()
                return offset
        container.close()
    except Exception:
        pass
    return 0.0
