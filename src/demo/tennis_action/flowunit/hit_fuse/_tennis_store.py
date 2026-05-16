#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
"""_tennis_store — shared in-process session state for the tennis_action pipeline.

All Python flowunits in the same ModelBox process share one Python interpreter,
so this module (once imported) is the same object everywhere.  The pipeline is
single-session (one video at a time), so a simple dict per stream type is safe.

The store is keyed by the *object identity* of the writing flowunit's `self`,
which is constant for the lifetime of one graph execution.  Callers should call
reset(key) from data_pre() so a restarted pipeline starts clean.

Concurrent writes within a single session (multiple threads processing different
frames) are protected by a threading.Lock because the CPython GIL does NOT
protect compound dict operations reliably across threads.
"""

from __future__ import annotations

import threading
from typing import Any

_lock = threading.Lock()

# {writer_key: {str(frame_idx): [cx, cy, peak]}}
_ball: dict[int, dict[str, list[float]]] = {}

# {writer_key: {str(frame_idx): [{track_id, bbox, kpts, score}, ...]}}
_poses: dict[int, dict[str, list[Any]]] = {}

# Per-session dropped-hits list. hit_fuse writes once in data_post; action_sink
# reads in its own data_post. Single producer / single consumer, so no per-key
# segregation needed — a list under one canonical key suffices.
_dropped: list[dict[str, Any]] = []


def reset_ball(key: int) -> None:
    with _lock:
        _ball[key] = {}


def reset_poses(key: int) -> None:
    with _lock:
        _poses[key] = {}


def write_ball(key: int, frame_idx: int, cx: float, cy: float, peak: float) -> None:
    with _lock:
        if key not in _ball:
            _ball[key] = {}
        _ball[key][str(frame_idx)] = [cx, cy, peak]


def write_poses(key: int, frame_idx: int, tracks: list[Any]) -> None:
    with _lock:
        if key not in _poses:
            _poses[key] = {}
        _poses[key][str(frame_idx)] = tracks


def read_ball_map(key: int) -> dict[int, tuple[float, float, float]]:
    """Return {frame_idx (int): (cx, cy, peak)} snapshot for `key`."""
    with _lock:
        raw = dict(_ball.get(key, {}))
    return {int(k): (float(v[0]), float(v[1]), float(v[2])) for k, v in raw.items()}


def read_poses_map(key: int) -> dict[int, list[Any]]:
    """Return {frame_idx (int): [track, ...]} snapshot for `key`."""
    with _lock:
        raw = dict(_poses.get(key, {}))
    return {int(k): v for k, v in raw.items()}


def get_all_ball() -> dict[int, tuple[float, float, float]]:
    """Return combined ball map across all keys (for single-session pipelines)."""
    combined: dict[int, tuple[float, float, float]] = {}
    with _lock:
        for per_key in _ball.values():
            for fi_str, triple in per_key.items():
                combined[int(fi_str)] = (float(triple[0]), float(triple[1]), float(triple[2]))
    return combined


def get_all_poses() -> dict[int, list[Any]]:
    """Return combined poses map across all keys (for single-session pipelines)."""
    combined: dict[int, list[Any]] = {}
    with _lock:
        for per_key in _poses.values():
            for fi_str, tracks in per_key.items():
                combined[int(fi_str)] = tracks
    return combined


def write_dropped(dropped: list[dict[str, Any]]) -> None:
    """Replace the per-session dropped-hits list (hit_fuse calls in data_post)."""
    global _dropped
    with _lock:
        _dropped = list(dropped)


def get_all_dropped() -> list[dict[str, Any]]:
    """Snapshot the dropped-hits list (action_sink reads in data_post)."""
    with _lock:
        return list(_dropped)


def reset_dropped() -> None:
    global _dropped
    with _lock:
        _dropped = []


def clear_all() -> None:
    """Wipe all state (called by hit_fuse / action_sink in data_pre)."""
    global _dropped
    with _lock:
        _ball.clear()
        _poses.clear()
        _dropped = []
