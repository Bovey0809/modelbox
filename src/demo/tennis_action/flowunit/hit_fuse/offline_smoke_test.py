#!/usr/bin/env python3
"""Offline smoke test for hit_fuse. Builds incrementally — Tasks 6a–6d each
extend this file."""

from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

_FU = types.ModuleType("_flowunit")

class _Buf:
    def __init__(self, dev, payload):
        self._payload = payload
        self._meta: dict = {}
    def as_object(self):
        return self._payload
    def set(self, k, v):
        self._meta[k] = v
_FU.Buffer = _Buf

class _Port:
    def __init__(self):
        self._items: list = []
    def push_back(self, b):
        self._items.append(b)
    def __iter__(self):
        return iter(self._items)

class _DC:
    def __init__(self, inputs: dict):
        self._inputs = inputs
        self._outputs: dict = {}
    def input(self, name):
        return self._inputs[name]
    def output(self, name):
        if name not in self._outputs:
            self._outputs[name] = _Port()
        return self._outputs[name]

class _SC: STATUS_SUCCESS = 0; STATUS_FAULT = 1
class _Status:
    StatusCode = _SC
    def __init__(self, *a, **k): pass
_FU.Status = _Status
_FU.FlowUnit = type("FlowUnit", (), {"__init__": lambda self: None,
                                     "get_bind_device": lambda self: None})
_FU.info = lambda msg: None
_FU.error = lambda msg: print(f"[ERR] {msg}", file=sys.stderr)
sys.modules["_flowunit"] = _FU

sys.path.insert(0, str(Path(__file__).parent))
from hit_fuse import (  # noqa: E402
    cross_confirm, assign_hitter, assemble_window, HitFuse
)


# --- Cross-confirm tests (Task 6a) ---

def _make_ball_track(frames: int, traj: dict[int, tuple[float, float]],
                     default_peak: float = 0.8) -> dict[int, tuple[float, float, float]]:
    """{frame_idx: (cx, cy, peak)}. Missing frames default to sentinel."""
    return {f: (traj[f][0], traj[f][1], default_peak) if f in traj
            else (-1.0, -1.0, 0.0) for f in range(frames)}


def test_cross_confirm_direction_flip_accepts():
    # Ball moving +x for 6 frames, then -x for 6 frames; hit at frame 50.
    # Construct positions so v_pre = (+x_dir) and v_post = (-x_dir) cleanly.
    ball_full: dict[int, tuple[float, float, float]] = {}
    for f in range(40, 60):
        if f <= 50:
            x = float(f - 40) * 10  # +10 per frame moving right
        else:
            x = float(50 - 40) * 10 - float(f - 50) * 10  # then -10 per frame
        ball_full[f] = (x, 0.0, 0.8)
    # Fill missing frames with sentinel.
    for f in list(range(0, 40)) + list(range(60, 100)):
        ball_full[f] = (-1.0, -1.0, 0.0)
    ok, reason = cross_confirm(ball_full, hit_frame=50, require_visual_confirm=True)
    assert ok, f"reason={reason}"
    print("test_cross_confirm_direction_flip_accepts: PASS")


def test_cross_confirm_no_flip_rejects():
    # Ball moving +x monotonically across the hit.
    ball_full = {f: (float(f), 0.0, 0.8) for f in range(40, 60)}
    for f in list(range(0, 40)) + list(range(60, 100)):
        ball_full[f] = (-1.0, -1.0, 0.0)
    ok, reason = cross_confirm(ball_full, hit_frame=50, require_visual_confirm=True)
    assert not ok
    assert reason == "trajectory_not_consistent", reason
    print("test_cross_confirm_no_flip_rejects: PASS")


def test_cross_confirm_missing_ball_with_require_drops():
    ball_full = {f: (-1.0, -1.0, 0.0) for f in range(100)}
    ok, reason = cross_confirm(ball_full, hit_frame=50, require_visual_confirm=True)
    assert not ok
    assert reason == "no_ball_near_hit", reason
    print("test_cross_confirm_missing_ball_with_require_drops: PASS")


def test_cross_confirm_missing_ball_without_require_accepts():
    ball_full = {f: (-1.0, -1.0, 0.0) for f in range(100)}
    ok, reason = cross_confirm(ball_full, hit_frame=50, require_visual_confirm=False)
    assert ok and reason == "visual_skipped", reason
    print("test_cross_confirm_missing_ball_without_require_accepts: PASS")


def main() -> int:
    test_cross_confirm_direction_flip_accepts()
    test_cross_confirm_no_flip_rejects()
    test_cross_confirm_missing_ball_with_require_drops()
    test_cross_confirm_missing_ball_without_require_accepts()
    # More tests appended in tasks 6b–6d.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
