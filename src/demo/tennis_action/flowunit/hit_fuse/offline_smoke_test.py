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


# --- Hitter assignment tests (Task 6b) ---

def _person(tid: int, bbox: tuple[float, float, float, float],
            wrist_l: tuple[float, float], wrist_r: tuple[float, float]) -> dict:
    """Build a tracked-pose dict mirroring the yolo_pose_track_post format."""
    kpts = [[0.0, 0.0, 0.0]] * 17
    kpts[9] = [wrist_l[0], wrist_l[1], 0.9]   # COCO L wrist
    kpts[10] = [wrist_r[0], wrist_r[1], 0.9]  # COCO R wrist
    return {"track_id": tid, "bbox": list(bbox), "score": 0.9, "kpts": kpts}


def test_assign_hitter_picks_nearest_wrist():
    poses = [
        _person(7, (50, 50, 200, 300), wrist_l=(100, 100), wrist_r=(180, 200)),
        _person(8, (500, 50, 700, 300), wrist_l=(550, 100), wrist_r=(680, 200)),
    ]
    pose_map = {50: poses}
    ball = (105, 103, 0.9)
    info = assign_hitter(pose_map, ball, hit_frame=50)
    assert info is not None
    assert info["track_id"] == 7, info
    assert not info["tie_resolved"]
    print("test_assign_hitter_picks_nearest_wrist: PASS")


def test_assign_hitter_no_player_returns_none():
    info = assign_hitter({}, (100, 100, 0.9), hit_frame=50)
    assert info is None
    print("test_assign_hitter_no_player_returns_none: PASS")


def test_assign_hitter_tie_break_by_track_id():
    # Both tracks have wrists at the same position -> tied at 0.0 distance.
    # Both bboxes contain the ball-box (ball_box = [140..170 × 140..170]):
    # track 2's bbox is [100..200, 100..200], track 1's is [125..200, 100..200].
    # Their IoUs with ball_box are identical (ball_box fully inside both).
    # So lower track_id (1) wins.
    poses = [
        _person(2, (100, 100, 200, 200), wrist_l=(155, 155), wrist_r=(160, 160)),
        _person(1, (125, 100, 200, 200), wrist_l=(155, 155), wrist_r=(160, 160)),
    ]
    info = assign_hitter({50: poses}, (155, 155, 0.9), hit_frame=50)
    assert info is not None and info["track_id"] == 1, info
    assert info["tie_resolved"], "should have flipped tie_resolved"
    print("test_assign_hitter_tie_break_by_track_id: PASS")


def test_assign_hitter_uses_nearby_frame_when_target_empty():
    poses = [_person(7, (50, 50, 200, 300), (100, 100), (180, 200))]
    # Pose data is at frame 48, but the hit is at frame 50.
    info = assign_hitter({48: poses}, (105, 103, 0.9), hit_frame=50)
    assert info is not None and info["track_id"] == 7, info
    assert info["frame_used"] == 48, f"frame_used={info['frame_used']}"
    print("test_assign_hitter_uses_nearby_frame_when_target_empty: PASS")


def main() -> int:
    test_cross_confirm_direction_flip_accepts()
    test_cross_confirm_no_flip_rejects()
    test_cross_confirm_missing_ball_with_require_drops()
    test_cross_confirm_missing_ball_without_require_accepts()
    test_assign_hitter_picks_nearest_wrist()
    test_assign_hitter_no_player_returns_none()
    test_assign_hitter_tie_break_by_track_id()
    test_assign_hitter_uses_nearby_frame_when_target_empty()
    # More tests appended in tasks 6b–6d.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
