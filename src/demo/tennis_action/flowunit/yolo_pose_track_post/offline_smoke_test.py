#!/usr/bin/env python3
"""Offline smoke test for yolo_pose_track_post."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

_FU = types.ModuleType("_flowunit")
_FU.Buffer = lambda dev, b: ("BUFFER", b)
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
from yolo_pose_track_post import (
    YoloPoseTrackPost, decode_pose, nms_pose, GreedyIoUTracker
)  # noqa: E402


def _make_feat_two_people(W: int = 640) -> np.ndarray:
    """Construct a synthetic [1, 56, N] feature tensor with two clear persons."""
    N = 8400
    feat = np.zeros((1, 56, N), dtype=np.float32)
    # Person 1: bbox center (160, 320), w=h=200, score 0.9, kpts at center.
    feat[0, 0, 0] = 160.0
    feat[0, 1, 0] = 320.0
    feat[0, 2, 0] = 200.0
    feat[0, 3, 0] = 200.0
    feat[0, 4, 0] = 0.9
    for k in range(17):
        feat[0, 5 + 3 * k + 0, 0] = 160.0
        feat[0, 5 + 3 * k + 1, 0] = 320.0
        feat[0, 5 + 3 * k + 2, 0] = 0.8
    # Person 2: bbox center (480, 320).
    feat[0, 0, 1] = 480.0
    feat[0, 1, 1] = 320.0
    feat[0, 2, 1] = 200.0
    feat[0, 3, 1] = 200.0
    feat[0, 4, 1] = 0.85
    for k in range(17):
        feat[0, 5 + 3 * k + 0, 1] = 480.0
        feat[0, 5 + 3 * k + 1, 1] = 320.0
        feat[0, 5 + 3 * k + 2, 1] = 0.8
    return feat


def test_decode_extracts_two_persons():
    feat = _make_feat_two_people()
    persons = decode_pose(feat, conf_threshold=0.25, scale_x=1.0, scale_y=1.0)
    assert len(persons) == 2, f"got {len(persons)} persons"
    print("test_decode_extracts_two_persons: PASS")


def test_nms_keeps_two_distinct_persons():
    feat = _make_feat_two_people()
    persons = decode_pose(feat, conf_threshold=0.25, scale_x=1.0, scale_y=1.0)
    kept = nms_pose(persons, iou_threshold=0.45)
    assert len(kept) == 2, f"NMS kept {len(kept)}"
    print("test_nms_keeps_two_distinct_persons: PASS")


def test_tracker_assigns_stable_ids_across_frames():
    tracker = GreedyIoUTracker(track_iou=0.3, max_lost=20)
    feat = _make_feat_two_people()
    ids_seen = []
    for _ in range(5):
        persons = nms_pose(decode_pose(feat, 0.25, 1.0, 1.0), 0.45)
        tracked = tracker.update(persons)
        ids_seen.append(sorted(t["track_id"] for t in tracked))
    # IDs must be stable across the 5 calls.
    assert all(s == ids_seen[0] for s in ids_seen), f"ID drift: {ids_seen}"
    print("test_tracker_assigns_stable_ids_across_frames: PASS")


def test_tracker_recycles_after_max_lost():
    tracker = GreedyIoUTracker(track_iou=0.3, max_lost=2)
    feat = _make_feat_two_people()
    tracker.update(nms_pose(decode_pose(feat, 0.25, 1.0, 1.0), 0.45))
    for _ in range(5):
        tracker.update([])  # no detections
    # After max_lost+ empty frames, tracks have been pruned.
    assert len(tracker.tracks) == 0, f"{len(tracker.tracks)} tracks remain"
    print("test_tracker_recycles_after_max_lost: PASS")


def test_open_reads_config_defaults():
    class _Cfg:
        def get_int(self, k, d): return d
        def get_float(self, k, d): return d
        def get_string(self, k, d): return d
        def get_bool(self, k, d): return d
    fu = YoloPoseTrackPost()
    rc = fu.open(_Cfg())
    assert rc == _SC.STATUS_SUCCESS
    assert fu.net_h == 640 and fu.net_w == 640
    print("test_open_reads_config_defaults: PASS")


def main() -> int:
    test_decode_extracts_two_persons()
    test_nms_keeps_two_distinct_persons()
    test_tracker_assigns_stable_ids_across_frames()
    test_tracker_recycles_after_max_lost()
    test_open_reads_config_defaults()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
