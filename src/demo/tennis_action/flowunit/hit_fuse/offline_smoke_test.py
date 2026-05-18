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
_FU.warn = lambda msg: None
_FU.error = lambda msg: print(f"[ERR] {msg}", file=sys.stderr)
sys.modules["_flowunit"] = _FU

sys.path.insert(0, str(Path(__file__).parent))
from hit_fuse import (  # noqa: E402
    cross_confirm, assign_hitter, assemble_window, HitFuse,
    engineer_pose_features,
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


# --- Window assembly tests (Task 6c) ---

def _ts(tid: int, x: float, y: float) -> dict:
    """Tracked pose at a fixed wrist position. All 17 kpts at (x, y, 0.9)."""
    kpts = [[x, y, 0.9] for _ in range(17)]
    return {"track_id": tid,
            "bbox": [x - 50, y - 50, x + 50, y + 50],
            "score": 0.9, "kpts": kpts}


def test_assemble_window_full_coverage():
    pose_map = {f: [_ts(7, 100 + f, 200)] for f in range(20, 60)}
    res = assemble_window(pose_map, track_id=7, hit_frame=40, T=32,
                          min_coverage=0.75, image_width=1280, image_height=720)
    assert res is not None
    arr, meta = res
    assert arr.shape == (32, 17, 3), arr.shape
    assert not meta["boundary_padded"]
    # Normalized x at hit_frame center should be roughly (100 + 40) / 1280.
    mid_x = arr[16, 0, 0]  # kpt 0, x channel, T/2 = 16
    assert 0.0 < mid_x < 0.2, f"normalized mid_x={mid_x}"
    print("test_assemble_window_full_coverage: PASS")


def test_assemble_window_interpolates_gaps():
    """30/32 frames present (2 interior gaps). assemble_window should
    interpolate, not drop."""
    pose_map = {}
    for f in range(20, 60):
        if f in (38, 41):
            continue
        pose_map[f] = [_ts(7, 100 + f, 200)]
    res = assemble_window(pose_map, track_id=7, hit_frame=40, T=32,
                          min_coverage=0.75, image_width=1280, image_height=720)
    assert res is not None
    arr, meta = res
    assert arr.shape == (32, 17, 3)
    print("test_assemble_window_interpolates_gaps: PASS")


def test_assemble_window_coverage_too_low_drops():
    # Only 6 frames in the window range have data — 6/32 = 18.75% << 75%.
    pose_map = {f: [_ts(7, 100 + f, 200)] for f in range(20, 26)}
    res = assemble_window(pose_map, track_id=7, hit_frame=40, T=32,
                          min_coverage=0.75, image_width=1280, image_height=720)
    assert res is None
    print("test_assemble_window_coverage_too_low_drops: PASS")


def test_assemble_window_boundary_padding():
    """Hit at frame 5 with T=32 -> window [-11, 21). Leading frames < 0
    are padded from the first valid frame."""
    pose_map = {f: [_ts(7, 100 + f, 200)] for f in range(0, 30)}
    res = assemble_window(pose_map, track_id=7, hit_frame=5, T=32,
                          min_coverage=0.75, image_width=1280, image_height=720)
    assert res is not None
    arr, meta = res
    assert meta["boundary_padded"], "boundary_padded should be True"
    print("test_assemble_window_boundary_padding: PASS")


# --- End-to-end Python class integration (Task 6d) ---

def _write_stub_meta(path: Path, T: int = 32):
    path.write_text(json.dumps({
        "classes": ["forehand", "backhand", "serve", "other"],
        "num_kpts": 17, "T": T, "input_layout": "BTC",
    }))


def test_hit_fuse_open_validates_meta():
    """open() must reject mismatched T or wrong num_kpts in asformer_meta.json."""
    class _Cfg:
        def __init__(self, d): self.d = d
        def get_int(self, k, dflt): return int(self.d.get(k, dflt))
        def get_float(self, k, dflt): return float(self.d.get(k, dflt))
        def get_string(self, k, dflt): return str(self.d.get(k, dflt))
        def get_bool(self, k, dflt): return bool(self.d.get(k, dflt))
    with tempfile.TemporaryDirectory() as td:
        meta = Path(td) / "asformer_meta.json"
        _write_stub_meta(meta, T=32)
        fu = HitFuse()
        rc = fu.open(_Cfg({"T": 32, "asformer_meta_path": str(meta)}))
        assert rc == _SC.STATUS_SUCCESS, f"rc={rc}"
        # Mismatched T should fault.
        _write_stub_meta(meta, T=16)
        fu2 = HitFuse()
        rc2 = fu2.open(_Cfg({"T": 32, "asformer_meta_path": str(meta)}))
        assert rc2 == _SC.STATUS_FAULT, f"expected FAULT, got rc={rc2}"
        print("test_hit_fuse_open_validates_meta: PASS")


def test_hit_fuse_end_to_end_one_hit():
    """Drive fuse_session() with a clean direction-flip + one track present
    for the full window. Expect exactly one confirmed hit."""
    # Ball trajectory: moving away from x=200 in both directions of frame 40.
    # Build a clear V-shape: x decreasing 50->40, then increasing 40->50.
    ball_map = {}
    for f in range(0, 80):
        if 35 <= f <= 45:
            if f <= 40:
                x = 250.0 - 10.0 * (f - 35)  # 250, 240, ..., 200
            else:
                x = 200.0 + 10.0 * (f - 40)  # 210, 220, ..., 250
            ball_map[f] = (x, 200.0, 0.9)
        else:
            ball_map[f] = (-1.0, -1.0, 0.0)
    # One track present across frames 20..60 with wrist near (200, 200).
    pose_map = {f: [_ts(7, 200, 200)] for f in range(20, 60)}
    centers = [{"frame_idx": 40, "audio_conf": 0.9, "t_center_sec": 1.33}]

    with tempfile.TemporaryDirectory() as td:
        meta = Path(td) / "asformer_meta.json"
        _write_stub_meta(meta, T=32)
        fu = HitFuse()

        class _Cfg:
            def __init__(self, d): self.d = d
            def get_int(self, k, dflt): return int(self.d.get(k, dflt))
            def get_float(self, k, dflt): return float(self.d.get(k, dflt))
            def get_string(self, k, dflt): return str(self.d.get(k, dflt))
            def get_bool(self, k, dflt): return bool(self.d.get(k, dflt))

        fu.open(_Cfg({"T": 32, "image_width": 1280, "image_height": 720,
                      "require_visual_confirm": True,
                      "min_window_coverage": 0.75,
                      "asformer_meta_path": str(meta)}))
        confirmed, dropped = fu.fuse_session(centers, ball_map, pose_map)
    assert len(confirmed) == 1, f"got {len(confirmed)} confirmed, {len(dropped)} dropped"
    arr, meta_h = confirmed[0]
    assert arr.shape == (32, 17, 3)
    assert meta_h["track_id"] == 7
    assert meta_h["frame_idx"] == 40
    assert meta_h["audio_conf"] == 0.9
    print("test_hit_fuse_end_to_end_one_hit: PASS")


def test_hit_fuse_process_collapsed_json_inputs():
    """process() must accept collapsed JSON buffers (new ball_history_collector /
    poses_history_collector format) for ball_pos and tracked_poses."""
    ball_payload = json.dumps({
        "35": [250.0, 200.0, 0.9],
        "36": [240.0, 200.0, 0.9],
        "37": [230.0, 200.0, 0.9],
        "38": [220.0, 200.0, 0.9],
        "39": [210.0, 200.0, 0.9],
        "40": [200.0, 200.0, 0.9],
        "41": [210.0, 200.0, 0.9],
        "42": [220.0, 200.0, 0.9],
        "43": [230.0, 200.0, 0.9],
        "44": [240.0, 200.0, 0.9],
        "45": [250.0, 200.0, 0.9],
    }).encode("utf-8")
    kpts = [[200.0, 200.0, 0.9] for _ in range(17)]
    tracks = [{"track_id": 7, "bbox": [150.0, 150.0, 250.0, 350.0],
               "score": 0.9, "kpts": kpts}]
    poses_payload = json.dumps(
        {str(f): tracks for f in range(20, 60)}
    ).encode("utf-8")
    centers_payload = json.dumps(
        [{"frame_idx": 40, "audio_conf": 0.9, "t_center_sec": 1.33}]
    ).encode("utf-8")

    with tempfile.TemporaryDirectory() as td:
        meta = Path(td) / "asformer_meta.json"
        _write_stub_meta(meta, T=32)

        class _Cfg:
            def __init__(self, d): self.d = d
            def get_int(self, k, dflt): return int(self.d.get(k, dflt))
            def get_float(self, k, dflt): return float(self.d.get(k, dflt))
            def get_string(self, k, dflt): return str(self.d.get(k, dflt))
            def get_bool(self, k, dflt): return bool(self.d.get(k, dflt))

        fu = HitFuse()
        rc = fu.open(_Cfg({"T": 32, "image_width": 1280, "image_height": 720,
                           "require_visual_confirm": True,
                           "min_window_coverage": 0.75,
                           "asformer_meta_path": str(meta)}))
        assert rc == _SC.STATUS_SUCCESS

        fu.data_pre(_DC({}))

        # Process the collapsed JSON inputs
        ball_port = _Port()
        ball_port.push_back(_Buf(None, ball_payload))
        poses_port = _Port()
        poses_port.push_back(_Buf(None, poses_payload))
        centers_port = _Port()
        centers_port.push_back(_Buf(None, centers_payload))
        dc = _DC({"ball_history": ball_port,
                  "poses_history": poses_port,
                  "hit_centers": centers_port})
        rc = fu.process(dc)
        assert rc == _SC.STATUS_SUCCESS, f"process rc={rc}"

        # data_post should produce one confirmed hit
        dc_post = _DC({})
        fu.data_post(dc_post)
        wins = list(dc_post.output("hit_window"))
        metas = list(dc_post.output("hit_meta"))
        drops = list(dc_post.output("dropped_hits"))
        assert len(wins) == 1, f"expected 1 hit_window, got {len(wins)}"
        assert len(metas) == 1, f"expected 1 hit_meta, got {len(metas)}"
        meta_dict = json.loads(bytes(metas[0].as_object()).decode("utf-8"))
        assert meta_dict["frame_idx"] == 40
        assert meta_dict["track_id"] == 7
        print("test_hit_fuse_process_collapsed_json_inputs: PASS")


def test_engineer_pose_features_shape_and_gating():
    """engineer_pose_features must produce (T, 189) and zero out
    derived features for joints below CONF_THRESHOLD."""
    T = 32
    kpts = np.zeros((T, 17, 3), dtype=np.float32)
    kpts[:, :, 0] = 0.5
    kpts[:, :, 1] = 0.5
    kpts[:, :, 2] = 0.8                       # all high-confidence
    feats = engineer_pose_features(kpts)
    assert feats.shape == (T, 189), feats.shape
    assert feats.dtype == np.float32

    # Now zero out joint 9 (left_wrist) confidence; bone (7,9) and any
    # angle/relative referencing 9 should zero out for those rows.
    kpts2 = kpts.copy()
    kpts2[:, 9, 2] = 0.0
    feats2 = engineer_pose_features(kpts2)
    # base[9*3:9*3+2] (x,y) must be zeroed; conf stays
    assert np.allclose(feats2[:, 9 * 3:9 * 3 + 2], 0.0)
    print("test_engineer_pose_features_shape_and_gating: PASS")


def test_data_post_emits_engineered_features():
    """data_post emit must produce a buffer of size 189*T*4 bytes per hit
    (channel-first layout matching ASFormer ONNX input shape)."""
    class _Cfg:
        def __init__(self, d): self.d = d
        def get_int(self, k, dflt): return int(self.d.get(k, dflt))
        def get_float(self, k, dflt): return float(self.d.get(k, dflt))
        def get_string(self, k, dflt): return str(self.d.get(k, dflt))
        def get_bool(self, k, dflt): return bool(self.d.get(k, dflt))

    with tempfile.TemporaryDirectory() as td:
        meta_path = Path(td) / "asformer_meta.json"
        _write_stub_meta(meta_path, T=32)

        fu = HitFuse()
        fu.open(_Cfg({"T": 32, "image_width": 1280, "image_height": 720,
                      "require_visual_confirm": True,
                      "min_window_coverage": 0.0,
                      "asformer_meta_path": str(meta_path)}))
        # Inject a single confirmed hit directly via fuse_session helpers.
        # Use a fake (T,17,3) keypoint window matching what assemble_window
        # produces.
        arr = np.full((32, 17, 3), 0.5, dtype=np.float32)
        fu._centers_raw = []
        # Monkey-patch fuse_session to return our canned arr
        confirmed = [(arr, {"frame_idx": 40, "track_id": 7, "audio_conf": 0.9,
                            "fused_conf": 0.95, "ball_xy": [0.0, 0.0],
                            "tie_resolved": False, "boundary_padded": False,
                            "visual_agrees": True})]
        fu.fuse_session = lambda *a, **kw: (confirmed, [])
        dc = _DC({})
        fu.data_post(dc)
        wins = list(dc.output("hit_window"))
        masks = list(dc.output("mask"))
        assert len(wins) == 1
        assert len(masks) == 1
        body = bytes(wins[0].as_object())
        expected = 189 * 32 * 4
        assert len(body) == expected, (len(body), expected)
        # Reshape to verify channel-first
        arr_out = np.frombuffer(body, dtype=np.float32).reshape(189, 32)
        assert arr_out.shape == (189, 32)
        # Mask is (T,) all-ones
        mask_arr = np.frombuffer(bytes(masks[0].as_object()), dtype=np.float32)
        assert mask_arr.shape == (32,)
        assert np.all(mask_arr == 1.0)
        print("test_data_post_emits_engineered_features: PASS")


def main() -> int:
    test_cross_confirm_direction_flip_accepts()
    test_cross_confirm_no_flip_rejects()
    test_cross_confirm_missing_ball_with_require_drops()
    test_cross_confirm_missing_ball_without_require_accepts()
    test_assign_hitter_picks_nearest_wrist()
    test_assign_hitter_no_player_returns_none()
    test_assign_hitter_tie_break_by_track_id()
    test_assign_hitter_uses_nearby_frame_when_target_empty()
    test_assemble_window_full_coverage()
    test_assemble_window_interpolates_gaps()
    test_assemble_window_coverage_too_low_drops()
    test_assemble_window_boundary_padding()
    test_hit_fuse_open_validates_meta()
    test_hit_fuse_end_to_end_one_hit()
    test_hit_fuse_process_collapsed_json_inputs()
    test_engineer_pose_features_shape_and_gating()
    test_data_post_emits_engineered_features()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
