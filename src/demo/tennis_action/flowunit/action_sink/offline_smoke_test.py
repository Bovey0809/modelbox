#!/usr/bin/env python3
"""Offline smoke test for action_sink. Verifies JSON schema + overlay write."""

from __future__ import annotations

import json
import subprocess
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
        return self._inputs.get(name, _Port())
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
from action_sink import ActionSink, write_results, render_overlay  # noqa: E402

# Path to the committed test video.
# parents[0]=action_sink, [1]=flowunit, [2]=tennis_action, [3]=demo
TEST_VIDEO = Path(__file__).resolve().parents[2] / "test" / "tennis_test.mp4"
CLASSES = ["forehand", "backhand", "serve", "other"]


def test_write_results_json_schema():
    confirmed = [{
        "hit_id": 0, "frame_idx": 40, "track_id": 7,
        "audio_conf": 0.9, "fused_conf": 0.95,
        "ball_xy": [200.0, 200.0],
        "tie_resolved": False, "boundary_padded": False,
        "visual_agrees": True,
        "action": "forehand", "action_conf": 0.7,
    }]
    dropped = [{"audio_hit_frame_idx": 10, "audio_conf": 0.6,
                "reason": "no_ball_near_hit"}]
    with tempfile.TemporaryDirectory() as td:
        j = Path(td) / "out.json"
        d = Path(td) / "dropped.json"
        write_results(j, d, confirmed, dropped, fps=30.0)
        body = json.loads(j.read_text())
        assert "summary" in body and "hits" in body
        assert body["summary"]["confirmed"] == 1
        assert body["summary"]["dropped"]["no_ball_near_hit"] == 1
        assert abs(body["hits"][0]["t_sec"] - (40 / 30.0)) < 1e-3
        d_body = json.loads(d.read_text())
        assert d_body[0]["reason"] == "no_ball_near_hit"
        print("test_write_results_json_schema: PASS")


def test_write_results_empty_inputs():
    """No hits and no drops -> both files written with empty arrays."""
    with tempfile.TemporaryDirectory() as td:
        j = Path(td) / "out.json"
        d = Path(td) / "dropped.json"
        write_results(j, d, [], [], fps=30.0)
        body = json.loads(j.read_text())
        assert body["summary"]["audio_hits"] == 0
        assert body["summary"]["confirmed"] == 0
        assert body["hits"] == []
        assert json.loads(d.read_text()) == []
        print("test_write_results_empty_inputs: PASS")


def test_render_overlay_produces_decodable_mp4():
    if not TEST_VIDEO.exists():
        print(f"test_render_overlay_produces_decodable_mp4: SKIP "
              f"(test video missing at {TEST_VIDEO})")
        return
    try:
        import cv2  # noqa: F401
    except ImportError:
        print("test_render_overlay_produces_decodable_mp4: SKIP (cv2 missing)")
        return
    confirmed = [{
        "hit_id": 0, "frame_idx": 40, "track_id": 7,
        "audio_conf": 0.9, "fused_conf": 0.95,
        "ball_xy": [640.0, 360.0],
        "tie_resolved": False, "boundary_padded": False,
        "visual_agrees": True,
        "action": "forehand", "action_conf": 0.7,
    }]
    poses_per_frame = {f: [{"track_id": 7, "bbox": [600, 300, 700, 500],
                            "kpts": [[640, 380, 0.9]] * 17}]
                       for f in range(0, 141)}
    ball_per_frame = {f: (640.0, 360.0, 0.9) for f in range(0, 141)}
    with tempfile.TemporaryDirectory() as td:
        out_mp4 = Path(td) / "overlay.mp4"
        ok = render_overlay(
            source_video=str(TEST_VIDEO),
            output_path=str(out_mp4),
            confirmed=confirmed,
            poses_per_frame=poses_per_frame,
            ball_per_frame=ball_per_frame,
            hit_label_persist_frames=30,
        )
        assert ok and out_mp4.exists() and out_mp4.stat().st_size > 5000, \
            f"ok={ok}, exists={out_mp4.exists()}, size={out_mp4.stat().st_size if out_mp4.exists() else 'n/a'}"
        # ffprobe sanity.
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=codec_type,duration", "-of", "default=noprint_wrappers=1",
             str(out_mp4)],
            capture_output=True, text=True, check=True,
        )
        assert "codec_type=video" in result.stdout
        print("test_render_overlay_produces_decodable_mp4: PASS")


def test_process_then_data_post_writes_files():
    """Drive process() + data_post() end-to-end with synthetic inputs.
    Verifies that ports are drained and JSON files are written."""
    fu = ActionSink()

    class _Cfg:
        def __init__(self, d): self.d = d
        def get_int(self, k, dflt): return int(self.d.get(k, dflt))
        def get_float(self, k, dflt): return float(self.d.get(k, dflt))
        def get_string(self, k, dflt): return str(self.d.get(k, dflt))
        def get_bool(self, k, dflt): return bool(self.d.get(k, dflt))

    with tempfile.TemporaryDirectory() as td:
        meta_path = Path(td) / "asformer_meta.json"
        meta_path.write_text(json.dumps({
            "classes": CLASSES, "num_kpts": 17, "T": 32, "input_layout": "BTC"
        }))
        cfg = _Cfg({
            "output_json": str(Path(td) / "out.json"),
            "output_overlay": str(Path(td) / "overlay.mp4"),
            "output_dropped": str(Path(td) / "dropped.json"),
            "class_names_path": str(meta_path),
            "overlay_encoder": "libx264",
            "overlay_fallback_encoder": "libx264",
            "hit_label_persist_frames": 30,
            "source_video": "",  # disable overlay rendering for this test
        })
        fu.open(cfg)
        fu.data_pre(_DC({}))

        # Feed: 1 hit logits buffer (4 classes, argmax->1=backhand),
        # 1 hit_meta buffer, 1 dropped_hits collapsed buffer (empty),
        # 1 ball_pos buffer, 1 tracked_poses buffer.
        logits_port = _Port()
        logits_port.push_back(_Buf(None,
            np.array([0.1, 0.9, 0.05, 0.05], dtype=np.float32).tobytes()))
        meta_port = _Port()
        meta_port.push_back(_Buf(None, json.dumps({
            "hit_id": 0, "frame_idx": 40, "track_id": 7,
            "audio_conf": 0.9, "fused_conf": 0.95, "ball_xy": [640, 360],
            "tie_resolved": False, "boundary_padded": False,
            "visual_agrees": True,
        }).encode("utf-8")))
        dropped_port = _Port()
        dropped_port.push_back(_Buf(None, json.dumps([]).encode("utf-8")))
        ball_port = _Port()
        ball_port.push_back(_Buf(None,
            np.array([640.0, 360.0, 0.9, 40.0], dtype=np.float32).tobytes()))
        poses_port = _Port()
        poses_port.push_back(_Buf(None, json.dumps({
            "frame_idx": 40, "tracks": []
        }).encode("utf-8")))

        dc_proc = _DC({"logits": logits_port, "hit_meta": meta_port,
                       "dropped_hits": dropped_port, "ball_pos": ball_port,
                       "tracked_poses": poses_port})
        rc = fu.process(dc_proc)
        assert rc == _SC.STATUS_SUCCESS

        fu.data_post(_DC({}))

        # Verify JSON written and parses.
        body = json.loads(Path(cfg.get_string("output_json", "")).read_text())
        assert body["summary"]["confirmed"] == 1
        assert body["hits"][0]["action"] == "backhand", body["hits"][0]
        assert body["hits"][0]["track_id"] == 7

        dropped = json.loads(
            Path(cfg.get_string("output_dropped", "")).read_text())
        assert dropped == []

        # Overlay was skipped (source_video="") so the mp4 should NOT exist.
        assert not Path(cfg.get_string("output_overlay", "")).exists()
        print("test_process_then_data_post_writes_files: PASS")


def main() -> int:
    test_write_results_json_schema()
    test_write_results_empty_inputs()
    test_render_overlay_produces_decodable_mp4()
    test_process_then_data_post_writes_files()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
