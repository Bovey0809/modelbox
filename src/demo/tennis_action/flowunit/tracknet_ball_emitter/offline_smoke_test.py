#!/usr/bin/env python3
"""Offline smoke test for tracknet_ball_emitter. Stubs the _flowunit module
so we can import and exercise the Python class without ModelBox linkage."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

_FU = types.ModuleType("_flowunit")
_FU.Buffer = lambda dev, b: ("BUFFER", b)  # opaque
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

# Upgraded stub: real-enough Buffer + DataContext for process() drive-through.
class _Buf:
    def __init__(self, dev, payload):
        self._payload = payload
        self._meta: dict = {}
    def as_object(self):
        return self._payload
    def set(self, k, v):
        self._meta[k] = v
    def meta(self, k):
        return self._meta.get(k)
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
        self._inputs = inputs  # {name: _Port-with-prepopulated-buffers}
        self._outputs: dict = {}
    def input(self, name):
        return self._inputs[name]
    def output(self, name):
        if name not in self._outputs:
            self._outputs[name] = _Port()
        return self._outputs[name]

sys.path.insert(0, str(Path(__file__).parent))
from tracknet_ball_emitter import TracknetBallEmitter, extract_centroid  # noqa: E402


def test_centroid_above_threshold():
    """A single hot pixel in channel 1 → (cx, cy) reported in net coords."""
    heat = np.zeros((3, 288, 512), dtype=np.float32)
    heat[1, 100, 200] = 1.0  # peak at (y=100, x=200) on channel 1
    cx, cy, peak = extract_centroid(heat, score_thr=0.3, mask_ratio=0.5,
                                    net_h=288, net_w=512)
    assert peak >= 0.99, f"peak={peak}"
    assert abs(cx - 200) < 1.0, f"cx={cx}"
    assert abs(cy - 100) < 1.0, f"cy={cy}"
    print("test_centroid_above_threshold: PASS")


def test_below_threshold_returns_sentinel():
    """All-zero heatmap → cx=cy=-1."""
    heat = np.zeros((3, 288, 512), dtype=np.float32)
    cx, cy, peak = extract_centroid(heat, score_thr=0.3, mask_ratio=0.5,
                                    net_h=288, net_w=512)
    assert peak < 0.3
    assert cx == -1.0 and cy == -1.0, f"({cx},{cy}) instead of (-1,-1)"
    print("test_below_threshold_returns_sentinel: PASS")


def test_extract_centroid_returns_net_coords():
    """Peak at net center → cx, cy are at net center (no scaling here)."""
    heat = np.zeros((3, 288, 512), dtype=np.float32)
    heat[1, 144, 256] = 0.9  # net-resolution center
    cx, cy, peak = extract_centroid(heat, score_thr=0.3, mask_ratio=0.5,
                                    net_h=288, net_w=512)
    assert peak >= 0.89
    assert abs(cx - 256) < 1.0 and abs(cy - 144) < 1.0, f"({cx},{cy})"
    print("test_extract_centroid_returns_net_coords: PASS")


def test_flowunit_open_reads_config():
    """The flowunit picks up config defaults correctly."""
    class _Cfg:
        def __init__(self, d): self.d = d
        def get_int(self, k, default): return int(self.d.get(k, default))
        def get_float(self, k, default): return float(self.d.get(k, default))
        def get_string(self, k, default): return str(self.d.get(k, default))
    fu = TracknetBallEmitter()
    rc = fu.open(_Cfg({"source_width": 1280, "source_height": 720,
                       "score_thr": 0.3, "net_h": 288, "net_w": 512}))
    assert rc == _SC.STATUS_SUCCESS
    assert fu.source_width == 1280 and fu.source_height == 720
    print("test_flowunit_open_reads_config: PASS")


def test_process_scales_to_source_coords():
    """Drive TracknetBallEmitter.process(): net center peak -> scaled source-frame coords."""
    heat = np.zeros((3, 288, 512), dtype=np.float32)
    heat[1, 144, 256] = 0.9

    fu = TracknetBallEmitter()
    class _Cfg:
        def __init__(self, d): self.d = d
        def get_int(self, k, default): return int(self.d.get(k, default))
        def get_float(self, k, default): return float(self.d.get(k, default))
        def get_string(self, k, default): return str(self.d.get(k, default))
    fu.open(_Cfg({"source_width": 1280, "source_height": 720,
                  "score_thr": 0.3, "mask_ratio": 0.5,
                  "net_h": 288, "net_w": 512}))

    in_port = _Port()
    in_port.push_back(_Buf(None, heat.astype(np.float32).tobytes()))
    dc = _DC({"heatmaps": in_port})
    rc = fu.process(dc)
    assert rc == _SC.STATUS_SUCCESS, f"rc={rc}"

    out_port = dc.output("ball_pos")
    bufs = list(out_port)
    assert len(bufs) == 1, f"expected 1 output buffer, got {len(bufs)}"
    arr = np.frombuffer(bufs[0].as_object(), dtype=np.float32)
    cx, cy, peak, fi = float(arr[0]), float(arr[1]), float(arr[2]), int(arr[3])
    # Net (256, 144) scaled by (1280/512, 720/288) -> (640, 360).
    assert abs(cx - 640) < 2.0, f"cx={cx} (expected ~640)"
    assert abs(cy - 360) < 2.0, f"cy={cy} (expected ~360)"
    assert peak >= 0.89
    assert fi == 0
    print("test_process_scales_to_source_coords: PASS")


def main() -> int:
    test_centroid_above_threshold()
    test_below_threshold_returns_sentinel()
    test_extract_centroid_returns_net_coords()     # renamed
    test_flowunit_open_reads_config()
    test_process_scales_to_source_coords()         # new
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
