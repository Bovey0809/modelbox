#!/usr/bin/env python3
"""Offline smoke test for hit_centers_emitter.

Three helper-function tests cover the algorithm; the fourth drives
process() + data_post() end-to-end via a stub DataContext to ensure the
ModelBox wiring is exercised.
"""

from __future__ import annotations

import json
import sys
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
from hit_centers_emitter import (
    HitCentersEmitter, smooth_majority, intervals_to_centers
)  # noqa: E402


def test_smooth_majority_filters_isolated_spikes():
    bin_seq = np.array([0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0])
    out = smooth_majority(bin_seq, window=5)
    assert out[2] == 0, "isolated 1 at idx 2 should be smoothed out"
    assert out[8] == 1, "consecutive 1s should remain"
    print("test_smooth_majority_filters_isolated_spikes: PASS")


def test_intervals_to_centers_two_clusters():
    """Two clear hit clusters at frames 30 and 90 → two centers."""
    logits = np.zeros((250, 2), dtype=np.float32)
    logits[:, 0] = 1.0  # "no hit" by default
    for idx in range(28, 33):
        logits[idx] = [0.0, 1.0]
    for idx in range(88, 93):
        logits[idx] = [0.0, 1.0]
    centers = intervals_to_centers(
        logits, video_fps=30.0, step_sec=0.02, window_sec=0.5,
        smooth_win=5, max_hit_dur=1.0
    )
    assert len(centers) == 2, f"got {len(centers)} centers: {centers}"
    f0 = centers[0]["frame_idx"]
    f1 = centers[1]["frame_idx"]
    assert 24 <= f0 <= 27, f"f0={f0}"
    # Cluster 2: windows 88-92, step_sec=0.02, window_sec=0.5, fps=30.0
    # t_start=1.76s, t_end=2.34s, t_center=2.05s -> frame=round(61.5)=61
    assert 58 <= f1 <= 64, f"f1={f1}"
    print("test_intervals_to_centers_two_clusters: PASS")


def test_fps_swap_shifts_frame_idx():
    logits = np.zeros((250, 2), dtype=np.float32)
    logits[:, 0] = 1.0
    for idx in range(48, 53):
        logits[idx] = [0.0, 1.0]
    c25 = intervals_to_centers(logits, video_fps=25.0, step_sec=0.02,
                               window_sec=0.5,
                               smooth_win=5, max_hit_dur=1.0)
    c30 = intervals_to_centers(logits, video_fps=30.0, step_sec=0.02,
                               window_sec=0.5,
                               smooth_win=5, max_hit_dur=1.0)
    assert len(c25) == 1 and len(c30) == 1
    assert c30[0]["frame_idx"] > c25[0]["frame_idx"], \
        f"{c25[0]} vs {c30[0]}"
    print("test_fps_swap_shifts_frame_idx: PASS")


def test_process_then_data_post_emits_one_buffer():
    """Drive process() + data_post(): feed N per-window logit buffers and
    confirm one collapsed buffer is emitted at session end with the
    expected JSON schema."""
    # 250 windows; one hit cluster.
    logits = np.zeros((250, 2), dtype=np.float32)
    logits[:, 0] = 1.0
    for idx in range(48, 53):
        logits[idx] = [0.0, 1.0]

    fu = HitCentersEmitter()
    class _Cfg:
        def get_int(self, k, d): return d
        def get_float(self, k, d): return d
        def get_string(self, k, d): return d
        def get_bool(self, k, d): return d
    fu.open(_Cfg())

    fu.data_pre(_DC({}))  # explicit session start
    in_port = _Port()
    # Feed each window as a separate buffer to mirror the real engine.
    for w in logits:
        in_port.push_back(_Buf(None, w.astype(np.float32).tobytes()))
    dc_proc = _DC({"logits": in_port})
    rc = fu.process(dc_proc)
    assert rc == _SC.STATUS_SUCCESS

    # data_post emits the collapsed buffer.
    dc_post = _DC({})
    fu.data_post(dc_post)
    out = list(dc_post.output("hit_centers"))
    assert len(out) == 1, f"expected 1 collapsed buffer, got {len(out)}"
    centers = json.loads(bytes(out[0].as_object()).decode("utf-8"))
    assert len(centers) == 1
    assert "frame_idx" in centers[0] and "audio_conf" in centers[0] \
        and "t_center_sec" in centers[0]
    print("test_process_then_data_post_emits_one_buffer: PASS")


def test_data_pre_resets_accumulator():
    """Two consecutive sessions must NOT carry over logits."""
    fu = HitCentersEmitter()
    class _Cfg:
        def get_int(self, k, d): return d
        def get_float(self, k, d): return d
        def get_string(self, k, d): return d
        def get_bool(self, k, d): return d
    fu.open(_Cfg())

    # Session 1: push a single hit-positive logit row, but never call
    # data_post — pretend the session ends abruptly.
    fu.data_pre(_DC({}))
    p1 = _Port()
    p1.push_back(_Buf(None, np.array([0.0, 1.0], dtype=np.float32).tobytes()))
    fu.process(_DC({"logits": p1}))
    assert len(fu._logits) == 1

    # Session 2 should reset the accumulator.
    fu.data_pre(_DC({}))
    assert len(fu._logits) == 0, \
        f"data_pre should clear accumulator, got {len(fu._logits)} entries"
    print("test_data_pre_resets_accumulator: PASS")


def test_long_run_dropped_by_max_hit_dur():
    """Run longer than max_hit_dur=1.0s (51 windows × 0.02s = 1.02s of
    run-start span) is dropped; a 49-window run is kept."""
    logits = np.zeros((200, 2), dtype=np.float32)
    logits[:, 0] = 1.0

    # 52-window run (indices 20..71): run_start=20, run_end=71,
    # span = (71-20)*0.02 = 1.02s > max_hit_dur=1.0 → should be dropped.
    for idx in range(20, 72):
        logits[idx] = [0.0, 1.0]
    centers = intervals_to_centers(
        logits, video_fps=30.0, step_sec=0.02, window_sec=0.5,
        smooth_win=5, max_hit_dur=1.0
    )
    assert len(centers) == 0, f"52-window run should be dropped, got {centers}"

    # Reset and try 49 windows: indices 20..68 inclusive → 49 windows, span = 48*0.02 = 0.96s.
    logits[:] = 0.0
    logits[:, 0] = 1.0
    for idx in range(20, 69):
        logits[idx] = [0.0, 1.0]
    centers = intervals_to_centers(
        logits, video_fps=30.0, step_sec=0.02, window_sec=0.5,
        smooth_win=5, max_hit_dur=1.0
    )
    assert len(centers) == 1, f"49-window run should be kept, got {centers}"
    print("test_long_run_dropped_by_max_hit_dur: PASS")


def main() -> int:
    test_smooth_majority_filters_isolated_spikes()
    test_intervals_to_centers_two_clusters()
    test_fps_swap_shifts_frame_idx()
    test_process_then_data_post_emits_one_buffer()
    test_data_pre_resets_accumulator()
    test_long_run_dropped_by_max_hit_dur()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
