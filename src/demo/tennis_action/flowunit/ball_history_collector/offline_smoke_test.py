#!/usr/bin/env python3
"""Offline smoke test for ball_history_collector. Stubs _flowunit so the
Python class can be exercised without ModelBox linkage."""

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
    def get(self, k):
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
        self._inputs = inputs
        self._outputs: dict = {}
    def input(self, name):
        return self._inputs[name]
    def output(self, name):
        if name not in self._outputs:
            self._outputs[name] = _Port()
        return self._outputs[name]

class _SC:
    STATUS_SUCCESS = 0
    STATUS_FAULT = 1

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
from ball_history_collector import BallHistoryCollector  # noqa: E402


def _make_ball_buf(cx: float, cy: float, peak: float, fi: int) -> _Buf:
    arr = np.array([cx, cy, peak, float(fi)], dtype=np.float32)
    return _Buf(None, arr.tobytes())


def test_basic_accumulation():
    """Three ball_pos buffers → one collapsed JSON with three entries."""
    fu = BallHistoryCollector()
    fu.open(object())  # config unused

    # data_pre resets state
    dc_pre = _DC({})
    fu.data_pre(dc_pre)

    # Simulate three process() calls (one buffer each)
    for fi, (cx, cy, peak) in enumerate([(100.0, 200.0, 0.8),
                                          (110.0, 205.0, 0.9),
                                          (-1.0, -1.0, 0.1)]):
        in_port = _Port()
        in_port.push_back(_make_ball_buf(cx, cy, peak, fi))
        dc = _DC({"ball_pos": in_port})
        rc = fu.process(dc)
        assert rc == _SC.STATUS_SUCCESS, f"process rc={rc}"

    # data_post should emit exactly one buffer
    dc_post = _DC({})
    fu.data_post(dc_post)
    out_items = list(dc_post.output("ball_history"))
    assert len(out_items) == 1, f"expected 1 output buffer, got {len(out_items)}"

    payload = json.loads(bytes(out_items[0].as_object()).decode("utf-8"))
    assert len(payload) == 3, f"expected 3 keys, got {len(payload)}"
    assert "0" in payload and "1" in payload and "2" in payload
    assert abs(payload["0"][0] - 100.0) < 0.01
    assert abs(payload["1"][2] - 0.9) < 0.01
    assert payload["2"][0] == -1.0
    print("test_basic_accumulation: PASS")


def test_empty_session():
    """No process() calls → data_post emits a single empty-dict buffer."""
    fu = BallHistoryCollector()
    fu.open(object())
    dc_pre = _DC({})
    fu.data_pre(dc_pre)

    dc_post = _DC({})
    fu.data_post(dc_post)
    out_items = list(dc_post.output("ball_history"))
    assert len(out_items) == 1
    payload = json.loads(bytes(out_items[0].as_object()).decode("utf-8"))
    assert payload == {}, f"expected empty dict, got {payload}"
    print("test_empty_session: PASS")


def test_data_pre_resets_between_sessions():
    """data_pre must clear state so a second session starts fresh."""
    fu = BallHistoryCollector()
    fu.open(object())

    # Session 1
    fu.data_pre(_DC({}))
    in_port = _Port()
    in_port.push_back(_make_ball_buf(50.0, 60.0, 0.7, 5))
    fu.process(_DC({"ball_pos": in_port}))

    # Session 2 starts with data_pre
    fu.data_pre(_DC({}))
    dc_post = _DC({})
    fu.data_post(dc_post)
    payload = json.loads(bytes(list(dc_post.output("ball_history"))[0].as_object()).decode("utf-8"))
    assert payload == {}, f"expected empty after reset, got {payload}"
    print("test_data_pre_resets_between_sessions: PASS")


def test_multiple_buffers_per_process_call():
    """process() handles a port with multiple buffers in one call."""
    fu = BallHistoryCollector()
    fu.open(object())
    fu.data_pre(_DC({}))

    in_port = _Port()
    for fi in range(5):
        in_port.push_back(_make_ball_buf(float(fi * 10), float(fi * 5), 0.8, fi))
    rc = fu.process(_DC({"ball_pos": in_port}))
    assert rc == _SC.STATUS_SUCCESS

    dc_post = _DC({})
    fu.data_post(dc_post)
    payload = json.loads(bytes(list(dc_post.output("ball_history"))[0].as_object()).decode("utf-8"))
    assert len(payload) == 5
    assert all(str(i) in payload for i in range(5))
    print("test_multiple_buffers_per_process_call: PASS")


def main() -> int:
    test_basic_accumulation()
    test_empty_session()
    test_data_pre_resets_between_sessions()
    test_multiple_buffers_per_process_call()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
