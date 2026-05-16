#!/usr/bin/env python3
"""Offline smoke test for poses_history_collector. Stubs _flowunit so the
Python class can be exercised without ModelBox linkage."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

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
from poses_history_collector import PosesHistoryCollector  # noqa: E402


def _make_pose_buf(frame_idx: int, tracks: list) -> _Buf:
    payload = json.dumps({"frame_idx": frame_idx, "tracks": tracks})
    return _Buf(None, payload.encode("utf-8"))


def _make_track(tid: int, x: float = 100.0, y: float = 200.0) -> dict:
    kpts = [[x + i, y + i, 0.9] for i in range(17)]
    return {"track_id": tid, "bbox": [x, y, x + 100, y + 200],
            "score": 0.85, "kpts": kpts}


def test_basic_accumulation():
    """Three pose frames → one collapsed JSON with three string-keyed entries."""
    fu = PosesHistoryCollector()
    fu.open(object())
    fu.data_pre(_DC({}))

    for fi in range(3):
        in_port = _Port()
        in_port.push_back(_make_pose_buf(fi, [_make_track(7), _make_track(8)]))
        rc = fu.process(_DC({"tracked_poses": in_port}))
        assert rc == _SC.STATUS_SUCCESS, f"process rc={rc}"

    dc_post = _DC({})
    fu.data_post(dc_post)
    out_items = list(dc_post.output("poses_history"))
    assert len(out_items) == 1, f"expected 1 output, got {len(out_items)}"

    payload = json.loads(bytes(out_items[0].as_object()).decode("utf-8"))
    assert len(payload) == 3, f"expected 3 keys, got {len(payload)}"
    assert all(str(i) in payload for i in range(3))
    tracks_f0 = payload["0"]
    assert len(tracks_f0) == 2
    assert tracks_f0[0]["track_id"] == 7
    assert len(tracks_f0[0]["kpts"]) == 17
    print("test_basic_accumulation: PASS")


def test_empty_session():
    """No process() calls → data_post emits a single empty-dict buffer."""
    fu = PosesHistoryCollector()
    fu.open(object())
    fu.data_pre(_DC({}))

    dc_post = _DC({})
    fu.data_post(dc_post)
    out_items = list(dc_post.output("poses_history"))
    assert len(out_items) == 1
    payload = json.loads(bytes(out_items[0].as_object()).decode("utf-8"))
    assert payload == {}, f"expected empty dict, got {payload}"
    print("test_empty_session: PASS")


def test_data_pre_resets_between_sessions():
    """data_pre clears state so a second session starts fresh."""
    fu = PosesHistoryCollector()
    fu.open(object())

    # Session 1
    fu.data_pre(_DC({}))
    in_port = _Port()
    in_port.push_back(_make_pose_buf(10, [_make_track(3)]))
    fu.process(_DC({"tracked_poses": in_port}))

    # Session 2 — data_pre resets
    fu.data_pre(_DC({}))
    dc_post = _DC({})
    fu.data_post(dc_post)
    payload = json.loads(bytes(list(dc_post.output("poses_history"))[0].as_object()).decode("utf-8"))
    assert payload == {}, f"expected empty after reset, got {payload}"
    print("test_data_pre_resets_between_sessions: PASS")


def test_multiple_buffers_per_process_call():
    """process() handles a port with multiple buffers in one call."""
    fu = PosesHistoryCollector()
    fu.open(object())
    fu.data_pre(_DC({}))

    in_port = _Port()
    for fi in range(5):
        in_port.push_back(_make_pose_buf(fi, [_make_track(fi + 1)]))
    rc = fu.process(_DC({"tracked_poses": in_port}))
    assert rc == _SC.STATUS_SUCCESS

    dc_post = _DC({})
    fu.data_post(dc_post)
    payload = json.loads(bytes(list(dc_post.output("poses_history"))[0].as_object()).decode("utf-8"))
    assert len(payload) == 5
    assert all(str(i) in payload for i in range(5))
    print("test_multiple_buffers_per_process_call: PASS")


def test_collapsed_json_shape():
    """Verify the output payload structure matches what hit_fuse and action_sink expect."""
    fu = PosesHistoryCollector()
    fu.open(object())
    fu.data_pre(_DC({}))

    in_port = _Port()
    in_port.push_back(_make_pose_buf(42, [_make_track(9, x=300.0, y=150.0)]))
    fu.process(_DC({"tracked_poses": in_port}))

    dc_post = _DC({})
    fu.data_post(dc_post)
    payload = json.loads(bytes(list(dc_post.output("poses_history"))[0].as_object()).decode("utf-8"))

    assert "42" in payload
    track = payload["42"][0]
    assert track["track_id"] == 9
    assert len(track["kpts"]) == 17
    assert track["score"] == 0.85
    assert len(track["bbox"]) == 4
    print("test_collapsed_json_shape: PASS")


def main() -> int:
    test_basic_accumulation()
    test_empty_session()
    test_data_pre_resets_between_sessions()
    test_multiple_buffers_per_process_call()
    test_collapsed_json_shape()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
