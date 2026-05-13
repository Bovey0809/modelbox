#!/usr/bin/env python3
"""Offline smoke test for HitSoundDetect — runs the flowunit's _run() logic
without ModelBox by stubbing _flowunit, so we can verify the algorithm
produces the same intervals as the standalone deploy/inference_onnx.py.

Usage:
    python offline_smoke_test.py --video VIDEO --onnx ONNX --intervals_csv OUT
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

# --- Stub the _flowunit module so we can import hit_sound_detect.py without
# linking against ModelBox. ---
_fu = types.ModuleType("_flowunit")
_fu.Buffer = object  # never instantiated in this offline path

class _Status:
    class StatusCode:
        STATUS_SUCCESS = 0
        STATUS_FAULT = 1

_fu.Status = _Status
_fu.FlowUnit = type("FlowUnit", (), {"__init__": lambda self: None})
_fu.info = lambda msg: print(f"[info] {msg}", file=sys.stderr)
_fu.error = lambda msg: print(f"[error] {msg}", file=sys.stderr)
_fu.debug = lambda msg: None
sys.modules["_flowunit"] = _fu

import hit_sound_detect as hsd  # noqa: E402


class _CfgStub:
    """Quacks like modelbox.Configuration with the typed getters we use."""

    def __init__(self, mapping):
        self.m = mapping

    def get_string(self, k, d=""):
        return str(self.m.get(k, d))

    def get_int(self, k, d=0):
        return int(self.m.get(k, d))

    def get_float(self, k, d=0.0):
        return float(self.m.get(k, d))

    def get_bool(self, k, d=False):
        v = self.m.get(k, d)
        if isinstance(v, str):
            return v.lower() in ("1", "true", "yes", "on")
        return bool(v)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--onnx", required=True)
    p.add_argument("--intervals_csv", default="/tmp/hit_sound_intervals_offline.csv")
    p.add_argument("--overlay_mp4", default="")
    p.add_argument("--mode", default="hit", choices=["hit", "rally", "both"])
    args = p.parse_args()

    unit = hsd.HitSoundDetect()
    cfg = _CfgStub({
        "model_path": args.onnx,
        "target_sr": 16000,
        "window_sec": 0.5,
        "step_sec": 0.02,
        "n_fft": 512,
        "hop_length": 170,
        "n_mels": 128,
        "threshold": 0.5,
        "smooth_win": 5,
        "max_hit_dur": 1.0,
        "max_gap": 3.0,
        "buffer_time": 4.0,
        "default_intervals_csv": args.intervals_csv,
        "default_overlay_mp4": args.overlay_mp4,
        "write_overlay": bool(args.overlay_mp4),
    })
    rc = unit.open(cfg)
    if rc != _Status.StatusCode.STATUS_SUCCESS:
        print("open failed", file=sys.stderr)
        return 1
    result = unit._run({
        "video_path": args.video,
        "interval_mode": args.mode,
        "intervals_csv": args.intervals_csv,
        "overlay_mp4": args.overlay_mp4,
    })
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
