#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke test for the hit_sound demo graph.

Usage:
    python test_hit_sound.py [VIDEO_PATH] [--endpoint URL]

Posts a JSON request to the hit_sound HTTP server, prints the response, and
returns non-zero on error. The server must be running, e.g.

    modelbox-tool flow run \
        -name HitSound \
        -path /usr/local/share/modelbox/demo/hit_sound/graph/hit_sound.toml
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

DEFAULT_VIDEO = "/Users/houbowei/Desktop/tennis/2026-01-04-action-06.mp4"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("video", nargs="?", default=DEFAULT_VIDEO)
    p.add_argument("--endpoint", default="http://127.0.0.1:7771/v1/hit_sound")
    p.add_argument("--mode", default="hit", choices=["hit", "rally", "both"])
    p.add_argument("--overlay", default="/tmp/hit_sound_overlay.mp4")
    p.add_argument("--intervals", default="/tmp/hit_sound_intervals.csv")
    args = p.parse_args()

    body = json.dumps({
        "video_path": args.video,
        "interval_mode": args.mode,
        "intervals_csv": args.intervals,
        "overlay_mp4": args.overlay,
    }).encode("utf-8")
    req = urllib.request.Request(args.endpoint, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        text = resp.read().decode("utf-8")
    try:
        result = json.loads(text)
    except Exception:
        print(text)
        return 1
    print(json.dumps(result, indent=2))
    return 0 if "error" not in result else 2


if __name__ == "__main__":
    sys.exit(main())
