#!/usr/bin/env python3
"""Smoke test for the sutrack demo graph.

Usage:
    python test_sutrack.py [VIDEO_PATH] --bbox X,Y,W,H

POSTs a JSON request, prints the response, returns non-zero on error.
The server must be running first, e.g.

    modelbox-tool flow run -name SUTrack \
      -path /usr/local/share/modelbox/demo/sutrack/graph/sutrack.toml
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("video", nargs="?", default="")
    p.add_argument("--bbox", required=True, help="Init bbox 'x,y,w,h' (pixels)")
    p.add_argument("--endpoint", default="http://127.0.0.1:7781/v1/sutrack")
    p.add_argument("--text", default="")
    p.add_argument("--output-video", default="/tmp/sutrack_overlay.mp4")
    p.add_argument("--output-json", default="/tmp/sutrack_bboxes.json")
    p.add_argument("--bbox-normalized", action="store_true")
    args = p.parse_args()

    init_bbox = [float(v) for v in args.bbox.split(",")]
    if len(init_bbox) != 4:
        print("bbox must be x,y,w,h", file=sys.stderr)
        return 1

    body = json.dumps({
        "video_path": args.video,
        "init_bbox": init_bbox,
        "text": args.text or None,
        "output_video": args.output_video,
        "output_json": args.output_json,
        "bbox_normalized": args.bbox_normalized,
    }).encode("utf-8")
    req = urllib.request.Request(args.endpoint, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as resp:
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
