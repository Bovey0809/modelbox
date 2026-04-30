#!/usr/bin/env python3
#
# Copyright 2021 The Modelbox Project Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export an Ultralytics YOLO model to ONNX for OpenVINO inference.

Tries `yolo26n.pt` first; falls back to `yolov8n.pt` if v26 weights are
not yet published. Output tensor format is identical for v8 / v10 / v11
/ v26 (anchor-free, [B, 4 + num_classes, N]), so the post-processor
flowunit handles either drop-in.

Usage:
    python export_yolo26n.py --out path/to/yolo26n.onnx --imgsz 640
"""

import argparse
import os
import shutil
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        required=True,
        help="Destination .onnx path (parent dir must exist).",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset (13+ keeps OpenVINO 2024 happy).",
    )
    args = parser.parse_args()

    if os.path.exists(args.out):
        print(f"[export_yolo26n] {args.out} already exists, skipping export.")
        return 0

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "[export_yolo26n] ultralytics is not installed. "
            "Run `pip install ultralytics`.",
            file=sys.stderr,
        )
        return 1

    candidates = ["yolo26n.pt", "yolov8n.pt"]
    last_err = None
    for weights in candidates:
        try:
            print(f"[export_yolo26n] trying {weights}...")
            model = YOLO(weights)
            exported = model.export(
                format="onnx",
                imgsz=args.imgsz,
                opset=args.opset,
                dynamic=False,
                simplify=True,
            )
            # Ultralytics returns a path (or list); normalize:
            if isinstance(exported, (list, tuple)):
                exported = exported[0]
            exported = str(exported)
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            if os.path.abspath(exported) != os.path.abspath(args.out):
                shutil.copyfile(exported, args.out)
            print(f"[export_yolo26n] wrote {args.out}")
            return 0
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[export_yolo26n] {weights} failed: {e}", file=sys.stderr)
            continue

    print(
        f"[export_yolo26n] all candidates failed; last error: {last_err}",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
