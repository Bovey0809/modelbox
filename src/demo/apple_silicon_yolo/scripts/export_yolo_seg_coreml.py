#!/usr/bin/env python3
#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
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

"""Export Ultralytics YOLO-seg to a Core ML .mlpackage.

Two outputs (matching ONNX/OpenVINO seg layout):
  - [1, 4+nc+nm, 8400] detection tensor (xywh + class scores + mask coefs)
  - [1, 32, 160, 160]  prototype masks
The modelbox yolo_seg_post flowunit takes both via in_feat / in_proto ports.

Usage:
    python export_yolo_seg_coreml.py --out path/to/yolov8n-seg.mlpackage
"""

import argparse
import os
import shutil
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    out_path = os.path.abspath(args.out)
    if os.path.isdir(out_path) and os.path.exists(
            os.path.join(out_path, "Manifest.json")):
        print(f"[seg] {out_path} already exists, skipping.")
        return 0

    try:
        from ultralytics import YOLO
    except ImportError as e:
        print(f"[seg] missing dep: {e}", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model = YOLO("yolov8n-seg.pt")
    exported = model.export(format="coreml", imgsz=args.imgsz, nms=False,
                             half=False)
    if isinstance(exported, (list, tuple)):
        exported = exported[0]
    exported = str(exported)
    if os.path.abspath(exported) != out_path:
        if os.path.exists(out_path):
            shutil.rmtree(out_path) if os.path.isdir(out_path) else os.remove(out_path)
        shutil.copytree(exported, out_path)
    print(f"[seg] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
