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

"""Export Ultralytics YOLO-pose to a Core ML .mlpackage.

Output is the raw pre-NMS [1, 5 + 17*3, 8400] = [1, 56, 8400] tensor
(xywh + person_score + 17 keypoints * (x,y,visibility)). The modelbox
yolo_pose_post flowunit consumes this layout directly.

Usage:
    python export_yolo_pose_coreml.py --out path/to/yolov8n-pose.mlpackage
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
        print(f"[pose] {out_path} already exists, skipping.")
        return 0

    try:
        from ultralytics import YOLO
    except ImportError as e:
        print(f"[pose] missing dep: {e}", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model = YOLO("yolov8n-pose.pt")
    exported = model.export(format="coreml", imgsz=args.imgsz, nms=False,
                             half=False)
    if isinstance(exported, (list, tuple)):
        exported = exported[0]
    exported = str(exported)
    if os.path.abspath(exported) != out_path:
        if os.path.exists(out_path):
            shutil.rmtree(out_path) if os.path.isdir(out_path) else os.remove(out_path)
        shutil.copytree(exported, out_path)
    print(f"[pose] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
