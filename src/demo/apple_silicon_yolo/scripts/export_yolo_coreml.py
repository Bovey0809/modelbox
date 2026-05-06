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

"""Export an Ultralytics YOLO model to a Core ML .mlpackage for inference on
Apple Silicon (CPU + GPU + ANE via Core ML).

Tries `yolo26n.pt` first, falls back to `yolov8n.pt`. Output tensor format is
the same anchor-free `[B, 4 + num_classes, N]` layout for v8/v10/v11/v26, so
the existing yolo26_post flowunit is the drop-in post-processor.

Usage:
    python export_yolo_coreml.py --out path/to/yolov8n.mlpackage --imgsz 640
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
        help="Destination .mlpackage directory path.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    out_path = os.path.abspath(args.out)

    # .mlpackage is a directory; existence check matches the Arc770 export
    # script's idempotent behavior.
    if os.path.exists(out_path):
        print(
            f"[export_yolo_coreml] {out_path} already exists, skipping export."
        )
        return 0

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "[export_yolo_coreml] ultralytics is not installed. "
            "Run `pip install ultralytics coremltools`.",
            file=sys.stderr,
        )
        return 1

    candidates = ["yolo26n.pt", "yolov8n.pt"]
    last_err = None
    for weights in candidates:
        try:
            print(f"[export_yolo_coreml] trying {weights}...")
            model = YOLO(weights)
            exported = model.export(
                format="coreml",
                imgsz=args.imgsz,
                # nms=False keeps the raw [B, 4 + nc, N] tensor that the
                # yolo26_post flowunit expects. nms=True wraps the model in
                # a CoreML pipeline with NonMaximumSuppression that produces
                # named outputs incompatible with our post-processor.
                nms=False,
                half=False,
            )
            if isinstance(exported, (list, tuple)):
                exported = exported[0]
            exported = str(exported)

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            if os.path.abspath(exported) != out_path:
                if os.path.isdir(exported):
                    shutil.copytree(exported, out_path)
                else:
                    shutil.copyfile(exported, out_path)
            print(f"[export_yolo_coreml] wrote {out_path}")
            return 0
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(
                f"[export_yolo_coreml] {weights} failed: {e}", file=sys.stderr
            )
            continue

    print(
        f"[export_yolo_coreml] all candidates failed; last error: {last_err}",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
