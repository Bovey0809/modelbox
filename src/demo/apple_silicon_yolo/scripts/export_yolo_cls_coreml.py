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

"""Export Ultralytics YOLO-classification to a Core ML .mlpackage with a
RAW [1, 1000] tensor output (no Apple class-label/dict wrapping).

Ultralytics' default `format=coreml` for cls models applies a class-label
post-processor that turns the output into MLFeatureType image-classifier
(string + dictionary) — we want a plain MLMultiArray of softmax probs so
the modelbox yolo_cls_post flowunit can read [1, 1000] floats and pick
top-K. We bypass Ultralytics' wrapper, trace the bare nn.Module
(`yolo.model`) with check_trace=False (its forward has data-dependent
shapes), and convert via coremltools.convert with TensorType input.

Usage:
    python export_yolo_cls_coreml.py --out path/to/yolov8n-cls.mlpackage
"""

import argparse
import os
import shutil
import sys


def _trace_and_convert(out_path, imgsz):
    import coremltools as ct
    import torch
    from ultralytics import YOLO

    yolo = YOLO("yolov8n-cls.pt")
    inner = yolo.model.eval()

    class ProbsOnly(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            out = self.m(x)
            if isinstance(out, (list, tuple)):
                out = out[0]
            return out

    wrapped = ProbsOnly(inner)
    example = torch.zeros(1, 3, imgsz, imgsz)
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapped, example, strict=False, check_trace=False
        )
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=example.shape)],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,
    )
    if os.path.exists(out_path):
        shutil.rmtree(out_path) if os.path.isdir(out_path) else os.remove(out_path)
    mlmodel.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--imgsz", type=int, default=224)
    args = parser.parse_args()

    out_path = os.path.abspath(args.out)
    if os.path.isdir(out_path) and os.path.exists(
            os.path.join(out_path, "Manifest.json")):
        print(f"[cls] {out_path} already exists, skipping.")
        return 0

    try:
        import coremltools  # noqa: F401
        import torch  # noqa: F401
        from ultralytics import YOLO  # noqa: F401
    except ImportError as e:
        print(f"[cls] missing dep: {e}", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    _trace_and_convert(out_path, args.imgsz)
    print(f"[cls] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
