#!/usr/bin/env python3
"""Export the TrackNet PyTorch checkpoint to ONNX for TensorRT consumption.

Loads the model definition from the user's TrackNet checkout (defaults to
~/tracknet) and exports a [1, 9, 288, 512] → [1, 3, 288, 512] ONNX file.

Usage:
    python3 export_tracknet_onnx.py \\
        --checkpoint ~/tracknet/checkpoints/model.pt \\
        --tracknet_repo ~/tracknet \\
        --output assets/tennis_action/tracknet_deep.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tracknet_repo", default=str(Path.home() / "tracknet"))
    p.add_argument("--output", required=True)
    p.add_argument("--height", type=int, default=288)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    sys.path.insert(0, args.tracknet_repo)
    try:
        from model import TrackNet  # type: ignore
    except ImportError as exc:
        print(f"[fatal] cannot import TrackNet from {args.tracknet_repo}: "
              f"{exc}", file=sys.stderr)
        return 1

    model = TrackNet()
    sd = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd)
    model.eval()

    dummy = torch.zeros(1, 9, args.height, args.width)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(output_path),
        input_names=["input"], output_names=["output"],
        opset_version=args.opset,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
