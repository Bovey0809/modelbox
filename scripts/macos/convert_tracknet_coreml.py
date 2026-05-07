#!/usr/bin/env python3
"""Convert TrackNet PyTorch checkpoint to Core ML .mlpackage.

Run on the `pose` GPU server (or any Linux box with torch + coremltools).
Pulls the upstream model.py via a tempdir git clone — no vendoring.

Default checkpoint: /root/autodl-fs/repos/tracknet/exp_deep/TrackNet_best.pt
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


UPSTREAM = "https://github.com/bcinno-dev/tracknet"


def _unwrap(state) -> dict:
    """Accept either raw state_dict or training-checkpoint dict."""
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        return state["model"]
    return state


def _import_model_module(repo_dir: Path):
    sys.path.insert(0, str(repo_dir))
    import model  # noqa: E402

    return model


def _build_model(model_module, variant: str, seq_len: int):
    if variant == "deep":
        return model_module.TrackNetDeep(seq_len=seq_len)
    if variant == "small":
        return model_module.TrackNet(seq_len=seq_len)
    raise SystemExit(f"unknown variant: {variant!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/root/autodl-fs/repos/tracknet/exp_deep/TrackNet_best.pt")
    ap.add_argument("--variant", choices=["deep", "small"], default="deep")
    ap.add_argument("--seq-len", type=int, default=3)
    ap.add_argument("--height", type=int, default=288)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--out", default="tracknet_deep.mlpackage")
    ap.add_argument(
        "--compute-units",
        choices=["ALL", "CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY"],
        default="ALL",
    )
    ap.add_argument(
        "--repo-dir",
        default=None,
        help="Path to a pre-cloned tracknet repo. If unset, clone from upstream.",
    )
    args = ap.parse_args()

    import coremltools as ct  # imported here so --help works without coremltools installed

    with tempfile.TemporaryDirectory() as td:
        if args.repo_dir:
            repo = Path(args.repo_dir).resolve()
            if not (repo / "model.py").is_file():
                raise SystemExit(f"--repo-dir {repo} missing model.py")
        else:
            repo = Path(td) / "tracknet"
            subprocess.check_call(["git", "clone", "--depth", "1", UPSTREAM, str(repo)])
        model_module = _import_model_module(repo)

        net = _build_model(model_module, args.variant, args.seq_len)
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        net.load_state_dict(_unwrap(state))
        net.eval()

        in_ch = args.seq_len * 3
        example = torch.randn(1, in_ch, args.height, args.width)
        traced = torch.jit.trace(net, example)

        compute_units = {
            "ALL": ct.ComputeUnit.ALL,
            "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
            "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
            "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        }[args.compute_units]

        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="frames", shape=(1, in_ch, args.height, args.width), dtype=np.float32)],
            outputs=[ct.TensorType(name="heatmaps", dtype=np.float32)],
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS14,
            compute_units=compute_units,
            convert_to="mlprogram",
        )

        out = Path(args.out).resolve()
        if out.exists():
            subprocess.check_call(["rm", "-rf", str(out)])
        mlmodel.save(str(out))

        # Sanity check: PyTorch vs Core ML on 5 random tensors.
        # `predict` only works on macOS; on other platforms, skip gracefully.
        try:
            max_err = 0.0
            for _ in range(5):
                x = torch.randn(1, in_ch, args.height, args.width)
                with torch.no_grad():
                    ref = net(x).numpy()
                cm = mlmodel.predict({"frames": x.numpy()})["heatmaps"]
                max_err = max(max_err, float(np.abs(ref - cm).max()))
            print(f"max |torch - coreml| = {max_err:.4e}")
            if max_err >= 1e-2:
                raise SystemExit(f"conversion sanity check failed: max_err={max_err:.4e}")
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"sanity check skipped: {exc} (run on macOS to verify)")

        # Report size + content hash.
        total = 0
        h = hashlib.sha256()
        for p in sorted(out.rglob("*")):
            if p.is_file():
                total += p.stat().st_size
                with p.open("rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
        print(f"saved: {out}  bytes={total}  sha256={h.hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
