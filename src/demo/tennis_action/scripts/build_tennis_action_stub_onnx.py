#!/usr/bin/env python3
"""Generate tiny deterministic ONNX models for the tennis_action --cpu_stub
graph test. The models do not represent real networks; they exist so the
inference flowunits can be loaded and the graph topology can be exercised
without TensorRT or real weights.

Usage:
    python build_tennis_action_stub_onnx.py --out_dir assets/tennis_action/

Produces:
    yolov8n-pose.onnx     input [1,3,640,640] f32 -> output [1,56,8400] f32
    tracknet_deep.onnx    input [1,9,288,512] f32 -> output [1,3,288,512] f32
    asformer.onnx         input [1,32,51]    f32 -> output [1,4] f32
    asformer_meta.json    {classes, num_kpts, T, input_layout}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _build_constant_onnx(name: str, in_shape: list[int], out_shape: list[int],
                         constant_value: float, out_path: Path) -> None:
    # input -> Identity (ignored) -> ConstantOfShape -> output
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, in_shape)
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, out_shape)

    shape_init = numpy_helper.from_array(
        np.array(out_shape, dtype=np.int64), name="out_shape"
    )
    value_attr = helper.make_tensor(
        name="value", data_type=TensorProto.FLOAT, dims=[1], vals=[constant_value]
    )

    # Use the input via Identity -> _ignored just so the graph isn't disconnected.
    node_id = helper.make_node("Identity", ["input"], ["_ignored"])
    node_const = helper.make_node(
        "ConstantOfShape", ["out_shape"], ["output"], value=value_attr
    )

    graph = helper.make_graph(
        nodes=[node_id, node_const],
        name=name,
        inputs=[inp],
        outputs=[out],
        initializer=[shape_init],
    )
    model = helper.make_model(graph, producer_name="tennis_action_stub",
                              opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # yolov8n-pose stub: pretend one person at (0.5W, 0.5H) with all kpts at center.
    # ConstantOfShape doesn't let us encode that semantically — the post-processor
    # is the Python flowunit's responsibility; the stub just needs the right shape.
    _build_constant_onnx("yolov8n_pose_stub", [1, 3, 640, 640], [1, 56, 8400],
                         0.0, out / "yolov8n-pose.onnx")

    # tracknet_deep stub: zero heatmap (= "no ball detected" path).
    _build_constant_onnx("tracknet_deep_stub", [1, 9, 288, 512], [1, 3, 288, 512],
                         0.0, out / "tracknet_deep.onnx")

    # asformer stub: pretend a 4-class classifier; emit a fixed argmax-0 logit.
    _build_constant_onnx("asformer_stub", [1, 32, 51], [1, 4],
                         0.0, out / "asformer.onnx")

    meta = {
        "classes": ["forehand", "backhand", "serve", "other"],
        "num_kpts": 17,
        "T": 32,
        "input_layout": "BTC",  # (batch=1, T=32, kpts*3=51)
    }
    (out / "asformer_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"Wrote stub ONNX + meta to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
