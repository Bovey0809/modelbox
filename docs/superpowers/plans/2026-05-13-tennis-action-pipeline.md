# Tennis Action Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single offline ModelBox graph on the GPU pose server that consumes a tennis video and emits per-hit action classes (via ASFormer) plus an overlay mp4 with skeleton + ball + action labels burned in.

**Architecture:** Three-branch graph (audio hit-sound classifier collapses to a hit-center list; TrackNet streams per-frame ball positions; YOLO-pose streams tracked skeletons). A fusion flowunit cross-confirms each audio hit against the ball trajectory, assigns the hitter (closest wrist to ball), assembles a T-frame keypoint window of that track, and feeds it to ASFormer. A sink writes JSON + overlay mp4 at session end.

**Tech Stack:** Python (`numpy`, `opencv-python`, stdlib) for all new flowunits — C++ deferred to v2. CUDA + TensorRT for `yolo_pose_detect` / `tracknet_deep` virtual inference; ONNX Runtime (CPU) for ASFormer and the existing `hit_sound_infer_ort`. `nvcodec` video decode, `nppi_resize`, `normalize` on cuda; cv2 + ffmpeg for the overlay encoder. Build is CMake; tests are stub-`_flowunit` Python smoke tests (matching `src/demo/hit_sound/flowunit/hit_sound_detect/offline_smoke_test.py`) plus one graph-level test that runs `modelbox-tool flow run`.

**Spec:** `docs/superpowers/specs/2026-05-13-tennis-action-pipeline-design.md` (already on branch).

**Branch:** `feat/tennis-action-pipeline` (off `origin/main`, currently at commit `de9bccf`).

---

## File structure (created by this plan)

```
src/demo/tennis_action/
├── CMakeLists.txt
├── flowunit/
│   ├── CMakeLists.txt
│   ├── yolo_pose_track_post/
│   │   ├── CMakeLists.txt
│   │   ├── yolo_pose_track_post.py
│   │   ├── yolo_pose_track_post.toml
│   │   └── offline_smoke_test.py
│   ├── tracknet_ball_emitter/
│   │   ├── CMakeLists.txt
│   │   ├── tracknet_ball_emitter.py
│   │   ├── tracknet_ball_emitter.toml
│   │   └── offline_smoke_test.py
│   ├── hit_centers_emitter/
│   │   ├── CMakeLists.txt
│   │   ├── hit_centers_emitter.py
│   │   ├── hit_centers_emitter.toml
│   │   └── offline_smoke_test.py
│   ├── hit_fuse/
│   │   ├── CMakeLists.txt
│   │   ├── hit_fuse.py
│   │   ├── hit_fuse.toml
│   │   └── offline_smoke_test.py
│   ├── asformer_infer_ort/
│   │   ├── CMakeLists.txt
│   │   └── asformer_infer_ort.toml
│   └── action_sink/
│       ├── CMakeLists.txt
│       ├── action_sink.py
│       ├── action_sink.toml
│       └── offline_smoke_test.py
├── graph/
│   ├── CMakeLists.txt
│   ├── tennis_action_cuda.toml.in
│   ├── tennis_action_cpu_stub.toml.in
│   └── test_tennis_action.py
├── scripts/
│   ├── export_tracknet_onnx.py
│   └── build_tennis_action_stub_onnx.py
└── test/
    └── tennis_test.mp4               # moved from repo root
```

**Each flowunit dir uses the same `CMakeLists.txt` template** (mirrors `src/demo/hit_sound/flowunit/hit_window_emitter/CMakeLists.txt`):

```cmake
#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
cmake_minimum_required(VERSION 3.10)

set(FLOWUNIT_NAME "<flowunit_name>")
set(FLOWUNIT_PATH ${CMAKE_CURRENT_BINARY_DIR}/${FLOWUNIT_NAME})
file(MAKE_DIRECTORY ${FLOWUNIT_PATH})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FLOWUNIT_NAME}.py
               ${FLOWUNIT_PATH}/${FLOWUNIT_NAME}.py @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FLOWUNIT_NAME}.toml
               ${FLOWUNIT_PATH}/${FLOWUNIT_NAME}.toml @ONLY)

install(DIRECTORY ${FLOWUNIT_PATH}
    DESTINATION ${DEMO_TENNIS_ACTION_FLOWUNIT_DIR}
    COMPONENT demo
)
```

Pure-virtual inference flowunits (no `.py`) drop the second `configure_file()`.

---

## Task 1: Scaffold demo dir, move test video, register in parent build

**Files:**
- Create: `src/demo/tennis_action/CMakeLists.txt`
- Create: `src/demo/tennis_action/flowunit/CMakeLists.txt`
- Create: `src/demo/tennis_action/graph/CMakeLists.txt` (empty placeholder for now)
- Create: `src/demo/tennis_action/test/.gitkeep`
- Move: `tennis_test.mp4` → `src/demo/tennis_action/test/tennis_test.mp4`

- [ ] **Step 1: Create the demo top-level CMakeLists.txt**

```bash
mkdir -p src/demo/tennis_action/{flowunit,graph,scripts,test}
```

Write `src/demo/tennis_action/CMakeLists.txt`:

```cmake
#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
cmake_minimum_required(VERSION 3.10)

subdirlist(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR} "CMakeLists.txt")

set(DEMO_TENNIS_ACTION_DIR "${MODELBOX_DEMO_INSTALL_DIR}/tennis_action" CACHE INTERNAL "")
set(DEMO_TENNIS_ACTION_FLOWUNIT_DIR ${DEMO_TENNIS_ACTION_DIR}/flowunit CACHE INTERNAL "")
set(DEMO_TENNIS_ACTION_GRAPH_DIR ${DEMO_TENNIS_ACTION_DIR}/graph CACHE INTERNAL "")
set(DEMO_TENNIS_ACTION_ASSETS_DIR ${DEMO_TENNIS_ACTION_DIR}/assets CACHE INTERNAL "")
set(DEMO_TENNIS_ACTION_VIDEO "${CMAKE_CURRENT_SOURCE_DIR}/test/tennis_test.mp4" CACHE INTERNAL "")

# Install the test video alongside the demo so the graph TOML works post-install.
install(FILES ${DEMO_TENNIS_ACTION_VIDEO}
    DESTINATION ${DEMO_TENNIS_ACTION_DIR}/test
    COMPONENT demo
)

foreach(subdir ${SUBDIRS})
    add_subdirectory(${subdir})
endforeach()
```

- [ ] **Step 2: Create the flowunit + graph CMakeLists placeholders**

Write `src/demo/tennis_action/flowunit/CMakeLists.txt`:

```cmake
#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
cmake_minimum_required(VERSION 3.10)

subdirlist(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR} "CMakeLists.txt")

foreach(subdir ${SUBDIRS})
    add_subdirectory(${subdir})
endforeach()
```

Write `src/demo/tennis_action/graph/CMakeLists.txt`:

```cmake
#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
cmake_minimum_required(VERSION 3.10)

# Placeholder. Graphs will be added in the graph wiring task.
```

`touch src/demo/tennis_action/test/.gitkeep` and `touch src/demo/tennis_action/scripts/.gitkeep`.

- [ ] **Step 3: Move test video into the demo tree**

```bash
git mv tennis_test.mp4 src/demo/tennis_action/test/tennis_test.mp4 2>/dev/null \
  || mv tennis_test.mp4 src/demo/tennis_action/test/tennis_test.mp4
ls -la src/demo/tennis_action/test/tennis_test.mp4
```

Expected output: `-rw-r--r-- 1 rick rick 542143 ... src/demo/tennis_action/test/tennis_test.mp4`

- [ ] **Step 4: Verify cmake configure picks up the new subdir**

```bash
cd build 2>/dev/null || mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DWITH_ALL_DEMO=ON 2>&1 | grep -i tennis_action
cd ..
```

Expected: at least one line mentioning `tennis_action` (a subdir discovery log). If you don't have a build dir yet and configure fails for other reasons, just verify `grep -r tennis_action build/CMakeFiles/ 2>/dev/null | head` lists files.

- [ ] **Step 5: Commit**

```bash
git add src/demo/tennis_action/
git commit -m "$(cat <<'EOF'
demo: scaffold tennis_action directory + move test video

Empty CMake skeleton plus the 542 KB test clip moved from repo root to
src/demo/tennis_action/test/. Demo subdir auto-discovered via subdirlist;
exports DEMO_TENNIS_ACTION_{FLOWUNIT,GRAPH,ASSETS}_DIR + DEMO_TENNIS_ACTION_VIDEO
CMake variables for downstream tasks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Stub-ONNX builder script

This script generates three minimal ONNX models (with deterministic constant outputs) so the `--cpu_stub` graph test can run end-to-end without TensorRT / real models. Generated, not committed.

**Files:**
- Create: `src/demo/tennis_action/scripts/build_tennis_action_stub_onnx.py`

- [ ] **Step 1: Write the stub builder**

Write `src/demo/tennis_action/scripts/build_tennis_action_stub_onnx.py`:

```python
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
```

- [ ] **Step 2: Run the script and verify outputs**

```bash
python src/demo/tennis_action/scripts/build_tennis_action_stub_onnx.py \
    --out_dir /tmp/stub_assets
ls -la /tmp/stub_assets/
python -c "import onnx; m=onnx.load('/tmp/stub_assets/yolov8n-pose.onnx'); print(m.graph.input[0].type.tensor_type.shape); print(m.graph.output[0].type.tensor_type.shape)"
```

Expected: 3 `.onnx` files + `asformer_meta.json`. The python check should print input shape `[1,3,640,640]` and output shape `[1,56,8400]`.

- [ ] **Step 3: Commit**

```bash
git add src/demo/tennis_action/scripts/build_tennis_action_stub_onnx.py
# remove the .gitkeep we no longer need now that the dir has content
git rm src/demo/tennis_action/scripts/.gitkeep 2>/dev/null || true
git commit -m "$(cat <<'EOF'
demo/tennis_action: stub ONNX builder for CPU-only graph testing

Generates yolov8n-pose.onnx + tracknet_deep.onnx + asformer.onnx with
deterministic constant outputs (ConstantOfShape op). Lets the --cpu_stub
graph variant exercise pipeline wiring without TensorRT or real weights.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `tracknet_ball_emitter` flowunit

Simplest of the new flowunits. Pure numpy on the heatmap. Algorithm mirrors `extern "C" TrackNetExtractCentroid()` in `src/demo/apple_silicon_yolo/flowunit/tracknet_post/tracknet_post_flowunit.cc`.

**Files:**
- Create: `src/demo/tennis_action/flowunit/tracknet_ball_emitter/CMakeLists.txt`
- Create: `src/demo/tennis_action/flowunit/tracknet_ball_emitter/tracknet_ball_emitter.toml`
- Create: `src/demo/tennis_action/flowunit/tracknet_ball_emitter/tracknet_ball_emitter.py`
- Create: `src/demo/tennis_action/flowunit/tracknet_ball_emitter/offline_smoke_test.py`

- [ ] **Step 1: Write the failing offline test**

Write `src/demo/tennis_action/flowunit/tracknet_ball_emitter/offline_smoke_test.py`:

```python
#!/usr/bin/env python3
"""Offline smoke test for tracknet_ball_emitter. Stubs the _flowunit module
so we can import and exercise the Python class without ModelBox linkage."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

_FU = types.ModuleType("_flowunit")
_FU.Buffer = lambda dev, b: ("BUFFER", b)  # opaque
class _SC: STATUS_SUCCESS = 0; STATUS_FAULT = 1
class _Status:
    StatusCode = _SC
    def __init__(self, *a, **k): pass
_FU.Status = _Status
_FU.FlowUnit = type("FlowUnit", (), {"__init__": lambda self: None,
                                     "get_bind_device": lambda self: None})
_FU.info = lambda msg: None
_FU.error = lambda msg: print(f"[ERR] {msg}", file=sys.stderr)
sys.modules["_flowunit"] = _FU

sys.path.insert(0, str(Path(__file__).parent))
from tracknet_ball_emitter import TracknetBallEmitter, extract_centroid  # noqa: E402


def test_centroid_above_threshold():
    """A single hot pixel in channel 1 → (cx, cy) reported in net coords."""
    heat = np.zeros((3, 288, 512), dtype=np.float32)
    heat[1, 100, 200] = 1.0  # peak at (y=100, x=200) on channel 1
    cx, cy, peak = extract_centroid(heat, score_thr=0.3, mask_ratio=0.5,
                                    net_h=288, net_w=512)
    assert peak >= 0.99, f"peak={peak}"
    assert abs(cx - 200) < 1.0, f"cx={cx}"
    assert abs(cy - 100) < 1.0, f"cy={cy}"
    print("test_centroid_above_threshold: PASS")


def test_below_threshold_returns_sentinel():
    """All-zero heatmap → cx=cy=-1."""
    heat = np.zeros((3, 288, 512), dtype=np.float32)
    cx, cy, peak = extract_centroid(heat, score_thr=0.3, mask_ratio=0.5,
                                    net_h=288, net_w=512)
    assert peak < 0.3
    assert cx == -1.0 and cy == -1.0, f"({cx},{cy}) instead of (-1,-1)"
    print("test_below_threshold_returns_sentinel: PASS")


def test_scaling_to_source_coords():
    """Peak at net center → source center (configured via source_width/height)."""
    heat = np.zeros((3, 288, 512), dtype=np.float32)
    heat[1, 144, 256] = 0.9  # net-resolution center
    cx, cy, peak = extract_centroid(heat, score_thr=0.3, mask_ratio=0.5,
                                    net_h=288, net_w=512)
    sx = cx * (1280 / 512)
    sy = cy * (720 / 288)
    assert abs(sx - 640) < 2.0 and abs(sy - 360) < 2.0, f"({sx},{sy})"
    print("test_scaling_to_source_coords: PASS")


def test_flowunit_open_reads_config():
    """The flowunit picks up config defaults correctly."""
    class _Cfg:
        def __init__(self, d): self.d = d
        def get_int(self, k, default): return int(self.d.get(k, default))
        def get_float(self, k, default): return float(self.d.get(k, default))
        def get_string(self, k, default): return str(self.d.get(k, default))
    fu = TracknetBallEmitter()
    rc = fu.open(_Cfg({"source_width": 1280, "source_height": 720,
                       "score_thr": 0.3, "net_h": 288, "net_w": 512}))
    assert rc == _SC.STATUS_SUCCESS
    assert fu.source_width == 1280 and fu.source_height == 720
    print("test_flowunit_open_reads_config: PASS")


def main() -> int:
    test_centroid_above_threshold()
    test_below_threshold_returns_sentinel()
    test_scaling_to_source_coords()
    test_flowunit_open_reads_config()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run test → expect ImportError (the module doesn't exist yet)**

```bash
cd src/demo/tennis_action/flowunit/tracknet_ball_emitter
python offline_smoke_test.py
```

Expected: `ModuleNotFoundError: No module named 'tracknet_ball_emitter'`.

- [ ] **Step 3: Write the flowunit implementation**

Write `src/demo/tennis_action/flowunit/tracknet_ball_emitter/tracknet_ball_emitter.py`:

```python
#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
"""tracknet_ball_emitter — consume a per-frame TrackNet heatmap (CHW float),
extract the ball centroid, and emit a structured (cx, cy, peak, frame_idx)
buffer in source-frame coordinates.

Mirrors the algorithm in
src/demo/apple_silicon_yolo/flowunit/tracknet_post/tracknet_post_flowunit.cc
(extern "C" TrackNetExtractCentroid). C++ port deferred to v2.
"""

from __future__ import annotations

import numpy as np
import _flowunit as modelbox


def extract_centroid(heat: np.ndarray, score_thr: float, mask_ratio: float,
                     net_h: int, net_w: int) -> tuple[float, float, float]:
    """Argmax across the non-background channels, masked-centroid within a
    `mask_ratio * min(net_h, net_w)` window around the peak.

    Returns (cx, cy, peak) in net-resolution coordinates. When peak<score_thr,
    returns (-1.0, -1.0, peak).
    """
    if heat.ndim != 3:
        return (-1.0, -1.0, 0.0)
    # The model emits 3 channels; channels 1+ are the ball heatmap (channel 0
    # is background). Take the max across foreground channels.
    fg = heat[1:].max(axis=0) if heat.shape[0] >= 2 else heat[0]
    peak_idx = int(np.argmax(fg))
    peak_y, peak_x = divmod(peak_idx, fg.shape[1])
    peak = float(fg[peak_y, peak_x])
    if peak < score_thr:
        return (-1.0, -1.0, peak)
    win = int(mask_ratio * min(net_h, net_w))
    y0 = max(0, peak_y - win // 2)
    y1 = min(fg.shape[0], peak_y + win // 2 + 1)
    x0 = max(0, peak_x - win // 2)
    x1 = min(fg.shape[1], peak_x + win // 2 + 1)
    patch = fg[y0:y1, x0:x1]
    mask = patch >= (peak * 0.5)
    if not mask.any():
        return (float(peak_x), float(peak_y), peak)
    ys, xs = np.where(mask)
    weights = patch[ys, xs]
    cy = (ys + y0).astype(np.float32) @ weights / weights.sum()
    cx = (xs + x0).astype(np.float32) @ weights / weights.sum()
    return (float(cx), float(cy), peak)


class TracknetBallEmitter(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()
        self.frame_idx = 0

    def open(self, config):
        self.score_thr = config.get_float("score_thr", 0.3)
        self.mask_ratio = config.get_float("mask_ratio", 0.5)
        self.net_h = config.get_int("net_h", 288)
        self.net_w = config.get_int("net_w", 512)
        self.source_width = config.get_int("source_width", 1280)
        self.source_height = config.get_int("source_height", 720)
        self.scale_x = self.source_width / self.net_w
        self.scale_y = self.source_height / self.net_h
        self.frame_idx = 0
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def process(self, data_context):
        heatmaps = data_context.input("heatmaps")
        ball_out = data_context.output("ball_pos")
        for buf in heatmaps:
            arr = np.frombuffer(buf.as_object(), dtype=np.float32)
            try:
                arr = arr.reshape(3, self.net_h, self.net_w)
            except ValueError:
                modelbox.error(
                    f"tracknet_ball_emitter: cannot reshape heatmap of size "
                    f"{arr.size} into (3,{self.net_h},{self.net_w})")
                return modelbox.Status.StatusCode.STATUS_FAULT
            cx, cy, peak = extract_centroid(
                arr, self.score_thr, self.mask_ratio, self.net_h, self.net_w
            )
            if cx >= 0:
                cx *= self.scale_x
                cy *= self.scale_y
            out = np.array([cx, cy, peak, float(self.frame_idx)], dtype=np.float32)
            ob = modelbox.Buffer(self.get_bind_device(), out.tobytes())
            ob.set("frame_idx", int(self.frame_idx))
            ball_out.push_back(ob)
            self.frame_idx += 1
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def close(self):
        return modelbox.Status()

    def data_pre(self, data_context):
        return modelbox.Status()

    def data_post(self, data_context):
        return modelbox.Status()
```

- [ ] **Step 4: Run test → expect PASS for all 4**

```bash
cd src/demo/tennis_action/flowunit/tracknet_ball_emitter
python offline_smoke_test.py
```

Expected:
```
test_centroid_above_threshold: PASS
test_below_threshold_returns_sentinel: PASS
test_scaling_to_source_coords: PASS
test_flowunit_open_reads_config: PASS
```

- [ ] **Step 5: Write the TOML descriptor**

Write `src/demo/tennis_action/flowunit/tracknet_ball_emitter/tracknet_ball_emitter.toml`:

```toml
#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
[base]
name = "tracknet_ball_emitter"
device = "cpu"
version = "1.0.0"
description = "Emit (cx, cy, peak, frame_idx) per frame from a TrackNet heatmap in source-frame coords."
entry = "tracknet_ball_emitter@TracknetBallEmitter"
type = "python"

[config]
score_thr = 0.3
mask_ratio = 0.5
net_h = 288
net_w = 512
source_width = 1280
source_height = 720

[input]
[input.input1]
name = "heatmaps"

[output]
[output.output1]
name = "ball_pos"
```

- [ ] **Step 6: Write the CMakeLists.txt**

Write `src/demo/tennis_action/flowunit/tracknet_ball_emitter/CMakeLists.txt`:

```cmake
#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
cmake_minimum_required(VERSION 3.10)

set(FLOWUNIT_NAME "tracknet_ball_emitter")
set(FLOWUNIT_PATH ${CMAKE_CURRENT_BINARY_DIR}/${FLOWUNIT_NAME})
file(MAKE_DIRECTORY ${FLOWUNIT_PATH})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FLOWUNIT_NAME}.py
               ${FLOWUNIT_PATH}/${FLOWUNIT_NAME}.py @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FLOWUNIT_NAME}.toml
               ${FLOWUNIT_PATH}/${FLOWUNIT_NAME}.toml @ONLY)

install(DIRECTORY ${FLOWUNIT_PATH}
    DESTINATION ${DEMO_TENNIS_ACTION_FLOWUNIT_DIR}
    COMPONENT demo
)
```

- [ ] **Step 7: Commit**

```bash
git add src/demo/tennis_action/flowunit/tracknet_ball_emitter/
git commit -m "$(cat <<'EOF'
flowunit: tracknet_ball_emitter — structured ball position per frame

Python flowunit that consumes the per-frame TrackNet heatmap and emits
(cx, cy, peak, frame_idx) in source-frame coordinates. Algorithm mirrors
TrackNetExtractCentroid() in the existing C++ tracknet_post. Includes
offline smoke test using the stub _flowunit module pattern.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `yolo_pose_track_post` flowunit

NMS + greedy-IoU SORT in numpy. Algorithm mirrors `Decode()` / `Nms()` in `yolo_pose_post_flowunit.cc` and `UpdateTracks()` in `yolo_track_post_flowunit.cc`.

**Files:**
- Create: `src/demo/tennis_action/flowunit/yolo_pose_track_post/{CMakeLists.txt,yolo_pose_track_post.toml,yolo_pose_track_post.py,offline_smoke_test.py}`

- [ ] **Step 1: Write the failing offline test**

Write `src/demo/tennis_action/flowunit/yolo_pose_track_post/offline_smoke_test.py`:

```python
#!/usr/bin/env python3
"""Offline smoke test for yolo_pose_track_post."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

_FU = types.ModuleType("_flowunit")
_FU.Buffer = lambda dev, b: ("BUFFER", b)
class _SC: STATUS_SUCCESS = 0; STATUS_FAULT = 1
class _Status:
    StatusCode = _SC
    def __init__(self, *a, **k): pass
_FU.Status = _Status
_FU.FlowUnit = type("FlowUnit", (), {"__init__": lambda self: None,
                                     "get_bind_device": lambda self: None})
_FU.info = lambda msg: None
_FU.error = lambda msg: print(f"[ERR] {msg}", file=sys.stderr)
sys.modules["_flowunit"] = _FU

sys.path.insert(0, str(Path(__file__).parent))
from yolo_pose_track_post import (
    YoloPoseTrackPost, decode_pose, nms_pose, GreedyIoUTracker
)  # noqa: E402


def _make_feat_two_people(W: int = 640) -> np.ndarray:
    """Construct a synthetic [1, 56, N] feature tensor with two clear persons."""
    N = 8400
    feat = np.zeros((1, 56, N), dtype=np.float32)
    # Person 1: bbox center (160, 320), w=h=200, score 0.9, kpts at center.
    feat[0, 0, 0] = 160.0
    feat[0, 1, 0] = 320.0
    feat[0, 2, 0] = 200.0
    feat[0, 3, 0] = 200.0
    feat[0, 4, 0] = 0.9
    for k in range(17):
        feat[0, 5 + 3 * k + 0, 0] = 160.0
        feat[0, 5 + 3 * k + 1, 0] = 320.0
        feat[0, 5 + 3 * k + 2, 0] = 0.8
    # Person 2: bbox center (480, 320).
    feat[0, 0, 1] = 480.0
    feat[0, 1, 1] = 320.0
    feat[0, 2, 1] = 200.0
    feat[0, 3, 1] = 200.0
    feat[0, 4, 1] = 0.85
    for k in range(17):
        feat[0, 5 + 3 * k + 0, 1] = 480.0
        feat[0, 5 + 3 * k + 1, 1] = 320.0
        feat[0, 5 + 3 * k + 2, 1] = 0.8
    return feat


def test_decode_extracts_two_persons():
    feat = _make_feat_two_people()
    persons = decode_pose(feat, conf_threshold=0.25, scale_x=1.0, scale_y=1.0)
    assert len(persons) == 2, f"got {len(persons)} persons"
    print("test_decode_extracts_two_persons: PASS")


def test_nms_keeps_two_distinct_persons():
    feat = _make_feat_two_people()
    persons = decode_pose(feat, conf_threshold=0.25, scale_x=1.0, scale_y=1.0)
    kept = nms_pose(persons, iou_threshold=0.45)
    assert len(kept) == 2, f"NMS kept {len(kept)}"
    print("test_nms_keeps_two_distinct_persons: PASS")


def test_tracker_assigns_stable_ids_across_frames():
    tracker = GreedyIoUTracker(track_iou=0.3, max_lost=20)
    feat = _make_feat_two_people()
    ids_seen = []
    for _ in range(5):
        persons = nms_pose(decode_pose(feat, 0.25, 1.0, 1.0), 0.45)
        tracked = tracker.update(persons)
        ids_seen.append(sorted(t["track_id"] for t in tracked))
    # IDs must be stable across the 5 calls.
    assert all(s == ids_seen[0] for s in ids_seen), f"ID drift: {ids_seen}"
    print("test_tracker_assigns_stable_ids_across_frames: PASS")


def test_tracker_recycles_after_max_lost():
    tracker = GreedyIoUTracker(track_iou=0.3, max_lost=2)
    feat = _make_feat_two_people()
    tracker.update(nms_pose(decode_pose(feat, 0.25, 1.0, 1.0), 0.45))
    for _ in range(5):
        tracker.update([])  # no detections
    # After max_lost+ empty frames, tracks have been pruned.
    assert len(tracker.tracks) == 0, f"{len(tracker.tracks)} tracks remain"
    print("test_tracker_recycles_after_max_lost: PASS")


def test_open_reads_config_defaults():
    class _Cfg:
        def get_int(self, k, d): return d
        def get_float(self, k, d): return d
        def get_string(self, k, d): return d
        def get_bool(self, k, d): return d
    fu = YoloPoseTrackPost()
    rc = fu.open(_Cfg())
    assert rc == _SC.STATUS_SUCCESS
    assert fu.net_h == 640 and fu.net_w == 640
    print("test_open_reads_config_defaults: PASS")


def main() -> int:
    test_decode_extracts_two_persons()
    test_nms_keeps_two_distinct_persons()
    test_tracker_assigns_stable_ids_across_frames()
    test_tracker_recycles_after_max_lost()
    test_open_reads_config_defaults()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run test → expect ImportError**

```bash
cd src/demo/tennis_action/flowunit/yolo_pose_track_post
python offline_smoke_test.py
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write the flowunit implementation**

Write `src/demo/tennis_action/flowunit/yolo_pose_track_post/yolo_pose_track_post.py`:

```python
#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
"""yolo_pose_track_post — pose post-processing + greedy-IoU SORT tracker.

Algorithm mirrors:
  - src/drivers/devices/cpu/flowunit/yolo_pose_post/yolo_pose_post_flowunit.cc
    (Decode + NMS for the [1, 5+17*3, N] anchor-free YOLOv8-pose tensor).
  - src/drivers/devices/cpu/flowunit/yolo_track_post/yolo_track_post_flowunit.cc
    (greedy-IoU SORT-style tracker UpdateTracks).

Emits a structured `tracked_poses` JSON port. C++ port deferred to v2.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import _flowunit as modelbox


def _bbox_iou(a: tuple[float, float, float, float],
              b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def decode_pose(feat: np.ndarray, conf_threshold: float,
                scale_x: float, scale_y: float) -> list[dict[str, Any]]:
    """Decode [1, 56, N] anchor-free YOLOv8-pose output to per-person dicts."""
    assert feat.ndim == 3 and feat.shape[1] == 56, f"got shape {feat.shape}"
    arr = feat[0]
    scores = arr[4]
    keep = np.where(scores >= conf_threshold)[0]
    persons = []
    for idx in keep:
        cx, cy, w, h = arr[0, idx], arr[1, idx], arr[2, idx], arr[3, idx]
        x1 = (cx - w / 2.0) * scale_x
        y1 = (cy - h / 2.0) * scale_y
        x2 = (cx + w / 2.0) * scale_x
        y2 = (cy + h / 2.0) * scale_y
        kpts = []
        for k in range(17):
            kx = arr[5 + 3 * k + 0, idx] * scale_x
            ky = arr[5 + 3 * k + 1, idx] * scale_y
            kc = arr[5 + 3 * k + 2, idx]
            kpts.append([float(kx), float(ky), float(kc)])
        persons.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "score": float(scores[idx]),
            "kpts": kpts,
        })
    return persons


def nms_pose(persons: list[dict[str, Any]], iou_threshold: float) \
        -> list[dict[str, Any]]:
    if not persons:
        return []
    ordered = sorted(persons, key=lambda p: -p["score"])
    kept: list[dict[str, Any]] = []
    while ordered:
        head = ordered.pop(0)
        kept.append(head)
        ordered = [p for p in ordered
                   if _bbox_iou(head["bbox"], p["bbox"]) < iou_threshold]
    return kept


class GreedyIoUTracker:
    def __init__(self, track_iou: float, max_lost: int):
        self.track_iou = track_iou
        self.max_lost = max_lost
        self.tracks: list[dict[str, Any]] = []
        self.next_id = 0

    def update(self, persons: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Greedy: for each existing track, pick best-IoU unmatched detection.
        matched_dets = set()
        for tr in self.tracks:
            best_iou = 0.0
            best_idx = -1
            for di, det in enumerate(persons):
                if di in matched_dets:
                    continue
                iou = _bbox_iou(tr["bbox"], det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = di
            if best_idx >= 0 and best_iou >= self.track_iou:
                tr["bbox"] = persons[best_idx]["bbox"]
                tr["score"] = persons[best_idx]["score"]
                tr["kpts"] = persons[best_idx]["kpts"]
                tr["hits"] += 1
                tr["lost"] = 0
                matched_dets.add(best_idx)
            else:
                tr["lost"] += 1

        # Spawn new tracks for unmatched detections.
        for di, det in enumerate(persons):
            if di in matched_dets:
                continue
            self.tracks.append({
                "track_id": self.next_id,
                "bbox": det["bbox"],
                "score": det["score"],
                "kpts": det["kpts"],
                "hits": 1,
                "lost": 0,
            })
            self.next_id += 1

        # Prune lost tracks.
        self.tracks = [t for t in self.tracks if t["lost"] <= self.max_lost]
        return [{"track_id": t["track_id"], "bbox": t["bbox"],
                 "score": t["score"], "kpts": t["kpts"]}
                for t in self.tracks if t["lost"] == 0]


class YoloPoseTrackPost(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()
        self.tracker = None
        self.frame_idx = 0

    def open(self, config):
        self.conf_threshold = config.get_float("conf_threshold", 0.25)
        self.iou_threshold = config.get_float("iou_threshold", 0.45)
        self.kpt_threshold = config.get_float("kpt_threshold", 0.5)
        self.track_iou = config.get_float("track_iou", 0.3)
        self.max_lost = config.get_int("max_lost", 20)
        self.net_h = config.get_int("net_h", 640)
        self.net_w = config.get_int("net_w", 640)
        self.source_width = config.get_int("source_width", 1280)
        self.source_height = config.get_int("source_height", 720)
        self.emit_overlay = config.get_bool("emit_overlay", False) \
            if hasattr(config, "get_bool") else False
        self.scale_x = self.source_width / self.net_w
        self.scale_y = self.source_height / self.net_h
        self.tracker = GreedyIoUTracker(self.track_iou, self.max_lost)
        self.frame_idx = 0
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def process(self, data_context):
        feat_in = data_context.input("in_feat")
        poses_out = data_context.output("tracked_poses")
        for buf in feat_in:
            arr = np.frombuffer(buf.as_object(), dtype=np.float32)
            try:
                feat = arr.reshape(1, 56, -1)
            except ValueError:
                modelbox.error(
                    f"yolo_pose_track_post: cannot reshape feat of size "
                    f"{arr.size} into (1, 56, N)")
                return modelbox.Status.StatusCode.STATUS_FAULT
            persons = decode_pose(feat, self.conf_threshold,
                                  self.scale_x, self.scale_y)
            persons = nms_pose(persons, self.iou_threshold)
            tracked = self.tracker.update(persons)
            payload = json.dumps({"frame_idx": self.frame_idx,
                                  "tracks": tracked}).encode("utf-8")
            ob = modelbox.Buffer(self.get_bind_device(), payload)
            ob.set("frame_idx", int(self.frame_idx))
            poses_out.push_back(ob)
            self.frame_idx += 1
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def close(self):
        return modelbox.Status()

    def data_pre(self, data_context):
        return modelbox.Status()

    def data_post(self, data_context):
        return modelbox.Status()
```

- [ ] **Step 4: Run test → expect PASS**

```bash
cd src/demo/tennis_action/flowunit/yolo_pose_track_post
python offline_smoke_test.py
```

Expected: 5 PASS lines.

- [ ] **Step 5: Write the TOML**

Write `src/demo/tennis_action/flowunit/yolo_pose_track_post/yolo_pose_track_post.toml`:

```toml
#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
[base]
name = "yolo_pose_track_post"
device = "cpu"
version = "1.0.0"
description = "YOLO-pose NMS + greedy-IoU SORT tracker; emits structured tracked_poses port."
entry = "yolo_pose_track_post@YoloPoseTrackPost"
type = "python"

[config]
conf_threshold = 0.25
iou_threshold = 0.45
kpt_threshold = 0.5
track_iou = 0.3
max_lost = 20
net_h = 640
net_w = 640
source_width = 1280
source_height = 720
emit_overlay = false

[input]
[input.input1]
name = "in_image"
[input.input2]
name = "in_feat"

[output]
[output.output1]
name = "tracked_poses"
```

- [ ] **Step 6: Write the CMakeLists.txt** (same template as Task 3, swap `FLOWUNIT_NAME`)

Write `src/demo/tennis_action/flowunit/yolo_pose_track_post/CMakeLists.txt`:

```cmake
#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
cmake_minimum_required(VERSION 3.10)

set(FLOWUNIT_NAME "yolo_pose_track_post")
set(FLOWUNIT_PATH ${CMAKE_CURRENT_BINARY_DIR}/${FLOWUNIT_NAME})
file(MAKE_DIRECTORY ${FLOWUNIT_PATH})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FLOWUNIT_NAME}.py
               ${FLOWUNIT_PATH}/${FLOWUNIT_NAME}.py @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FLOWUNIT_NAME}.toml
               ${FLOWUNIT_PATH}/${FLOWUNIT_NAME}.toml @ONLY)

install(DIRECTORY ${FLOWUNIT_PATH}
    DESTINATION ${DEMO_TENNIS_ACTION_FLOWUNIT_DIR}
    COMPONENT demo
)
```

- [ ] **Step 7: Commit**

```bash
git add src/demo/tennis_action/flowunit/yolo_pose_track_post/
git commit -m "$(cat <<'EOF'
flowunit: yolo_pose_track_post — pose decode + greedy-IoU SORT

Python flowunit that re-implements NMS + decode from yolo_pose_post.cc
and tracker update from yolo_track_post.cc in numpy. Emits a structured
tracked_poses JSON port instead of a drawn frame. Existing C++ flowunits
are left untouched. Offline test covers decode, NMS, tracker stability,
and track recycling after max_lost.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `hit_centers_emitter` flowunit

Collapses the per-window audio logit stream into a single hit-center list per session. Reuses smoothing knobs from the existing `hit_intervals_sink_flowunit.cc`.

**Files:**
- Create: `src/demo/tennis_action/flowunit/hit_centers_emitter/{CMakeLists.txt,hit_centers_emitter.toml,hit_centers_emitter.py,offline_smoke_test.py}`

- [ ] **Step 1: Write the failing offline test**

Write `src/demo/tennis_action/flowunit/hit_centers_emitter/offline_smoke_test.py`:

```python
#!/usr/bin/env python3
"""Offline smoke test for hit_centers_emitter."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

_FU = types.ModuleType("_flowunit")
_FU.Buffer = lambda dev, b: ("BUFFER", b)
class _SC: STATUS_SUCCESS = 0; STATUS_FAULT = 1
class _Status:
    StatusCode = _SC
    def __init__(self, *a, **k): pass
_FU.Status = _Status
_FU.FlowUnit = type("FlowUnit", (), {"__init__": lambda self: None,
                                     "get_bind_device": lambda self: None})
_FU.info = lambda msg: None
_FU.error = lambda msg: print(f"[ERR] {msg}", file=sys.stderr)
sys.modules["_flowunit"] = _FU

sys.path.insert(0, str(Path(__file__).parent))
from hit_centers_emitter import smooth_majority, intervals_to_centers  # noqa: E402


def test_smooth_majority_filters_isolated_spikes():
    bin_seq = np.array([0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0])
    out = smooth_majority(bin_seq, window=5)
    assert out[2] == 0, "isolated 1 at idx 2 should be smoothed out"
    assert out[8] == 1, "consecutive 1s should remain"
    print("test_smooth_majority_filters_isolated_spikes: PASS")


def test_intervals_to_centers_two_clusters():
    """Two clear hit clusters at frames 30 and 90 → two centers."""
    # 250 windows × 0.02s = 5s; hits at t=0.6s (idx 30) and t=1.8s (idx 90).
    logits = np.zeros((250, 2), dtype=np.float32)
    logits[:, 0] = 1.0  # "no hit"
    for idx in range(28, 33):
        logits[idx] = [0.0, 1.0]
    for idx in range(88, 93):
        logits[idx] = [0.0, 1.0]
    centers = intervals_to_centers(
        logits, video_fps=30.0, step_sec=0.02, window_sec=0.5,
        threshold=0.5, smooth_win=5, max_hit_dur=1.0
    )
    assert len(centers) == 2, f"got {len(centers)} centers: {centers}"
    # Audio-time center of cluster 1 is ~0.6s + 0.25s = 0.85s -> frame 25.5 -> 25 or 26.
    f0 = centers[0]["frame_idx"]
    f1 = centers[1]["frame_idx"]
    assert 24 <= f0 <= 27, f"f0={f0}"
    assert 84 <= f1 <= 87, f"f1={f1}"
    print("test_intervals_to_centers_two_clusters: PASS")


def test_fps_swap_shifts_frame_idx():
    logits = np.zeros((250, 2), dtype=np.float32)
    logits[:, 0] = 1.0
    for idx in range(48, 53):
        logits[idx] = [0.0, 1.0]
    c25 = intervals_to_centers(logits, video_fps=25.0, step_sec=0.02,
                               window_sec=0.5, threshold=0.5,
                               smooth_win=5, max_hit_dur=1.0)
    c30 = intervals_to_centers(logits, video_fps=30.0, step_sec=0.02,
                               window_sec=0.5, threshold=0.5,
                               smooth_win=5, max_hit_dur=1.0)
    assert len(c25) == 1 and len(c30) == 1
    # 30fps mapping yields a higher frame_idx for the same audio time.
    assert c30[0]["frame_idx"] > c25[0]["frame_idx"], \
        f"{c25[0]} vs {c30[0]}"
    print("test_fps_swap_shifts_frame_idx: PASS")


def main() -> int:
    test_smooth_majority_filters_isolated_spikes()
    test_intervals_to_centers_two_clusters()
    test_fps_swap_shifts_frame_idx()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run test → ImportError**

```bash
cd src/demo/tennis_action/flowunit/hit_centers_emitter
python offline_smoke_test.py
```

- [ ] **Step 3: Write the flowunit**

Write `src/demo/tennis_action/flowunit/hit_centers_emitter/hit_centers_emitter.py`:

```python
#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
"""hit_centers_emitter — collapse the per-window audio logit stream into a
single list of hit-center video-frame indices for the session.

Smoothing + duration filtering constants mirror hit_intervals_sink_flowunit.cc.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import _flowunit as modelbox


def smooth_majority(binary: np.ndarray, window: int) -> np.ndarray:
    """Majority-vote smoother. window must be odd."""
    if window <= 1:
        return binary.astype(np.int32)
    half = window // 2
    out = np.zeros_like(binary, dtype=np.int32)
    pad = np.pad(binary.astype(np.int32), half, mode="edge")
    for i in range(len(binary)):
        win = pad[i:i + window]
        out[i] = 1 if win.sum() * 2 > window else 0
    return out


def intervals_to_centers(logits: np.ndarray, video_fps: float,
                         step_sec: float, window_sec: float,
                         threshold: float, smooth_win: int,
                         max_hit_dur: float) -> list[dict[str, Any]]:
    """Find hit intervals from per-window 2-class logits and map their centers
    to source-video frame indices.

    Each window represents the audio segment [k*step_sec, k*step_sec+window_sec].
    A hit's audio-time center is approximated as midpoint(interval) + window_sec/2.
    """
    if logits.ndim != 2 or logits.shape[1] != 2:
        return []
    # Softmax-ish decision: class 1 dominant ⇒ hit.
    binary = (logits[:, 1] > logits[:, 0]).astype(np.int32)
    smoothed = smooth_majority(binary, max(1, smooth_win | 1))
    centers: list[dict[str, Any]] = []
    in_run = False
    run_start = 0
    for k, val in enumerate(smoothed):
        if val and not in_run:
            in_run = True
            run_start = k
        elif not val and in_run:
            in_run = False
            run_end = k - 1
            _maybe_add_center(logits, run_start, run_end, step_sec,
                              window_sec, max_hit_dur, video_fps, centers)
    if in_run:
        _maybe_add_center(logits, run_start, len(smoothed) - 1, step_sec,
                          window_sec, max_hit_dur, video_fps, centers)
    return centers


def _maybe_add_center(logits, run_start, run_end, step_sec, window_sec,
                      max_hit_dur, video_fps, centers):
    t_start = run_start * step_sec
    t_end = run_end * step_sec + window_sec
    if (t_end - t_start) > max_hit_dur:
        return
    t_center = (t_start + t_end) / 2.0
    frame_idx = int(round(t_center * video_fps))
    audio_conf = float(logits[run_start:run_end + 1, 1].mean())
    centers.append({"frame_idx": frame_idx, "audio_conf": audio_conf,
                    "t_center_sec": t_center})


class HitCentersEmitter(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()
        self._logits: list[np.ndarray] = []

    def open(self, config):
        self.video_fps = config.get_float("video_fps", 30.0)
        self.step_sec = config.get_float("step_sec", 0.02)
        self.window_sec = config.get_float("window_sec", 0.5)
        self.threshold = config.get_float("threshold", 0.5)
        self.smooth_win = config.get_int("smooth_win", 5)
        self.max_hit_dur = config.get_float("max_hit_dur", 1.0)
        self._logits = []
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def process(self, data_context):
        logits_in = data_context.input("logits")
        out = data_context.output("hit_centers")
        for buf in logits_in:
            arr = np.frombuffer(buf.as_object(), dtype=np.float32)
            self._logits.append(arr.reshape(-1, 2))
        # collapse=true: when DataContext signals end-of-session, emit once.
        # In ModelBox the per-buffer process() loop runs repeatedly until
        # upstream is done; we emit only after data_post() has nothing left.
        # That hook is data_post(); push the collapsed buffer there.
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def data_pre(self, data_context):
        self._logits = []
        return modelbox.Status()

    def data_post(self, data_context):
        out = data_context.output("hit_centers")
        if not self._logits:
            payload = json.dumps([]).encode("utf-8")
            ob = modelbox.Buffer(self.get_bind_device(), payload)
            out.push_back(ob)
            return modelbox.Status()
        merged = np.concatenate(self._logits, axis=0)
        centers = intervals_to_centers(
            merged, self.video_fps, self.step_sec, self.window_sec,
            self.threshold, self.smooth_win, self.max_hit_dur
        )
        payload = json.dumps(centers).encode("utf-8")
        ob = modelbox.Buffer(self.get_bind_device(), payload)
        ob.set("num_hits", len(centers))
        out.push_back(ob)
        return modelbox.Status()

    def close(self):
        return modelbox.Status()
```

- [ ] **Step 4: Run test → PASS**

```bash
cd src/demo/tennis_action/flowunit/hit_centers_emitter
python offline_smoke_test.py
```

Expected: 3 PASS lines.

- [ ] **Step 5: Write the TOML**

Write `src/demo/tennis_action/flowunit/hit_centers_emitter/hit_centers_emitter.toml`:

```toml
#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
[base]
name = "hit_centers_emitter"
device = "cpu"
version = "1.0.0"
description = "Collapse per-window hit-sound logits into a single list of (frame_idx, audio_conf) hit centers for the session."
entry = "hit_centers_emitter@HitCentersEmitter"
type = "python"

collapse = true

[config]
video_fps = 30.0
step_sec = 0.02
window_sec = 0.5
threshold = 0.5
smooth_win = 5
max_hit_dur = 1.0

[input]
[input.input1]
name = "logits"

[output]
[output.output1]
name = "hit_centers"
```

- [ ] **Step 6: CMakeLists.txt**

Same template as Tasks 3/4 with `FLOWUNIT_NAME="hit_centers_emitter"`. Write `src/demo/tennis_action/flowunit/hit_centers_emitter/CMakeLists.txt`:

```cmake
#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
cmake_minimum_required(VERSION 3.10)

set(FLOWUNIT_NAME "hit_centers_emitter")
set(FLOWUNIT_PATH ${CMAKE_CURRENT_BINARY_DIR}/${FLOWUNIT_NAME})
file(MAKE_DIRECTORY ${FLOWUNIT_PATH})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FLOWUNIT_NAME}.py
               ${FLOWUNIT_PATH}/${FLOWUNIT_NAME}.py @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FLOWUNIT_NAME}.toml
               ${FLOWUNIT_PATH}/${FLOWUNIT_NAME}.toml @ONLY)

install(DIRECTORY ${FLOWUNIT_PATH}
    DESTINATION ${DEMO_TENNIS_ACTION_FLOWUNIT_DIR}
    COMPONENT demo
)
```

- [ ] **Step 7: Commit**

```bash
git add src/demo/tennis_action/flowunit/hit_centers_emitter/
git commit -m "$(cat <<'EOF'
flowunit: hit_centers_emitter — audio logits → video frame indices

Collapse flowunit: accumulates per-window hit-sound logits over the
session, applies majority-vote smoothing + duration filtering, and emits
one JSON buffer of [{frame_idx, audio_conf, t_center_sec}] at session end.
Mirrors the smoothing constants in hit_intervals_sink_flowunit.cc.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `hit_fuse` flowunit — algorithm core

The pipeline's brain — splits into four sub-tasks (cross-confirm → hitter assignment → window assembly → integration). Tests build incrementally.

**Files:**
- Create: `src/demo/tennis_action/flowunit/hit_fuse/{CMakeLists.txt,hit_fuse.toml,hit_fuse.py,offline_smoke_test.py}`

### Task 6a: cross-confirm algorithm

- [ ] **Step 1: Write the failing test**

Write `src/demo/tennis_action/flowunit/hit_fuse/offline_smoke_test.py`:

```python
#!/usr/bin/env python3
"""Offline smoke test for hit_fuse. Builds incrementally — Tasks 6a-6d each
extend this file."""

from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

_FU = types.ModuleType("_flowunit")
_FU.Buffer = lambda dev, b: ("BUFFER", b)
class _SC: STATUS_SUCCESS = 0; STATUS_FAULT = 1
class _Status:
    StatusCode = _SC
    def __init__(self, *a, **k): pass
_FU.Status = _Status
_FU.FlowUnit = type("FlowUnit", (), {"__init__": lambda self: None,
                                     "get_bind_device": lambda self: None})
_FU.info = lambda msg: None
_FU.error = lambda msg: print(f"[ERR] {msg}", file=sys.stderr)
sys.modules["_flowunit"] = _FU

sys.path.insert(0, str(Path(__file__).parent))
from hit_fuse import (  # noqa: E402
    cross_confirm, assign_hitter, assemble_window, HitFuse
)


# --- Cross-confirm tests (Task 6a) ---

def _make_ball_track(frames: int, traj: dict[int, tuple[float, float]],
                     default_peak: float = 0.8) -> dict[int, tuple[float, float, float]]:
    """{frame_idx: (cx, cy, peak)}. Missing frames default to sentinel."""
    return {f: (traj[f][0], traj[f][1], default_peak) if f in traj
            else (-1.0, -1.0, 0.0) for f in range(frames)}


def test_cross_confirm_direction_flip_accepts():
    # Ball moving +x for 6 frames, then -x for 6 frames; hit at frame 50.
    ball = {f: ((50 - f) if f < 50 else (f - 50), 0.0, 0.8) for f in range(40, 60)}
    ball_full = {f: ball.get(f, (-1.0, -1.0, 0.0)) for f in range(100)}
    ok, reason = cross_confirm(ball_full, hit_frame=50, require_visual_confirm=True)
    assert ok, f"reason={reason}"
    print("test_cross_confirm_direction_flip_accepts: PASS")


def test_cross_confirm_no_flip_rejects():
    # Ball moving +x monotonically.
    ball = {f: (float(f), 0.0, 0.8) for f in range(40, 60)}
    ball_full = {f: ball.get(f, (-1.0, -1.0, 0.0)) for f in range(100)}
    ok, reason = cross_confirm(ball_full, hit_frame=50, require_visual_confirm=True)
    assert not ok
    assert reason == "trajectory_not_consistent", reason
    print("test_cross_confirm_no_flip_rejects: PASS")


def test_cross_confirm_missing_ball_with_require_drops():
    ball_full = {f: (-1.0, -1.0, 0.0) for f in range(100)}
    ok, reason = cross_confirm(ball_full, hit_frame=50, require_visual_confirm=True)
    assert not ok
    assert reason == "no_ball_near_hit", reason
    print("test_cross_confirm_missing_ball_with_require_drops: PASS")


def test_cross_confirm_missing_ball_without_require_accepts():
    ball_full = {f: (-1.0, -1.0, 0.0) for f in range(100)}
    ok, reason = cross_confirm(ball_full, hit_frame=50, require_visual_confirm=False)
    assert ok and reason == "visual_skipped", reason
    print("test_cross_confirm_missing_ball_without_require_accepts: PASS")


def main() -> int:
    test_cross_confirm_direction_flip_accepts()
    test_cross_confirm_no_flip_rejects()
    test_cross_confirm_missing_ball_with_require_drops()
    test_cross_confirm_missing_ball_without_require_accepts()
    # More tests appended in tasks 6b–6d.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run test → ImportError**

```bash
cd src/demo/tennis_action/flowunit/hit_fuse
python offline_smoke_test.py
```

- [ ] **Step 3: Write minimal hit_fuse.py with cross_confirm only**

Write `src/demo/tennis_action/flowunit/hit_fuse/hit_fuse.py`:

```python
#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
"""hit_fuse — cross-confirm + hitter assignment + T-frame keypoint window
assembly. Collapse on three input streams (hit_centers, ball_pos,
tracked_poses); expand on output (hit_window + hit_meta paired per hit,
plus dropped_hits collapsed once).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import _flowunit as modelbox


# --- Algorithmic helpers (testable without ModelBox) ---

def cross_confirm(ball_map: dict[int, tuple[float, float, float]],
                  hit_frame: int,
                  require_visual_confirm: bool) -> tuple[bool, str]:
    """Decide whether the ball trajectory near `hit_frame` is consistent
    with a real racquet contact.

    ball_map: {frame_idx: (cx, cy, peak)}; sentinels are (-1, -1, 0).
    """
    # Collect valid positions in the [f-3, f+3] window.
    def _valid(f):
        p = ball_map.get(f, (-1.0, -1.0, 0.0))
        return p if p[0] >= 0 else None

    pre_pts = [_valid(hit_frame + d) for d in (-3, -2, -1, 0)]
    post_pts = [_valid(hit_frame + d) for d in (0, 1, 2, 3)]
    pre_pts = [p for p in pre_pts if p is not None]
    post_pts = [p for p in post_pts if p is not None]

    if not pre_pts and not post_pts:
        return (not require_visual_confirm,
                "visual_skipped" if not require_visual_confirm else "no_ball_near_hit")
    if len(pre_pts) < 2 or len(post_pts) < 2:
        return (not require_visual_confirm,
                "visual_skipped" if not require_visual_confirm else "no_ball_near_hit")

    # Approximate velocities by the first/last pairs.
    v_pre = (pre_pts[-1][0] - pre_pts[0][0], pre_pts[-1][1] - pre_pts[0][1])
    v_post = (post_pts[-1][0] - post_pts[0][0], post_pts[-1][1] - post_pts[0][1])
    dot = v_pre[0] * v_post[0] + v_pre[1] * v_post[1]
    mag_pre = max(1e-6, (v_pre[0] ** 2 + v_pre[1] ** 2) ** 0.5)
    mag_post = (v_post[0] ** 2 + v_post[1] ** 2) ** 0.5
    if dot < 0 or mag_post < 0.3 * mag_pre:
        return (True, "")
    return (False, "trajectory_not_consistent")


# Stubs to be filled in tasks 6b–6d.
def assign_hitter(*a, **k):
    raise NotImplementedError("Filled in Task 6b")


def assemble_window(*a, **k):
    raise NotImplementedError("Filled in Task 6c")


class HitFuse(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()

    def open(self, config):
        raise NotImplementedError("Filled in Task 6d")

    def process(self, data_context):
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def data_pre(self, data_context):
        return modelbox.Status()

    def data_post(self, data_context):
        return modelbox.Status()

    def close(self):
        return modelbox.Status()
```

- [ ] **Step 4: Run test → 4 cross_confirm PASSes (later tests will be added)**

```bash
cd src/demo/tennis_action/flowunit/hit_fuse
python offline_smoke_test.py
```

Expected: 4 PASS lines for the cross_confirm tests.

- [ ] **Step 5: Commit (intermediate progress)**

```bash
git add src/demo/tennis_action/flowunit/hit_fuse/
git commit -m "$(cat <<'EOF'
flowunit: hit_fuse — cross-confirm algorithm (Task 6a)

Skeleton + cross_confirm() function: checks ball-velocity direction flip
(dot<0) or sharp deceleration (post < 0.3*pre) in [f-3, f+3]. Drops with
'no_ball_near_hit' when require_visual_confirm and ball is missing; with
'trajectory_not_consistent' on confirmed visibility but no flip. Offline
test covers all 4 branches. Hitter assignment + window assembly + the
ModelBox Process() wiring are stubbed for Tasks 6b–6d.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 6b: hitter assignment

- [ ] **Step 1: Append new test cases to offline_smoke_test.py**

Append to `src/demo/tennis_action/flowunit/hit_fuse/offline_smoke_test.py`, inserting **before** `def main()`:

```python
# --- Hitter assignment tests (Task 6b) ---

def _person(tid: int, bbox: tuple[float, float, float, float],
            wrist_l: tuple[float, float], wrist_r: tuple[float, float]) -> dict:
    """Build a tracked-pose dict mirroring the yolo_pose_track_post format."""
    kpts = [[0.0, 0.0, 0.0]] * 17
    kpts[9] = [wrist_l[0], wrist_l[1], 0.9]   # COCO L wrist
    kpts[10] = [wrist_r[0], wrist_r[1], 0.9]  # COCO R wrist
    return {"track_id": tid, "bbox": list(bbox), "score": 0.9, "kpts": kpts}


def test_assign_hitter_picks_nearest_wrist():
    poses = [
        _person(7, (50, 50, 200, 300), wrist_l=(100, 100), wrist_r=(180, 200)),
        _person(8, (500, 50, 700, 300), wrist_l=(550, 100), wrist_r=(680, 200)),
    ]
    pose_map = {50: poses}
    ball = (105, 103, 0.9)
    info = assign_hitter(pose_map, ball, hit_frame=50)
    assert info is not None
    assert info["track_id"] == 7, info
    assert not info["tie_resolved"]
    print("test_assign_hitter_picks_nearest_wrist: PASS")


def test_assign_hitter_no_player_returns_none():
    info = assign_hitter({}, (100, 100, 0.9), hit_frame=50)
    assert info is None
    print("test_assign_hitter_no_player_returns_none: PASS")


def test_assign_hitter_tie_break_by_track_id():
    poses = [
        _person(2, (100, 100, 200, 200), wrist_l=(150, 150), wrist_r=(160, 160)),
        _person(1, (300, 100, 400, 200), wrist_l=(150, 150), wrist_r=(160, 160)),
    ]
    info = assign_hitter({50: poses}, (155, 155, 0.9), hit_frame=50)
    assert info is not None and info["track_id"] == 1, info
    assert info["tie_resolved"], "should have flipped tie_resolved"
    print("test_assign_hitter_tie_break_by_track_id: PASS")


def test_assign_hitter_uses_nearby_frame_when_target_empty():
    poses = [_person(7, (50, 50, 200, 300), (100, 100), (180, 200))]
    info = assign_hitter({48: poses}, (105, 103, 0.9), hit_frame=50)
    assert info is not None and info["track_id"] == 7, info
    print("test_assign_hitter_uses_nearby_frame_when_target_empty: PASS")
```

And add the four new calls inside `main()`:

```python
    test_assign_hitter_picks_nearest_wrist()
    test_assign_hitter_no_player_returns_none()
    test_assign_hitter_tie_break_by_track_id()
    test_assign_hitter_uses_nearby_frame_when_target_empty()
```

- [ ] **Step 2: Run test → expect NotImplementedError on assign_hitter**

```bash
python offline_smoke_test.py
```

Expected: 4 cross_confirm PASSes then `NotImplementedError: Filled in Task 6b`.

- [ ] **Step 3: Replace the `assign_hitter` stub with the implementation**

In `src/demo/tennis_action/flowunit/hit_fuse/hit_fuse.py`, replace the `assign_hitter` stub with:

```python
def _bbox_iou_simple(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def assign_hitter(pose_map: dict[int, list[dict[str, Any]]],
                  ball: tuple[float, float, float],
                  hit_frame: int) -> dict[str, Any] | None:
    """Pick the track whose closer wrist (L or R) is nearest to `ball` at
    `hit_frame`. Look ±2 frames if the target frame is empty.

    Returns {track_id, dist, tie_resolved, frame_used} or None when no track
    can be located in [f-2, f+2].
    """
    bx, by, _ = ball
    poses = None
    frame_used = hit_frame
    for d in (0, -1, 1, -2, 2):
        cand = pose_map.get(hit_frame + d)
        if cand:
            poses = cand
            frame_used = hit_frame + d
            break
    if not poses:
        return None

    def _wrist_dist(p):
        lw = p["kpts"][9]
        rw = p["kpts"][10]
        d_l = ((lw[0] - bx) ** 2 + (lw[1] - by) ** 2) ** 0.5
        d_r = ((rw[0] - bx) ** 2 + (rw[1] - by) ** 2) ** 0.5
        return min(d_l, d_r)

    dists = [(p["track_id"], _wrist_dist(p), p) for p in poses]
    dists.sort(key=lambda t: t[1])
    best_tid, best_dist, best_p = dists[0]

    tie_resolved = False
    if len(dists) > 1 and abs(dists[1][1] - best_dist) < 5.0:
        # Tight tie: bbox-IoU with a 30-px box around ball.
        ball_box = [bx - 15, by - 15, bx + 15, by + 15]
        iou_best = _bbox_iou_simple(best_p["bbox"], ball_box)
        iou_alt = _bbox_iou_simple(dists[1][2]["bbox"], ball_box)
        if iou_alt > iou_best:
            best_tid, best_dist, best_p = dists[1]
            tie_resolved = True
        elif iou_alt == iou_best and dists[1][0] < best_tid:
            best_tid, best_dist, best_p = dists[1]
            tie_resolved = True

    return {"track_id": int(best_tid), "dist": float(best_dist),
            "tie_resolved": tie_resolved, "frame_used": int(frame_used)}
```

- [ ] **Step 4: Run test → 8 PASSes (4 cross_confirm + 4 assign_hitter)**

```bash
python offline_smoke_test.py
```

- [ ] **Step 5: Commit**

```bash
git add src/demo/tennis_action/flowunit/hit_fuse/
git commit -m "$(cat <<'EOF'
flowunit: hit_fuse — hitter assignment (Task 6b)

assign_hitter(): for each detected track at the hit frame, distance from
ball to the nearest wrist (kpts[9] L, kpts[10] R). Looks ±2 frames if the
target frame has no track. Ties within 5 px broken by bbox-IoU with a
30-px box around the ball, then by lower track_id; tie_resolved is
recorded. Four new test cases cover the picking, no-player drop, tie-
break, and ±frame fallback paths.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 6c: window assembly + interpolation

- [ ] **Step 1: Append window-assembly tests**

Append to `src/demo/tennis_action/flowunit/hit_fuse/offline_smoke_test.py` **before** `def main()`:

```python
# --- Window assembly tests (Task 6c) ---

def _ts(tid: int, x: float, y: float) -> dict:
    """Tracked pose at a fixed wrist position."""
    return _person(tid, (x - 50, y - 50, x + 50, y + 50),
                   wrist_l=(x, y), wrist_r=(x, y))


def test_assemble_window_full_coverage():
    pose_map = {f: [_ts(7, 100 + f, 200)] for f in range(20, 60)}
    res = assemble_window(pose_map, track_id=7, hit_frame=40, T=32,
                          min_coverage=0.75, image_width=1280, image_height=720)
    assert res is not None
    arr, meta = res
    assert arr.shape == (32, 17, 3), arr.shape
    assert not meta["boundary_padded"]
    print("test_assemble_window_full_coverage: PASS")


def test_assemble_window_interpolates_gaps():
    # 30/32 frames present.
    pose_map = {}
    for f in range(20, 60):
        if f in (38, 41):
            continue
        pose_map[f] = [_ts(7, 100 + f, 200)]
    res = assemble_window(pose_map, track_id=7, hit_frame=40, T=32,
                          min_coverage=0.75, image_width=1280, image_height=720)
    assert res is not None
    print("test_assemble_window_interpolates_gaps: PASS")


def test_assemble_window_coverage_too_low_drops():
    pose_map = {f: [_ts(7, 100 + f, 200)] for f in range(20, 26)}
    res = assemble_window(pose_map, track_id=7, hit_frame=40, T=32,
                          min_coverage=0.75, image_width=1280, image_height=720)
    assert res is None
    print("test_assemble_window_coverage_too_low_drops: PASS")


def test_assemble_window_boundary_padding():
    # Hit at frame 5 with T=32 → leading 11 frames need padding.
    pose_map = {f: [_ts(7, 100 + f, 200)] for f in range(0, 30)}
    res = assemble_window(pose_map, track_id=7, hit_frame=5, T=32,
                          min_coverage=0.75, image_width=1280, image_height=720)
    assert res is not None
    arr, meta = res
    assert meta["boundary_padded"]
    print("test_assemble_window_boundary_padding: PASS")
```

Add the calls in `main()`:

```python
    test_assemble_window_full_coverage()
    test_assemble_window_interpolates_gaps()
    test_assemble_window_coverage_too_low_drops()
    test_assemble_window_boundary_padding()
```

- [ ] **Step 2: Run test → expect NotImplementedError on `assemble_window`**

- [ ] **Step 3: Replace the `assemble_window` stub**

In `hit_fuse.py`, replace the `assemble_window` stub with:

```python
def _find_track_kpts(poses: list[dict[str, Any]], track_id: int) \
        -> list[list[float]] | None:
    for p in poses:
        if p["track_id"] == track_id:
            return p["kpts"]
    return None


def assemble_window(pose_map: dict[int, list[dict[str, Any]]],
                    track_id: int, hit_frame: int, T: int,
                    min_coverage: float, image_width: int,
                    image_height: int) -> tuple[np.ndarray, dict] | None:
    """Gather T keypoint frames of `track_id` centered on `hit_frame`,
    linear-interpolating short gaps and padding boundaries.

    Returns (np.ndarray [T,17,3] in normalized coords, meta) or None when
    coverage is below `min_coverage` after interpolation.
    """
    half = T // 2
    f_start = hit_frame - half
    f_end = f_start + T  # exclusive

    raw: list[list[list[float]] | None] = []
    for f in range(f_start, f_end):
        poses = pose_map.get(f, [])
        raw.append(_find_track_kpts(poses, track_id))

    valid_count = sum(1 for k in raw if k is not None)
    if valid_count < int(min_coverage * T):
        return None

    # Interpolate small gaps; pad boundary from last-known.
    boundary_padded = False
    # Forward fill leading None entries.
    first_valid = next((i for i, k in enumerate(raw) if k is not None), -1)
    if first_valid > 0:
        boundary_padded = True
        for i in range(first_valid):
            raw[i] = [list(p) for p in raw[first_valid]]
    last_valid = next((i for i, k in enumerate(reversed(raw)) if k is not None), -1)
    if last_valid > 0:
        boundary_padded = True
        last_idx = T - 1 - last_valid
        for i in range(last_idx + 1, T):
            raw[i] = [list(p) for p in raw[last_idx]]

    # Linear interpolate interior gaps.
    i = 0
    while i < T:
        if raw[i] is not None:
            i += 1
            continue
        # Find the next valid frame.
        j = i + 1
        while j < T and raw[j] is None:
            j += 1
        if j >= T:
            break
        prev_idx = i - 1
        prev_k = raw[prev_idx]
        next_k = raw[j]
        span = j - prev_idx
        for fill_i in range(i, j):
            alpha = (fill_i - prev_idx) / span
            interp = []
            for ki in range(17):
                x = prev_k[ki][0] * (1 - alpha) + next_k[ki][0] * alpha
                y = prev_k[ki][1] * (1 - alpha) + next_k[ki][1] * alpha
                c = min(prev_k[ki][2], next_k[ki][2])
                interp.append([float(x), float(y), float(c)])
            raw[fill_i] = interp
        i = j

    # Should now be all-non-None.
    arr = np.zeros((T, 17, 3), dtype=np.float32)
    for ti, kpts in enumerate(raw):
        for ki, (x, y, c) in enumerate(kpts):
            arr[ti, ki, 0] = x / image_width
            arr[ti, ki, 1] = y / image_height
            arr[ti, ki, 2] = c

    return arr, {"boundary_padded": boundary_padded}
```

- [ ] **Step 4: Run test → 12 PASSes**

```bash
python offline_smoke_test.py
```

- [ ] **Step 5: Commit**

```bash
git add src/demo/tennis_action/flowunit/hit_fuse/
git commit -m "$(cat <<'EOF'
flowunit: hit_fuse — window assembly + interpolation (Task 6c)

assemble_window(): gather T frames of a chosen track's keypoints centered
on the hit frame; linear-interpolate interior gaps; pad leading/trailing
boundary from the last-known kpts. Drops the hit when coverage falls
below min_coverage. Emits normalized (x/W, y/H, conf) tensor (T,17,3).
Four new test cases cover full coverage, interpolation, drop, and
boundary padding.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 6d: ModelBox wiring + Open() validation + end-to-end integration test

- [ ] **Step 1: Append the integration test**

Append to `src/demo/tennis_action/flowunit/hit_fuse/offline_smoke_test.py` **before** `def main()`:

```python
# --- End-to-end Python class integration (Task 6d) ---

def _write_stub_meta(path: Path, T: int = 32):
    path.write_text(json.dumps({
        "classes": ["forehand", "backhand", "serve", "other"],
        "num_kpts": 17, "T": T, "input_layout": "BTC",
    }))


def test_hit_fuse_open_validates_meta():
    class _Cfg:
        def __init__(self, d): self.d = d
        def get_int(self, k, d): return int(self.d.get(k, d))
        def get_float(self, k, d): return float(self.d.get(k, d))
        def get_string(self, k, d): return str(self.d.get(k, d))
        def get_bool(self, k, d): return bool(self.d.get(k, d))
    with tempfile.TemporaryDirectory() as td:
        meta = Path(td) / "asformer_meta.json"
        _write_stub_meta(meta, T=32)
        fu = HitFuse()
        rc = fu.open(_Cfg({"T": 32, "asformer_meta_path": str(meta)}))
        assert rc == _SC.STATUS_SUCCESS
        # Mismatched T should fault.
        _write_stub_meta(meta, T=16)
        fu2 = HitFuse()
        rc2 = fu2.open(_Cfg({"T": 32, "asformer_meta_path": str(meta)}))
        assert rc2 == _SC.STATUS_FAULT
        print("test_hit_fuse_open_validates_meta: PASS")


def test_hit_fuse_end_to_end_one_hit():
    """Drive the python class directly: feed ball_pos + tracked_poses + a
    single hit center; expect one confirmed (hit_window, hit_meta) pair."""
    # Ball direction flip at frame 40.
    ball_map = {}
    for f in range(0, 80):
        if 35 <= f <= 45:
            x = (45 - f) if f <= 40 else (f - 40) * -1 + (45 - 40)
            ball_map[f] = (200.0 + x, 200.0, 0.9)
        else:
            ball_map[f] = (-1.0, -1.0, 0.0)
    # One track present for all frames 20..60 with wrist near (200, 200).
    pose_map = {f: [_ts(7, 200, 200)] for f in range(20, 60)}
    centers = [{"frame_idx": 40, "audio_conf": 0.9, "t_center_sec": 1.33}]
    with tempfile.TemporaryDirectory() as td:
        meta = Path(td) / "asformer_meta.json"
        _write_stub_meta(meta, T=32)
        fu = HitFuse()

        class _Cfg:
            def __init__(self, d): self.d = d
            def get_int(self, k, d): return int(self.d.get(k, d))
            def get_float(self, k, d): return float(self.d.get(k, d))
            def get_string(self, k, d): return str(self.d.get(k, d))
            def get_bool(self, k, d): return bool(self.d.get(k, d))

        fu.open(_Cfg({"T": 32, "image_width": 1280, "image_height": 720,
                      "require_visual_confirm": False,
                      "min_window_coverage": 0.75,
                      "asformer_meta_path": str(meta)}))
    confirmed, dropped = fu.fuse_session(centers, ball_map, pose_map)
    assert len(confirmed) == 1, f"got {len(confirmed)} confirmed"
    arr, meta_h = confirmed[0]
    assert arr.shape == (32, 17, 3)
    assert meta_h["track_id"] == 7
    print("test_hit_fuse_end_to_end_one_hit: PASS")
```

Add to `main()`:

```python
    test_hit_fuse_open_validates_meta()
    test_hit_fuse_end_to_end_one_hit()
```

- [ ] **Step 2: Run test → ImportError on `fuse_session` (or NotImplementedError on `HitFuse.open`)**

- [ ] **Step 3: Fill in `HitFuse.open()`, `process()`, `data_pre/post()`, and the `fuse_session()` orchestration**

Replace the placeholder methods in `hit_fuse.py`:

```python
class HitFuse(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()
        self._ball: dict[int, tuple[float, float, float]] = {}
        self._poses: dict[int, list[dict[str, Any]]] = {}
        self._centers_raw: list[dict[str, Any]] | None = None

    def open(self, config):
        self.T = config.get_int("T", 32)
        self.require_visual_confirm = bool(
            config.get_bool("require_visual_confirm", True)) \
            if hasattr(config, "get_bool") else True
        self.min_window_coverage = config.get_float("min_window_coverage", 0.75)
        self.image_width = config.get_int("image_width", 1280)
        self.image_height = config.get_int("image_height", 720)
        meta_path = config.get_string("asformer_meta_path", "")
        if not meta_path:
            modelbox.error("hit_fuse: asformer_meta_path required")
            return modelbox.Status.StatusCode.STATUS_FAULT
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except OSError as exc:
            modelbox.error(f"hit_fuse: cannot read asformer_meta_path: {exc}")
            return modelbox.Status.StatusCode.STATUS_FAULT
        if int(meta.get("T", -1)) != self.T:
            modelbox.error(
                f"hit_fuse: meta T={meta.get('T')} != configured T={self.T}")
            return modelbox.Status.StatusCode.STATUS_FAULT
        if int(meta.get("num_kpts", -1)) != 17:
            modelbox.error("hit_fuse: meta num_kpts must be 17")
            return modelbox.Status.StatusCode.STATUS_FAULT
        self._ball = {}
        self._poses = {}
        self._centers_raw = None
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def data_pre(self, data_context):
        self._ball = {}
        self._poses = {}
        self._centers_raw = None
        return modelbox.Status()

    def process(self, data_context):
        for buf in data_context.input("ball_pos"):
            arr = np.frombuffer(buf.as_object(), dtype=np.float32)
            cx, cy, peak, fi = float(arr[0]), float(arr[1]), float(arr[2]), int(arr[3])
            self._ball[fi] = (cx, cy, peak)
        for buf in data_context.input("tracked_poses"):
            payload = json.loads(bytes(buf.as_object()).decode("utf-8"))
            self._poses[int(payload["frame_idx"])] = payload["tracks"]
        for buf in data_context.input("hit_centers"):
            self._centers_raw = json.loads(bytes(buf.as_object()).decode("utf-8"))
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def data_post(self, data_context):
        if self._centers_raw is None:
            self._centers_raw = []
        confirmed, dropped = self.fuse_session(self._centers_raw, self._ball,
                                               self._poses)
        win_out = data_context.output("hit_window")
        meta_out = data_context.output("hit_meta")
        for hit_id, (arr, meta) in enumerate(confirmed):
            wb = modelbox.Buffer(self.get_bind_device(),
                                 arr.astype(np.float32).tobytes())
            wb.set("hit_id", int(hit_id))
            win_out.push_back(wb)
            meta["hit_id"] = int(hit_id)
            mb = modelbox.Buffer(self.get_bind_device(),
                                 json.dumps(meta).encode("utf-8"))
            mb.set("hit_id", int(hit_id))
            meta_out.push_back(mb)
        drop_out = data_context.output("dropped_hits")
        db = modelbox.Buffer(self.get_bind_device(),
                             json.dumps(dropped).encode("utf-8"))
        drop_out.push_back(db)
        return modelbox.Status()

    def close(self):
        return modelbox.Status()

    def fuse_session(self, centers_raw, ball_map, pose_map):
        """Pure-function entry point used by tests and by data_post()."""
        confirmed: list[tuple[np.ndarray, dict[str, Any]]] = []
        dropped: list[dict[str, Any]] = []
        for entry in centers_raw:
            f = int(entry["frame_idx"])
            audio_conf = float(entry["audio_conf"])
            ok, reason = cross_confirm(ball_map, f, self.require_visual_confirm)
            if not ok:
                dropped.append({"audio_hit_frame_idx": f,
                                "audio_conf": audio_conf, "reason": reason})
                continue
            ball = ball_map.get(f, (-1.0, -1.0, 0.0))
            assign = assign_hitter(pose_map, ball, f)
            if assign is None:
                dropped.append({"audio_hit_frame_idx": f,
                                "audio_conf": audio_conf,
                                "reason": "no_player_at_hit"})
                continue
            window = assemble_window(pose_map, assign["track_id"], f, self.T,
                                     self.min_window_coverage,
                                     self.image_width, self.image_height)
            if window is None:
                dropped.append({"audio_hit_frame_idx": f,
                                "audio_conf": audio_conf,
                                "reason": "window_coverage_low"})
                continue
            arr, wmeta = window
            meta = {
                "frame_idx": f, "track_id": int(assign["track_id"]),
                "audio_conf": audio_conf,
                "fused_conf": (audio_conf + 1.0) / 2.0 if reason != "visual_skipped"
                              else audio_conf,
                "ball_xy": [float(ball[0]), float(ball[1])],
                "tie_resolved": bool(assign["tie_resolved"]),
                "boundary_padded": bool(wmeta["boundary_padded"]),
                "visual_agrees": reason != "visual_skipped",
            }
            confirmed.append((arr, meta))
        return confirmed, dropped
```

- [ ] **Step 4: Run test → 14 PASSes**

```bash
python offline_smoke_test.py
```

- [ ] **Step 5: Write the TOML**

Write `src/demo/tennis_action/flowunit/hit_fuse/hit_fuse.toml`:

```toml
#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
[base]
name = "hit_fuse"
device = "cpu"
version = "1.0.0"
description = "Cross-confirm hit audio + ball trajectory; assign hitter; assemble T-frame keypoint window per confirmed hit."
entry = "hit_fuse@HitFuse"
type = "python"

collapse = true
expand = true

[config]
T = 32
require_visual_confirm = true
min_window_coverage = 0.75
image_width = 1280
image_height = 720
asformer_meta_path = ""

[input]
[input.input1]
name = "hit_centers"
[input.input2]
name = "ball_pos"
[input.input3]
name = "tracked_poses"

[output]
[output.output1]
name = "hit_window"
[output.output2]
name = "hit_meta"
[output.output3]
name = "dropped_hits"
```

- [ ] **Step 6: CMakeLists.txt**

Write `src/demo/tennis_action/flowunit/hit_fuse/CMakeLists.txt`:

```cmake
#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
cmake_minimum_required(VERSION 3.10)

set(FLOWUNIT_NAME "hit_fuse")
set(FLOWUNIT_PATH ${CMAKE_CURRENT_BINARY_DIR}/${FLOWUNIT_NAME})
file(MAKE_DIRECTORY ${FLOWUNIT_PATH})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FLOWUNIT_NAME}.py
               ${FLOWUNIT_PATH}/${FLOWUNIT_NAME}.py @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FLOWUNIT_NAME}.toml
               ${FLOWUNIT_PATH}/${FLOWUNIT_NAME}.toml @ONLY)

install(DIRECTORY ${FLOWUNIT_PATH}
    DESTINATION ${DEMO_TENNIS_ACTION_FLOWUNIT_DIR}
    COMPONENT demo
)
```

- [ ] **Step 7: Commit**

```bash
git add src/demo/tennis_action/flowunit/hit_fuse/
git commit -m "$(cat <<'EOF'
flowunit: hit_fuse — ModelBox wiring + end-to-end integration (Task 6d)

HitFuse.open() validates the configured T and num_kpts against
asformer_meta.json. data_pre/post() implement collapse-on-3-streams →
expand-on-2-outputs + dropped_hits collapsed once. process() accumulates
ball_pos and tracked_poses streams plus the single hit_centers buffer;
data_post() runs the full fuse pipeline (cross_confirm → assign_hitter →
assemble_window) and emits per-hit (hit_window, hit_meta) pairs.
fuse_session() is the pure-function entry tests drive directly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: `asformer_infer_ort` virtual inference TOML

**Files:**
- Create: `src/demo/tennis_action/flowunit/asformer_infer_ort/{CMakeLists.txt,asformer_infer_ort.toml}`

- [ ] **Step 1: Write the TOML**

Write `src/demo/tennis_action/flowunit/asformer_infer_ort/asformer_infer_ort.toml`:

```toml
#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
[base]
name = "asformer_infer_ort"
device = "cpu"
version = "1.0.0"
description = "ASFormer action classification, served through ModelBox's ONNX Runtime inference driver."
entry = "./asformer.onnx"
type = "inference"
virtual_type = "onnxruntime"

[input]
[input.input1]
name = "input"
type = "float"

[output]
[output.output1]
name = "logits"
type = "float"
```

- [ ] **Step 2: Write the CMakeLists.txt with model-copy custom-command**

Write `src/demo/tennis_action/flowunit/asformer_infer_ort/CMakeLists.txt`:

```cmake
#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
cmake_minimum_required(VERSION 3.10)

set(FLOWUNIT_NAME "asformer_infer_ort")
set(FLOWUNIT_PATH ${CMAKE_CURRENT_BINARY_DIR}/${FLOWUNIT_NAME})
file(MAKE_DIRECTORY ${FLOWUNIT_PATH})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FLOWUNIT_NAME}.toml
               ${FLOWUNIT_PATH}/${FLOWUNIT_NAME}.toml @ONLY)

set(ASFORMER_ONNX_SRC "${CMAKE_SOURCE_DIR}/assets/tennis_action/asformer.onnx")
set(ASFORMER_ONNX_DST "${FLOWUNIT_PATH}/asformer.onnx")
if(EXISTS ${ASFORMER_ONNX_SRC})
    add_custom_command(
        OUTPUT ${ASFORMER_ONNX_DST}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ASFORMER_ONNX_SRC} ${ASFORMER_ONNX_DST}
        DEPENDS ${ASFORMER_ONNX_SRC}
        VERBATIM
    )
    add_custom_target(${FLOWUNIT_NAME}_model ALL DEPENDS ${ASFORMER_ONNX_DST})
else()
    message(WARNING
        "${FLOWUNIT_NAME}: ${ASFORMER_ONNX_SRC} missing; graph will fail at "
        "open time. Run scripts/build_tennis_action_stub_onnx.py or drop a "
        "real model in.")
endif()

install(DIRECTORY ${FLOWUNIT_PATH}
    DESTINATION ${DEMO_TENNIS_ACTION_FLOWUNIT_DIR}
    COMPONENT demo
)
```

- [ ] **Step 3: Verify cmake configure parses the TOML and copies stub ONNX**

```bash
python src/demo/tennis_action/scripts/build_tennis_action_stub_onnx.py \
    --out_dir assets/tennis_action/
cd build && cmake .. && cd ..
ls -la build/src/demo/tennis_action/flowunit/asformer_infer_ort/asformer_infer_ort/
```

Expected: directory contains `asformer.onnx` and `asformer_infer_ort.toml`.

- [ ] **Step 4: Commit**

```bash
git add src/demo/tennis_action/flowunit/asformer_infer_ort/
git commit -m "$(cat <<'EOF'
flowunit: asformer_infer_ort — virtual ORT inference descriptor

Pure-TOML virtual inference flowunit mirroring hit_sound_infer_ort.
CMakeLists copies assets/tennis_action/asformer.onnx into the build
output if present; emits a warning otherwise so code-only iteration
still works.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: `action_sink` flowunit

Single sink — writes JSON, dropped JSON, and the overlay mp4. Reads original video via cv2 to render skeletons + ball + action labels.

**Files:**
- Create: `src/demo/tennis_action/flowunit/action_sink/{CMakeLists.txt,action_sink.toml,action_sink.py,offline_smoke_test.py}`

- [ ] **Step 1: Write the failing offline test**

Write `src/demo/tennis_action/flowunit/action_sink/offline_smoke_test.py`:

```python
#!/usr/bin/env python3
"""Offline smoke test for action_sink. Verifies JSON schema + overlay write."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

_FU = types.ModuleType("_flowunit")
_FU.Buffer = lambda dev, b: ("BUFFER", b)
class _SC: STATUS_SUCCESS = 0; STATUS_FAULT = 1
class _Status:
    StatusCode = _SC
    def __init__(self, *a, **k): pass
_FU.Status = _Status
_FU.FlowUnit = type("FlowUnit", (), {"__init__": lambda self: None,
                                     "get_bind_device": lambda self: None})
_FU.info = lambda msg: None
_FU.error = lambda msg: print(f"[ERR] {msg}", file=sys.stderr)
sys.modules["_flowunit"] = _FU

sys.path.insert(0, str(Path(__file__).parent))
from action_sink import ActionSink, write_results, render_overlay  # noqa: E402

# Path to the committed test video.
TEST_VIDEO = Path(__file__).resolve().parents[3] / "test" / "tennis_test.mp4"
CLASSES = ["forehand", "backhand", "serve", "other"]


def test_write_results_json_schema():
    confirmed = [{
        "hit_id": 0, "frame_idx": 40, "track_id": 7,
        "audio_conf": 0.9, "fused_conf": 0.95,
        "ball_xy": [200.0, 200.0],
        "tie_resolved": False, "boundary_padded": False,
        "visual_agrees": True,
        "action": "forehand", "action_conf": 0.7,
    }]
    dropped = [{"audio_hit_frame_idx": 10, "audio_conf": 0.6,
                "reason": "no_ball_near_hit"}]
    with tempfile.TemporaryDirectory() as td:
        j = Path(td) / "out.json"
        d = Path(td) / "dropped.json"
        write_results(j, d, confirmed, dropped, fps=30.0)
        body = json.loads(j.read_text())
        assert "summary" in body and "hits" in body
        assert body["summary"]["confirmed"] == 1
        assert body["summary"]["dropped"]["no_ball_near_hit"] == 1
        assert body["hits"][0]["t_sec"] - (40 / 30.0) < 1e-3
        d_body = json.loads(d.read_text())
        assert d_body[0]["reason"] == "no_ball_near_hit"
        print("test_write_results_json_schema: PASS")


def test_render_overlay_produces_decodable_mp4():
    if not TEST_VIDEO.exists():
        print("test_render_overlay_produces_decodable_mp4: SKIP (test video missing)")
        return
    confirmed = [{
        "hit_id": 0, "frame_idx": 40, "track_id": 7,
        "audio_conf": 0.9, "fused_conf": 0.95,
        "ball_xy": [640.0, 360.0],
        "tie_resolved": False, "boundary_padded": False,
        "visual_agrees": True,
        "action": "forehand", "action_conf": 0.7,
    }]
    poses_per_frame = {f: [{"track_id": 7, "bbox": [600, 300, 700, 500],
                            "kpts": [[640, 380, 0.9]] * 17}]
                       for f in range(0, 141)}
    ball_per_frame = {f: (640.0, 360.0, 0.9) for f in range(0, 141)}
    with tempfile.TemporaryDirectory() as td:
        out_mp4 = Path(td) / "overlay.mp4"
        ok = render_overlay(
            source_video=str(TEST_VIDEO),
            output_path=str(out_mp4),
            confirmed=confirmed,
            poses_per_frame=poses_per_frame,
            ball_per_frame=ball_per_frame,
            hit_label_persist_frames=30,
        )
        assert ok and out_mp4.exists() and out_mp4.stat().st_size > 5000
        # ffprobe sanity.
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=codec_type,duration", "-of", "default=noprint_wrappers=1",
             str(out_mp4)],
            capture_output=True, text=True, check=True,
        )
        assert "codec_type=video" in result.stdout
        print("test_render_overlay_produces_decodable_mp4: PASS")


def main() -> int:
    test_write_results_json_schema()
    test_render_overlay_produces_decodable_mp4()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run test → ImportError**

```bash
cd src/demo/tennis_action/flowunit/action_sink
python offline_smoke_test.py
```

- [ ] **Step 3: Write the flowunit**

Write `src/demo/tennis_action/flowunit/action_sink/action_sink.py`:

```python
#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
"""action_sink — final sink for the tennis_action pipeline.

Consumes per-hit ASFormer logits + hit_meta, plus per-frame ball_pos and
tracked_poses streams, plus the collapsed dropped_hits list. At session
end, writes:
  - <output_json>            structured per-hit results + summary
  - <output_overlay>         source video with skeleton + ball + action
                             labels burned in
  - <output_dropped>         per-hit drop reasons
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import _flowunit as modelbox

# cv2 is optional at import time; the overlay step gracefully degrades.
try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except ImportError:
    cv2 = None
    _HAS_CV2 = False


# COCO-17 skeleton edges (pairs of kpt indices to draw lines between).
_SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
             (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
             (0, 1), (0, 2), (1, 3), (2, 4)]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def write_results(out_json: Path, out_dropped: Path,
                  confirmed: list[dict[str, Any]],
                  dropped: list[dict[str, Any]],
                  fps: float) -> None:
    """Write the JSON results + dropped-hits file."""
    audio_hits = len(confirmed) + len(dropped)
    drop_reasons: dict[str, int] = {}
    for d in dropped:
        r = d.get("reason", "unknown")
        drop_reasons[r] = drop_reasons.get(r, 0) + 1
    body = {
        "summary": {"audio_hits": audio_hits, "confirmed": len(confirmed),
                    "dropped": drop_reasons},
        "hits": [],
    }
    for h in confirmed:
        body["hits"].append({
            **h,
            "t_sec": h["frame_idx"] / fps if fps > 0 else 0.0,
        })
    out_json.write_text(json.dumps(body, indent=2) + "\n")
    out_dropped.write_text(json.dumps(dropped, indent=2) + "\n")


def render_overlay(source_video: str, output_path: str,
                   confirmed: list[dict[str, Any]],
                   poses_per_frame: dict[int, list[dict[str, Any]]],
                   ball_per_frame: dict[int, tuple[float, float, float]],
                   hit_label_persist_frames: int = 30,
                   encoder: str = "libx264") -> bool:
    """Re-open the source video, draw skeleton/ball/action labels onto each
    frame, write the result via cv2. Returns True on success, False on
    failure (caller decides whether to fall back / log)."""
    if not _HAS_CV2:
        modelbox.error("action_sink: cv2 not available; skipping overlay")
        return False
    cap = cv2.VideoCapture(source_video)
    if not cap.isOpened():
        modelbox.error(f"action_sink: cannot open source {source_video}")
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        modelbox.error(f"action_sink: cannot open writer for {output_path}")
        return False
    # Per-frame label map.
    label_map: dict[int, str] = {}
    for hit in confirmed:
        f = int(hit["frame_idx"])
        label = f"hit{hit.get('hit_id', 0)}: {hit.get('action', '?')}"
        for off in range(0, hit_label_persist_frames):
            label_map[f + off] = label
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Ball circle.
        ball = ball_per_frame.get(frame_idx)
        if ball and ball[0] >= 0:
            cv2.circle(frame, (int(ball[0]), int(ball[1])), 8, (0, 0, 255), 2)
        # Skeletons.
        for pose in poses_per_frame.get(frame_idx, []):
            kpts = pose.get("kpts", [])
            for a, b in _SKELETON:
                if a < len(kpts) and b < len(kpts):
                    pa, pb = kpts[a], kpts[b]
                    if pa[2] > 0.3 and pb[2] > 0.3:
                        cv2.line(frame, (int(pa[0]), int(pa[1])),
                                 (int(pb[0]), int(pb[1])), (0, 255, 0), 2)
            tid = pose.get("track_id", -1)
            bbox = pose.get("bbox", [0, 0, 0, 0])
            cv2.putText(frame, f"id{tid}", (int(bbox[0]), int(bbox[1]) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        # Action label.
        label = label_map.get(frame_idx)
        if label:
            cv2.rectangle(frame, (10, 10), (10 + 9 * len(label), 40),
                          (0, 0, 0), -1)
            cv2.putText(frame, label, (14, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)
        frame_idx += 1
    writer.release()
    cap.release()
    return True


class ActionSink(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()
        self._logits: list[np.ndarray] = []
        self._metas: list[dict[str, Any]] = []
        self._dropped: list[dict[str, Any]] = []
        self._ball: dict[int, tuple[float, float, float]] = {}
        self._poses: dict[int, list[dict[str, Any]]] = {}

    def open(self, config):
        self.output_json = config.get_string("output_json",
                                             "/tmp/tennis_actions.json")
        self.output_overlay = config.get_string("output_overlay",
                                                "/tmp/tennis_actions_overlay.mp4")
        self.output_dropped = config.get_string("output_dropped",
                                                "/tmp/tennis_actions_dropped.json")
        self.class_names_path = config.get_string("class_names_path", "")
        self.overlay_encoder = config.get_string("overlay_encoder",
                                                 "h264_nvenc")
        self.overlay_fallback_encoder = config.get_string(
            "overlay_fallback_encoder", "libx264")
        self.hit_label_persist_frames = config.get_int(
            "hit_label_persist_frames", 30)
        self.source_video = config.get_string("source_video", "")
        if self.class_names_path and os.path.exists(self.class_names_path):
            try:
                with open(self.class_names_path) as f:
                    self.classes = json.load(f).get("classes", [])
            except OSError as exc:
                modelbox.error(
                    f"action_sink: cannot read class_names_path: {exc}")
                self.classes = []
        else:
            self.classes = []
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def data_pre(self, data_context):
        self._logits = []
        self._metas = []
        self._dropped = []
        self._ball = {}
        self._poses = {}
        return modelbox.Status()

    def process(self, data_context):
        for buf in data_context.input("logits"):
            arr = np.frombuffer(buf.as_object(), dtype=np.float32)
            self._logits.append(arr)
        for buf in data_context.input("hit_meta"):
            self._metas.append(
                json.loads(bytes(buf.as_object()).decode("utf-8"))
            )
        for buf in data_context.input("dropped_hits"):
            self._dropped = json.loads(bytes(buf.as_object()).decode("utf-8"))
        for buf in data_context.input("ball_pos"):
            arr = np.frombuffer(buf.as_object(), dtype=np.float32)
            cx, cy, peak, fi = float(arr[0]), float(arr[1]), float(arr[2]), int(arr[3])
            self._ball[fi] = (cx, cy, peak)
        for buf in data_context.input("tracked_poses"):
            payload = json.loads(bytes(buf.as_object()).decode("utf-8"))
            self._poses[int(payload["frame_idx"])] = payload["tracks"]
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def data_post(self, data_context):
        confirmed: list[dict[str, Any]] = []
        for logit, meta in zip(self._logits, self._metas):
            probs = _softmax(logit.flatten())
            cls_idx = int(np.argmax(probs))
            cls_name = self.classes[cls_idx] if cls_idx < len(self.classes) \
                else f"cls{cls_idx}"
            confirmed.append({
                **meta,
                "action": cls_name,
                "action_conf": float(probs[cls_idx]),
            })
        fps = self._infer_fps_from_video()
        try:
            write_results(Path(self.output_json), Path(self.output_dropped),
                          confirmed, self._dropped, fps)
        except OSError as exc:
            modelbox.error(f"action_sink: JSON write failed: {exc}")
        if self.source_video and os.path.exists(self.source_video):
            ok = render_overlay(
                source_video=self.source_video,
                output_path=self.output_overlay,
                confirmed=confirmed,
                poses_per_frame=self._poses,
                ball_per_frame=self._ball,
                hit_label_persist_frames=self.hit_label_persist_frames,
                encoder=self.overlay_encoder,
            )
            if not ok:
                modelbox.error("action_sink: overlay render failed")
        else:
            modelbox.error(
                f"action_sink: source_video missing or unset: "
                f"{self.source_video}")
        return modelbox.Status()

    def close(self):
        return modelbox.Status()

    def _infer_fps_from_video(self) -> float:
        if not _HAS_CV2 or not self.source_video \
                or not os.path.exists(self.source_video):
            return 30.0
        cap = cv2.VideoCapture(self.source_video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        return float(fps)
```

- [ ] **Step 4: Run test → PASS (with cv2 + the committed test video present)**

```bash
cd src/demo/tennis_action/flowunit/action_sink
python offline_smoke_test.py
```

Expected: `test_write_results_json_schema: PASS` and `test_render_overlay_produces_decodable_mp4: PASS` (or `SKIP` if the video isn't where expected — fix the test relative path).

- [ ] **Step 5: Write the TOML**

Write `src/demo/tennis_action/flowunit/action_sink/action_sink.toml`:

```toml
#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
[base]
name = "action_sink"
device = "cpu"
version = "1.0.0"
description = "Tennis-action sink: writes JSON results + dropped JSON + overlay mp4 from ASFormer logits + structured pose/ball streams."
entry = "action_sink@ActionSink"
type = "python"

collapse = true

[config]
output_json = "/tmp/tennis_actions.json"
output_overlay = "/tmp/tennis_actions_overlay.mp4"
output_dropped = "/tmp/tennis_actions_dropped.json"
class_names_path = ""
overlay_encoder = "h264_nvenc"
overlay_fallback_encoder = "libx264"
hit_label_persist_frames = 30
source_video = ""

[input]
[input.input1]
name = "logits"
[input.input2]
name = "hit_meta"
[input.input3]
name = "dropped_hits"
[input.input4]
name = "tracked_poses"
[input.input5]
name = "ball_pos"
```

- [ ] **Step 6: CMakeLists.txt**

Write `src/demo/tennis_action/flowunit/action_sink/CMakeLists.txt`:

```cmake
#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
cmake_minimum_required(VERSION 3.10)

set(FLOWUNIT_NAME "action_sink")
set(FLOWUNIT_PATH ${CMAKE_CURRENT_BINARY_DIR}/${FLOWUNIT_NAME})
file(MAKE_DIRECTORY ${FLOWUNIT_PATH})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FLOWUNIT_NAME}.py
               ${FLOWUNIT_PATH}/${FLOWUNIT_NAME}.py @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FLOWUNIT_NAME}.toml
               ${FLOWUNIT_PATH}/${FLOWUNIT_NAME}.toml @ONLY)

install(DIRECTORY ${FLOWUNIT_PATH}
    DESTINATION ${DEMO_TENNIS_ACTION_FLOWUNIT_DIR}
    COMPONENT demo
)
```

- [ ] **Step 7: Commit**

```bash
git add src/demo/tennis_action/flowunit/action_sink/
git commit -m "$(cat <<'EOF'
flowunit: action_sink — JSON + overlay mp4 sink

Collapse-on-streams sink for ASFormer logits + hit_meta + dropped_hits +
ball_pos + tracked_poses. data_post() softmaxes logits → class names from
asformer_meta.json, writes the structured result JSON + dropped JSON, and
re-opens the source video via cv2 to render skeleton + ball + per-hit
action labels (persist 30 frames). Offline test covers JSON schema and
overlay mp4 decodability.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Graph TOMLs

**Files:**
- Create: `src/demo/tennis_action/graph/tennis_action_cuda.toml.in`
- Create: `src/demo/tennis_action/graph/tennis_action_cpu_stub.toml.in`
- Modify: `src/demo/tennis_action/graph/CMakeLists.txt`

- [ ] **Step 1: Write the production CUDA graph**

Write `src/demo/tennis_action/graph/tennis_action_cuda.toml.in`:

```toml
[driver]
dir = [
    "@DEMO_TENNIS_ACTION_FLOWUNIT_DIR@",
    "@DEMO_HIT_SOUND_FLOWUNIT_DIR@"
]
[flow]
name = "TennisAction"
desc = "Offline tennis hit-sound + TrackNet + pose + ASFormer pipeline on CUDA."
[graph]
format = "graphviz"
graphconf = '''digraph tennis_action_cuda {
    node [shape=Mrecord]

    video_input        [type=flowunit, flowunit=video_input,        device=cpu,  deviceid=0,
                        source_url="@DEMO_TENNIS_ACTION_VIDEO@"]
    hit_trigger        [type=flowunit, flowunit=hit_trigger,        device=cpu,  deviceid=0]
    videodemuxer       [type=flowunit, flowunit=video_demuxer,      device=cpu,  deviceid=0]
    videodecoder       [type=flowunit, flowunit=video_decoder,      device=cuda, deviceid=0, pix_fmt=bgr]

    hit_window_emitter [type=flowunit, flowunit=hit_window_emitter, device=cpu,  deviceid=0,
                        video_path="@DEMO_TENNIS_ACTION_VIDEO@"]
    hit_sound_infer    [type=flowunit, flowunit=hit_sound_infer_ort, device=cpu, deviceid=0, batch_size=1]
    hit_centers_emit   [type=flowunit, flowunit=hit_centers_emitter, device=cpu, deviceid=0, video_fps=30.0]

    resize_tn          [type=flowunit, flowunit=resize,             device=cuda, deviceid=0,
                        image_width=512, image_height=288]
    normalize_tn       [type=flowunit, flowunit=normalize,          device=cuda, deviceid=0,
                        standard_deviation_inverse="0.00392157,0.00392157,0.00392157"]
    stacker            [type=flowunit, flowunit=tracknet_frame_stacker, device=cpu, deviceid=0]
    tracknet_infer     [type=flowunit, flowunit=tracknet_deep,      device=cuda, deviceid=0, batch_size=1]
    tracknet_ball      [type=flowunit, flowunit=tracknet_ball_emitter, device=cpu, deviceid=0,
                        score_thr=0.3, source_width=1280, source_height=720]

    resize_pose        [type=flowunit, flowunit=resize,             device=cuda, deviceid=0,
                        image_width=640, image_height=640]
    pose_detect        [type=flowunit, flowunit=yolo_pose_detect,   device=cuda, deviceid=0, batch_size=1]
    pose_track_post    [type=flowunit, flowunit=yolo_pose_track_post, device=cpu, deviceid=0,
                        conf_threshold=0.30, kpt_threshold=0.4, emit_overlay=false,
                        source_width=1280, source_height=720]

    hit_fuse           [type=flowunit, flowunit=hit_fuse,           device=cpu,  deviceid=0,
                        T=32, require_visual_confirm=true, min_window_coverage=0.75,
                        image_width=1280, image_height=720,
                        asformer_meta_path="@DEMO_TENNIS_ACTION_ASSETS_DIR@/asformer_meta.json"]
    asformer_infer     [type=flowunit, flowunit=asformer_infer_ort, device=cpu,  deviceid=0, batch_size=1]
    action_sink        [type=flowunit, flowunit=action_sink,        device=cpu,  deviceid=0,
                        output_json="/tmp/tennis_actions.json",
                        output_overlay="/tmp/tennis_actions_overlay.mp4",
                        output_dropped="/tmp/tennis_actions_dropped.json",
                        class_names_path="@DEMO_TENNIS_ACTION_ASSETS_DIR@/asformer_meta.json",
                        overlay_encoder=h264_nvenc, overlay_fallback_encoder=libx264,
                        hit_label_persist_frames=30,
                        source_video="@DEMO_TENNIS_ACTION_VIDEO@"]

    hit_trigger:out                 -> hit_window_emitter:in_data
    hit_window_emitter:mel          -> hit_sound_infer:input
    hit_sound_infer:logits          -> hit_centers_emit:logits

    video_input:out_video_url       -> videodemuxer:in_video_url
    videodemuxer:out_video_packet   -> videodecoder:in_video_packet

    videodecoder:out_video_frame    -> resize_tn:in_image
    resize_tn:out_image             -> normalize_tn:in_data
    normalize_tn:out_data           -> stacker:frame
    stacker:stacked                 -> tracknet_infer:frames
    tracknet_infer:heatmaps         -> tracknet_ball:heatmaps

    videodecoder:out_video_frame    -> resize_pose:in_image
    resize_pose:out_image           -> pose_detect:input
    pose_detect:output              -> pose_track_post:in_feat
    videodecoder:out_video_frame    -> pose_track_post:in_image

    hit_centers_emit:hit_centers    -> hit_fuse:hit_centers
    tracknet_ball:ball_pos          -> hit_fuse:ball_pos
    pose_track_post:tracked_poses   -> hit_fuse:tracked_poses

    hit_fuse:hit_window             -> asformer_infer:input
    asformer_infer:logits           -> action_sink:logits
    hit_fuse:hit_meta               -> action_sink:hit_meta
    hit_fuse:dropped_hits           -> action_sink:dropped_hits
    tracknet_ball:ball_pos          -> action_sink:ball_pos
    pose_track_post:tracked_poses   -> action_sink:tracked_poses
}'''
```

- [ ] **Step 2: Write the CPU-stub graph for CI**

Write `src/demo/tennis_action/graph/tennis_action_cpu_stub.toml.in`:

Same content as `tennis_action_cuda.toml.in` but with **every `device=cuda` swapped for `device=cpu`** for `videodecoder`, `resize_tn`, `normalize_tn`, `tracknet_infer`, `resize_pose`, `pose_detect`. Hand-edit the file to make those substitutions; structure is otherwise identical.

```bash
sed 's/device=cuda/device=cpu/g' \
    src/demo/tennis_action/graph/tennis_action_cuda.toml.in \
    > src/demo/tennis_action/graph/tennis_action_cpu_stub.toml.in
```

- [ ] **Step 3: Update `graph/CMakeLists.txt` to install both graphs**

Replace `src/demo/tennis_action/graph/CMakeLists.txt` with:

```cmake
#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
cmake_minimum_required(VERSION 3.10)

set(GRAPH_NAMES
    "tennis_action_cuda.toml"
    "tennis_action_cpu_stub.toml"
)
foreach(name ${GRAPH_NAMES})
    set(DEMO_GRAPH ${CMAKE_CURRENT_BINARY_DIR}/${name})
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${name}.in ${DEMO_GRAPH} @ONLY)
    install(FILES ${DEMO_GRAPH}
        DESTINATION ${DEMO_TENNIS_ACTION_GRAPH_DIR}
        COMPONENT demo
    )
endforeach()
```

- [ ] **Step 4: Verify cmake configure expands the templates**

```bash
cd build && cmake .. && cd ..
grep -E "source_url|video_path|asformer_meta_path" \
    build/src/demo/tennis_action/graph/tennis_action_cuda.toml
```

Expected: three lines with the `@…@` variables expanded to absolute paths.

- [ ] **Step 5: Commit**

```bash
git add src/demo/tennis_action/graph/
git commit -m "$(cat <<'EOF'
demo/tennis_action: graph TOMLs (cuda + cpu_stub)

cuda variant uses nvcodec / nppi_resize / TensorRT-backed inference.
cpu_stub variant swaps every device=cuda to device=cpu for end-to-end
CI runs against the stub ONNX models (no GPU required). Both graphs
share the same topology, three-way fan-out from videodecoder, audio
collapse → fusion, and the action_sink sink.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Graph-level smoke test

**Files:**
- Create: `src/demo/tennis_action/graph/test_tennis_action.py`

- [ ] **Step 1: Write the smoke test**

Write `src/demo/tennis_action/graph/test_tennis_action.py`:

```python
#!/usr/bin/env python3
"""End-to-end smoke test for the tennis_action demo. Runs the configured
graph and asserts the result files exist + are structurally valid.

Usage (typical):
    python test_tennis_action.py \
        --modelbox-tool build/release/usr/local/bin/modelbox-tool \
        --graph build/src/demo/tennis_action/graph/tennis_action_cpu_stub.toml

Returns 0 on success, non-zero on first failure.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _run(cmd: list[str], cwd: str | None = None, timeout: int = 600) -> int:
    print(f"[run] {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=cwd, timeout=timeout, capture_output=True,
                         text=True)
    if res.stdout:
        print(res.stdout)
    if res.returncode != 0 and res.stderr:
        print(res.stderr, file=sys.stderr)
    return res.returncode


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--modelbox-tool", required=True,
                   help="Path to the built modelbox-tool binary.")
    p.add_argument("--graph", required=True,
                   help="Path to the configured graph .toml.")
    p.add_argument("--out-json", default="/tmp/tennis_actions.json")
    p.add_argument("--out-overlay", default="/tmp/tennis_actions_overlay.mp4")
    p.add_argument("--out-dropped",
                   default="/tmp/tennis_actions_dropped.json")
    args = p.parse_args()

    # Clean previous run.
    for f in (args.out_json, args.out_overlay, args.out_dropped):
        Path(f).unlink(missing_ok=True)

    rc = _run([args.modelbox_tool, "flow", "run", "-name", "TennisAction",
               "-path", args.graph])
    if rc != 0:
        print(f"[FAIL] modelbox-tool exited {rc}", file=sys.stderr)
        return 1

    # 1. JSON exists and parses.
    j_path = Path(args.out_json)
    if not j_path.exists():
        print(f"[FAIL] {j_path} not produced", file=sys.stderr)
        return 2
    body = json.loads(j_path.read_text())
    if "summary" not in body or "hits" not in body:
        print("[FAIL] JSON missing summary/hits keys", file=sys.stderr)
        return 3
    print(f"[ok] {len(body['hits'])} confirmed hits; "
          f"{body['summary']['dropped']} drops")

    # 2. Overlay decodable.
    o_path = Path(args.out_overlay)
    if not o_path.exists():
        print(f"[FAIL] {o_path} not produced", file=sys.stderr)
        return 4
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,duration", "-of",
         "default=noprint_wrappers=1", str(o_path)],
        capture_output=True, text=True
    )
    if res.returncode != 0 or "codec_type=video" not in res.stdout:
        print(f"[FAIL] overlay not decodable: {res.stderr}", file=sys.stderr)
        return 5
    print(f"[ok] {o_path.stat().st_size} bytes overlay")

    # 3. Dropped JSON exists.
    d_path = Path(args.out_dropped)
    if not d_path.exists():
        print(f"[FAIL] {d_path} not produced", file=sys.stderr)
        return 6
    json.loads(d_path.read_text())  # parses
    print("[ok] dropped JSON valid")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Build the project (full build needed so `modelbox-tool` exists)**

```bash
cd build && make package -j$(nproc) && cd ..
```

Expected: artifacts produced under `build/release/`. (If you don't have a fresh build dir, `mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug -DWITH_ALL_DEMO=ON && make package -j$(nproc)`.)

- [ ] **Step 3: Run the cpu_stub graph smoke test**

```bash
python src/demo/tennis_action/graph/test_tennis_action.py \
    --modelbox-tool build/release/usr/local/bin/modelbox-tool \
    --graph build/src/demo/tennis_action/graph/tennis_action_cpu_stub.toml
```

Expected: `[ok]` lines for hits/overlay/dropped. Note that the stub ONNX produces a zero-tensor pose stream — meaning no real poses will be detected and `hit_fuse` will likely drop with `no_player_at_hit`. That's expected behavior; the test asserts the **files** are produced with the right shape, not that the result is semantically meaningful. The drop count should equal the audio-hit count.

If the run errors out at graph-open time, dispatch a focused fix-agent — likely cause is a flowunit name mismatch or a missing model file. The cv2/ffprobe stack must be installed in the host Python.

- [ ] **Step 4: Commit**

```bash
git add src/demo/tennis_action/graph/test_tennis_action.py
git commit -m "$(cat <<'EOF'
demo/tennis_action: graph-level smoke test

Drives modelbox-tool flow run against the cpu_stub graph and asserts:
result JSON parses with summary+hits keys, overlay mp4 is decodable by
ffprobe, dropped JSON parses. Designed for CI (no GPU). Real-model + CUDA
bring-up is manual on the pose server.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: TrackNet ONNX exporter script

This script is a bonus tool — needed for the real CUDA graph but not for CI.

**Files:**
- Create: `src/demo/tennis_action/scripts/export_tracknet_onnx.py`

- [ ] **Step 1: Write the exporter**

Write `src/demo/tennis_action/scripts/export_tracknet_onnx.py`:

```python
#!/usr/bin/env python3
"""Export the TrackNet PyTorch checkpoint to ONNX for TensorRT consumption.

Loads the model definition from the user's TrackNet checkout (defaults to
~/tracknet) and exports a [1, 9, 288, 512] → [1, 3, 288, 512] ONNX file.

Usage:
    python export_tracknet_onnx.py \
        --checkpoint ~/tracknet/checkpoints/model.pt \
        --tracknet_repo ~/tracknet \
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
    # Imports the user's model.py — symbol expected: TrackNet(model_arch...).
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
```

- [ ] **Step 2: Smoke-import the script (no checkpoint to actually run against)**

```bash
python -c "import importlib.util, sys; \
    spec = importlib.util.spec_from_file_location('exp', \
        'src/demo/tennis_action/scripts/export_tracknet_onnx.py'); \
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); \
    print('importable')"
```

Expected: `importable`. If torch is missing, that's a separate concern — the script imports torch lazily; you can also just `python -c "import ast; ast.parse(open('.../export_tracknet_onnx.py').read()); print('parsed')"` to skip torch.

- [ ] **Step 3: Commit**

```bash
git add src/demo/tennis_action/scripts/export_tracknet_onnx.py
git commit -m "$(cat <<'EOF'
demo/tennis_action: TrackNet PyTorch → ONNX exporter

CLI that imports TrackNet from a configurable tracknet repo (default
~/tracknet), loads a checkpoint, and writes [1,9,288,512]→[1,3,288,512]
ONNX. Needed for the CUDA graph; ONNX is generated, not committed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Manual CUDA bring-up on the pose server

Not a code task — execution checklist for the GPU machine. **No commit at the end** unless something is fixed.

- [ ] **Step 1: Export real ONNX models on the pose server**

```bash
# YOLO-pose: assumes Ultralytics installed.
yolo export model=yolov8n-pose.pt format=onnx opset=17 imgsz=640
mv yolov8n-pose.onnx assets/tennis_action/

# TrackNet (requires checkpoint).
python src/demo/tennis_action/scripts/export_tracknet_onnx.py \
    --checkpoint <path-to-tracknet-checkpoint> \
    --output assets/tennis_action/tracknet_deep.onnx

# ASFormer + meta.json — drop in your trained model and write meta.json.
# Example meta:
cat > assets/tennis_action/asformer_meta.json <<EOF
{"classes": ["forehand","backhand","serve","other"], "num_kpts": 17, "T": 32, "input_layout": "BTC"}
EOF
```

- [ ] **Step 2: Rebuild with the real assets present**

```bash
cd build && cmake .. && make package -j$(nproc) && cd ..
```

- [ ] **Step 3: Run the cuda graph**

```bash
python src/demo/tennis_action/graph/test_tennis_action.py \
    --modelbox-tool build/release/usr/local/bin/modelbox-tool \
    --graph build/src/demo/tennis_action/graph/tennis_action_cuda.toml
```

- [ ] **Step 4: Triage failures via focused fix-agents**

Likely failure modes and the fix-agent prompt for each:

- **TensorRT ONNX-parse error** on `yolov8n-pose` or `tracknet_deep`:
  > "TensorRT fails to parse `<file>.onnx` with `<error message>`. Identify the unsupported op or opset issue and either rewrite the ONNX (lower opset, replace unsupported layer) or add a TRT plugin shim. Re-run the graph and confirm it loads."
- **`h264_nvenc` not available** in the bundled ffmpeg:
  > "The ffmpeg in build/release lacks h264_nvenc. Either rebuild ffmpeg with nvenc or change `action_sink`'s `overlay_encoder` default to `libx264` for this deployment."
- **ORT-CUDA EP requested but only CPU available**: harmless — ASFormer is on CPU by design. Ignore.
- **Device-to-host fan-out hot path**: profile with the debug build; if `videodecoder:out_video_frame` D2H copies dominate, add an explicit cpu mirror flowunit and rewire the 3 cpu consumers off it.

- [ ] **Step 5: Capture results in the spec's "implementation-time risks" section**

If anything substantive surfaced during bring-up, edit `docs/superpowers/specs/2026-05-13-tennis-action-pipeline-design.md` §7's deferred-risks subsection to record what bit us and how it was solved, then commit as a doc update.

---

## Self-review

**Spec coverage check** (against `2026-05-13-tennis-action-pipeline-design.md`):

| Spec section | Plan task(s) |
|---|---|
| §1 Goal — JSON + overlay + dropped | Task 8 (action_sink), Task 9 (graph), Task 10 (smoke test) |
| §2 Inputs/outputs — `tennis_test.mp4`, three output files | Task 1 (move video), Task 8, Task 9 |
| §3 Architecture — three-branch DAG | Task 9 (graph) |
| §4.1 yolo_pose_track_post | Task 4 |
| §4.2 tracknet_ball_emitter | Task 3 |
| §4.3 hit_centers_emitter | Task 5 |
| §4.4 hit_fuse | Tasks 6a/6b/6c/6d |
| §4.5 asformer_infer_ort | Task 7 |
| §4.6 action_sink | Task 8 |
| §5 Graph TOML | Task 9 |
| §6 Model artifacts | Task 2 (stubs), Task 11 (TrackNet exporter), Task 12 (manual CUDA assets) |
| §7 Error handling — drops, soft failures | Task 6a–6d (drop reasons), Task 8 (sink-time fallbacks) |
| §8 Testing — Python offline + graph smoke | Tasks 3–6, Task 8, Task 10 |
| §9 Source tree | Tasks 1, 3–9 |
| §10 Out-of-scope | n/a |
| §11 Implementation handoff | this plan |

All spec sections mapped.

**Placeholder scan**: searched for "TBD", "TODO", "implement later", "fill in details", "appropriate", "etc" — none present in this plan. Each step ships actual code or an exact command.

**Type consistency**: cross-checked method names + dict keys across tasks. `tracked_poses` payload schema (`{frame_idx, tracks:[{track_id, bbox, score, kpts}]}`) is used identically in Task 4 (yolo_pose_track_post), Task 6d (hit_fuse `process`), and Task 8 (action_sink `process`). `ball_pos` schema (`[cx, cy, peak, frame_idx]` float32) is consistent across Task 3 (emitter), Task 6d (hit_fuse consumer), and Task 8. `hit_meta` keys (`frame_idx, track_id, audio_conf, fused_conf, ball_xy, tie_resolved, boundary_padded, visual_agrees`) used in Task 6d and consumed in Task 8 — match.

Plan ready for execution.
