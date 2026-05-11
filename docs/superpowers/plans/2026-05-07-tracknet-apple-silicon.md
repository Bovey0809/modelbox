# TrackNet on Apple Silicon Implementation Plan

> **Status (2026-05-07):** SUPERSEDED IN PARTS. Implementation landed in commits `d3381c6..e35d21c`. Two intentional deviations from this plan: (1) `tracknet_frame_stacker.toml` and `tracknet_post.toml` were dropped — C++ flowunits in this codebase register via `MODELBOX_DRIVER_FLOWUNIT` inside the `.dylib`/`.so` and the driver loader globs both extensions, so per-flowunit TOMLs are extraneous; only the virtual `tracknet_deep` Core ML wrapper ships a TOML. (2) Library naming follows the project's `modelbox-unit-cpu-<name>` convention (not `modelbox-flowunit-cpu-<name>`). Smoke test ends in a known EOF deadlock at end-of-stream — see the spec's "Known issues" section.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone ModelBox graph that runs TrackNetDeep ball-tracking on Apple Silicon via Core ML, drawing the predicted ball position on each video frame.

**Architecture:** Convert pretrained `.pt` → `.mlpackage` on the `pose` GPU server with a self-contained Python script. On the Mac side, add three flowunits (a stateful 3-frame stacker, a Core ML inference wrapper, a heatmap-post-process + drawer) plus a graph TOML wired in the same shape as the existing `apple_silicon_yolo_pose` demo.

**Tech Stack:** C++17 (flowunits), CMake, Core ML / coremltools, PyTorch (conversion only), OpenCV (drawing), ModelBox engine.

**Reference spec:** `docs/superpowers/specs/2026-05-07-tracknet-apple-silicon-design.md`

**Reference flowunits to mirror:**
- `src/demo/apple_silicon_yolo/flowunit/yolo_pose_detect/` — Core ML wrapper TOML + CMakeLists pattern.
- `src/drivers/devices/cpu/flowunit/yolo_track_post/yolo_track_post_flowunit.{h,cc}` — C++ flowunit shape with per-stream state and OpenCV drawing.
- `src/demo/apple_silicon_yolo/graph/apple_silicon_yolo_pose.toml.in` — graph TOML template.
- `scripts/macos/run-apple-silicon-yolo.sh` — runner script.

---

## File Structure

**Create:**
- `scripts/macos/convert_tracknet_coreml.py` — `.pt` → `.mlpackage`. Designed to run on `ssh pose`.
- `src/demo/apple_silicon_yolo/flowunit/tracknet_deep/CMakeLists.txt` — copies `.mlpackage` into the flowunit install dir.
- `src/demo/apple_silicon_yolo/flowunit/tracknet_deep/tracknet_deep.toml` — Core ML inference wrapper config (`virtual_type = "coreml"`).
- `src/demo/apple_silicon_yolo/flowunit/tracknet_frame_stacker/CMakeLists.txt`
- `src/demo/apple_silicon_yolo/flowunit/tracknet_frame_stacker/tracknet_frame_stacker.toml`
- `src/demo/apple_silicon_yolo/flowunit/tracknet_frame_stacker/tracknet_frame_stacker_flowunit.h`
- `src/demo/apple_silicon_yolo/flowunit/tracknet_frame_stacker/tracknet_frame_stacker_flowunit.cc`
- `src/demo/apple_silicon_yolo/flowunit/tracknet_frame_stacker/tracknet_frame_stacker_test.cc` — standalone gtest for the pure stacking helper.
- `src/demo/apple_silicon_yolo/flowunit/tracknet_post/CMakeLists.txt`
- `src/demo/apple_silicon_yolo/flowunit/tracknet_post/tracknet_post.toml`
- `src/demo/apple_silicon_yolo/flowunit/tracknet_post/tracknet_post_flowunit.h`
- `src/demo/apple_silicon_yolo/flowunit/tracknet_post/tracknet_post_flowunit.cc`
- `src/demo/apple_silicon_yolo/flowunit/tracknet_post/tracknet_post_test.cc` — standalone gtest for centroid extraction.
- `src/demo/apple_silicon_yolo/graph/apple_silicon_tracknet.toml.in`
- `scripts/macos/run-apple-silicon-tracknet.sh`

**Modify:**
- `src/demo/apple_silicon_yolo/flowunit/CMakeLists.txt` — `add_subdirectory(tracknet_*)` for the three new flowunits.
- `src/demo/apple_silicon_yolo/graph/CMakeLists.txt` — `configure_file` for the new graph TOML.

**Pretrained weight asset path on `pose`:** `/root/autodl-fs/repos/tracknet/exp_deep/TrackNet_best.pt`.

---

## Task 1: Conversion script (`.pt` → `.mlpackage`)

**Files:**
- Create: `scripts/macos/convert_tracknet_coreml.py`

**Run target:** `ssh pose` (Linux + CUDA optional). Imports: `torch`, `coremltools`, `numpy`. Clones the upstream `bcinno-dev/tracknet` repo into a tempdir and `sys.path.insert`s it to load `model.py` without vendoring.

- [ ] **Step 1: Write the script**

```python
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
    args = ap.parse_args()

    import coremltools as ct  # imported here so --help works without coremltools installed

    with tempfile.TemporaryDirectory() as td:
        repo = Path(td) / "tracknet"
        subprocess.check_call(["git", "clone", "--depth", "1", UPSTREAM, str(repo)])
        model_module = _import_model_module(repo)

        net = _build_model(model_module, args.variant, args.seq_len)
        state = torch.load(args.checkpoint, map_location="cpu")
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

        # Sanity check: PyTorch vs Core ML on 5 random tensors.
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

        out = Path(args.out).resolve()
        if out.exists():
            subprocess.check_call(["rm", "-rf", str(out)])
        mlmodel.save(str(out))

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
```

- [ ] **Step 2: Make it executable + commit**

```bash
chmod +x scripts/macos/convert_tracknet_coreml.py
git add scripts/macos/convert_tracknet_coreml.py
git commit -m "scripts/macos: TrackNet .pt -> .mlpackage conversion"
```

- [ ] **Step 3: Run on pose**

```bash
ssh pose 'cd /tmp && python3 -m pip install --user --quiet coremltools' || true
ssh pose 'python3 -c "import coremltools, torch; print(coremltools.__version__, torch.__version__)"'
scp scripts/macos/convert_tracknet_coreml.py pose:/tmp/convert_tracknet_coreml.py
ssh pose 'cd /tmp && python3 convert_tracknet_coreml.py --out /tmp/tracknet_deep.mlpackage'
```

Expected: prints `max |torch - coreml| = <small>` and `saved: /tmp/tracknet_deep.mlpackage  bytes=... sha256=...`.

- [ ] **Step 4: Pull the .mlpackage to the Mac**

```bash
mkdir -p assets/macos/tracknet
scp -r pose:/tmp/tracknet_deep.mlpackage assets/macos/tracknet/
ls assets/macos/tracknet/tracknet_deep.mlpackage/Manifest.json   # must exist
```

(Do NOT commit the `.mlpackage` — it's a build/runtime asset, not source. CMake in Task 4 references it from `assets/macos/tracknet/`.)

---

## Task 2: `tracknet_frame_stacker` flowunit

**Files:**
- Create: `src/demo/apple_silicon_yolo/flowunit/tracknet_frame_stacker/tracknet_frame_stacker_flowunit.h`
- Create: `src/demo/apple_silicon_yolo/flowunit/tracknet_frame_stacker/tracknet_frame_stacker_flowunit.cc`
- Create: `src/demo/apple_silicon_yolo/flowunit/tracknet_frame_stacker/tracknet_frame_stacker.toml`
- Create: `src/demo/apple_silicon_yolo/flowunit/tracknet_frame_stacker/CMakeLists.txt`
- Create: `src/demo/apple_silicon_yolo/flowunit/tracknet_frame_stacker/tracknet_frame_stacker_test.cc`

The flowunit is a sliding-window 3-frame stacker. State (`prev1`, `prev2`, `frame_idx`) is per-stream, held in a small struct stashed via `DataContext::SetPrivate`. The pure stacking logic lives in a free function so we can unit-test it without the ModelBox runtime.

- [ ] **Step 1: Write the failing test**

`tracknet_frame_stacker_test.cc`:

```cpp
#include <gtest/gtest.h>
#include <vector>

// Pure helper under test (declared in the .h, defined in the .cc).
extern "C" void TrackNetStackFrames(const float *cur,
                                    const float *prev1,
                                    const float *prev2,
                                    bool have_prev1,
                                    bool have_prev2,
                                    size_t per_frame,
                                    float *out);

namespace {

constexpr size_t H = 2;
constexpr size_t W = 2;
constexpr size_t C = 3;
constexpr size_t PER = C * H * W;  // 12

std::vector<float> Filled(float v) { return std::vector<float>(PER, v); }

}  // namespace

TEST(StackFrames, FirstFrameDuplicatesCurrent) {
  auto cur = Filled(7.f);
  std::vector<float> out(3 * PER, -1.f);
  TrackNetStackFrames(cur.data(), nullptr, nullptr, false, false, PER, out.data());
  for (size_t i = 0; i < 3 * PER; ++i) EXPECT_EQ(out[i], 7.f) << "i=" << i;
}

TEST(StackFrames, SecondFrameUsesPrev1Twice) {
  auto cur = Filled(2.f);
  auto prev1 = Filled(1.f);
  std::vector<float> out(3 * PER, -1.f);
  TrackNetStackFrames(cur.data(), prev1.data(), nullptr, true, false, PER, out.data());
  for (size_t i = 0; i < PER; ++i) EXPECT_EQ(out[i], 1.f);
  for (size_t i = PER; i < 2 * PER; ++i) EXPECT_EQ(out[i], 1.f);
  for (size_t i = 2 * PER; i < 3 * PER; ++i) EXPECT_EQ(out[i], 2.f);
}

TEST(StackFrames, ThirdAndAfterUsePrev2Prev1Cur) {
  auto cur = Filled(3.f);
  auto prev1 = Filled(2.f);
  auto prev2 = Filled(1.f);
  std::vector<float> out(3 * PER, -1.f);
  TrackNetStackFrames(cur.data(), prev1.data(), prev2.data(), true, true, PER, out.data());
  for (size_t i = 0; i < PER; ++i) EXPECT_EQ(out[i], 1.f);
  for (size_t i = PER; i < 2 * PER; ++i) EXPECT_EQ(out[i], 2.f);
  for (size_t i = 2 * PER; i < 3 * PER; ++i) EXPECT_EQ(out[i], 3.f);
}
```

- [ ] **Step 2: Implement the header**

`tracknet_frame_stacker_flowunit.h`:

```cpp
#ifndef MODELBOX_FLOWUNIT_TRACKNET_FRAME_STACKER_CPU_H_
#define MODELBOX_FLOWUNIT_TRACKNET_FRAME_STACKER_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/buffer.h>
#include <modelbox/flowunit.h>

#include <cstddef>
#include <vector>

constexpr const char *FLOWUNIT_NAME = "tracknet_frame_stacker";
constexpr const char *FLOWUNIT_TYPE = "cpu";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: Sliding 3-frame stacker for TrackNet.\n"
    "\t@Inputs:  frame  (float CHW 3x288x512)\n"
    "\t@Outputs: stacked (float CHW 9x288x512), concat[t-2,t-1,t]\n"
    "\t  First two frames pad by repeating the available frames.\n";

// Pure helper — exported with C linkage so tests link without C++ name mangling.
extern "C" void TrackNetStackFrames(const float *cur,
                                    const float *prev1,
                                    const float *prev2,
                                    bool have_prev1,
                                    bool have_prev2,
                                    size_t per_frame,
                                    float *out);

class TrackNetFrameStackerFlowUnit : public modelbox::FlowUnit {
 public:
  TrackNetFrameStackerFlowUnit() = default;
  ~TrackNetFrameStackerFlowUnit() override = default;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;
  modelbox::Status Close() override;
  modelbox::Status Process(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;
  modelbox::Status DataPre(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

 private:
  size_t per_frame_floats_{3 * 288 * 512};
};

#endif  // MODELBOX_FLOWUNIT_TRACKNET_FRAME_STACKER_CPU_H_
```

- [ ] **Step 3: Implement the .cc**

`tracknet_frame_stacker_flowunit.cc`:

```cpp
#include "tracknet_frame_stacker_flowunit.h"

#include <cstring>
#include <memory>

namespace {

struct StreamState {
  std::vector<float> prev1;
  std::vector<float> prev2;
  bool have_prev1 = false;
  bool have_prev2 = false;
};

constexpr const char *kStateKey = "tracknet_frame_stacker_state";

}  // namespace

extern "C" void TrackNetStackFrames(const float *cur,
                                    const float *prev1,
                                    const float *prev2,
                                    bool have_prev1,
                                    bool have_prev2,
                                    size_t per_frame,
                                    float *out) {
  const float *slot0;
  const float *slot1;
  if (have_prev2 && have_prev1) {
    slot0 = prev2;
    slot1 = prev1;
  } else if (have_prev1) {
    slot0 = prev1;
    slot1 = prev1;
  } else {
    slot0 = cur;
    slot1 = cur;
  }
  std::memcpy(out + 0 * per_frame, slot0, per_frame * sizeof(float));
  std::memcpy(out + 1 * per_frame, slot1, per_frame * sizeof(float));
  std::memcpy(out + 2 * per_frame, cur, per_frame * sizeof(float));
}

modelbox::Status TrackNetFrameStackerFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  per_frame_floats_ = static_cast<size_t>(opts->GetUint64("per_frame_floats", 3 * 288 * 512));
  return modelbox::STATUS_OK;
}

modelbox::Status TrackNetFrameStackerFlowUnit::Close() {
  return modelbox::STATUS_OK;
}

modelbox::Status TrackNetFrameStackerFlowUnit::DataPre(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto state = std::make_shared<StreamState>();
  state->prev1.resize(per_frame_floats_);
  state->prev2.resize(per_frame_floats_);
  data_ctx->SetPrivate(kStateKey, state);
  return modelbox::STATUS_OK;
}

modelbox::Status TrackNetFrameStackerFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto state =
      std::static_pointer_cast<StreamState>(data_ctx->GetPrivate(kStateKey));
  if (!state) {
    return {modelbox::STATUS_FAULT, "stream state missing"};
  }

  auto in_list = data_ctx->Input("frame");
  auto out_list = data_ctx->Output("stacked");
  if (!in_list || !out_list) {
    return {modelbox::STATUS_FAULT, "ports missing"};
  }

  const size_t per = per_frame_floats_;
  const size_t per_bytes = per * sizeof(float);

  for (size_t i = 0; i < in_list->Size(); ++i) {
    auto in = in_list->At(i);
    if (in->GetBytes() < per_bytes) {
      return {modelbox::STATUS_FAULT, "frame buffer too small"};
    }
    const auto *cur = static_cast<const float *>(in->ConstData());

    auto out = std::make_shared<modelbox::Buffer>(GetBindDevice());
    auto status = out->Build(3 * per_bytes);
    if (!status) return status;
    auto *dst = static_cast<float *>(out->MutableData());

    TrackNetStackFrames(
        cur,
        state->have_prev1 ? state->prev1.data() : nullptr,
        state->have_prev2 ? state->prev2.data() : nullptr,
        state->have_prev1, state->have_prev2,
        per, dst);

    // Carry source-resolution metadata through if upstream attached any.
    out->CopyMeta(in);

    out_list->PushBack(out);

    // Shift state.
    if (state->have_prev1) {
      std::memcpy(state->prev2.data(), state->prev1.data(), per_bytes);
      state->have_prev2 = true;
    }
    std::memcpy(state->prev1.data(), cur, per_bytes);
    state->have_prev1 = true;
  }

  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(TrackNetFrameStackerFlowUnit, FlowUnitDesc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Image");
  desc.AddFlowUnitInput({"frame"});
  desc.AddFlowUnitOutput({"stacked"});
  desc.SetFlowType(modelbox::STREAM);
  desc.SetDescription(FLOWUNIT_DESC);
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
```

> **Note for the engineer:** the exact `MODELBOX_FLOWUNIT` and `MODELBOX_DRIVER_FLOWUNIT` macro form may differ from the project's current macro names. Cross-check against `src/drivers/devices/cpu/flowunit/yolo_track_post/yolo_track_post_flowunit.cc` and copy whichever registration idiom that file uses. Same applies for the C++ flowunit in Task 3.

- [ ] **Step 4: Write the TOML**

`tracknet_frame_stacker.toml`:

```toml
[base]
name = "tracknet_frame_stacker"
device = "cpu"
version = "1.0.0"
description = "Sliding 3-frame stacker for TrackNet"
type = "flowunit"
entry = "libmodelbox-flowunit-cpu-tracknet-frame-stacker.so"

[input]
[input.input1]
name = "frame"
type = "float"

[output]
[output.output1]
name = "stacked"
type = "float"
```

- [ ] **Step 5: Write CMakeLists.txt**

Mirror `src/drivers/devices/cpu/flowunit/yolo_track_post/CMakeLists.txt` but trimmed for the demo tree. Read that file once, then write a parallel one that:
- Builds a shared library named `modelbox-flowunit-cpu-tracknet-frame-stacker` from `tracknet_frame_stacker_flowunit.cc`.
- Links against the same engine targets the yolo_track_post CMake links to.
- `configure_file` copies `tracknet_frame_stacker.toml` into the flowunit install dir.
- Builds `tracknet_frame_stacker_test` as a separate gtest executable from `tracknet_frame_stacker_test.cc` + `tracknet_frame_stacker_flowunit.cc` with `gtest_main`. Register it via `add_test`.

(Engineer: copy verbatim whatever pattern `yolo_track_post/CMakeLists.txt` uses for engine target deps; do not invent new names.)

- [ ] **Step 6: Run the unit test**

```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make tracknet_frame_stacker_test -j$(sysctl -n hw.ncpu)
./src/demo/apple_silicon_yolo/flowunit/tracknet_frame_stacker/tracknet_frame_stacker_test
```

Expected: 3 tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/demo/apple_silicon_yolo/flowunit/tracknet_frame_stacker/
git commit -m "flowunit: tracknet_frame_stacker (sliding 3-frame window)"
```

---

## Task 3: `tracknet_post` flowunit

**Files:**
- Create: `src/demo/apple_silicon_yolo/flowunit/tracknet_post/tracknet_post_flowunit.h`
- Create: `src/demo/apple_silicon_yolo/flowunit/tracknet_post/tracknet_post_flowunit.cc`
- Create: `src/demo/apple_silicon_yolo/flowunit/tracknet_post/tracknet_post.toml`
- Create: `src/demo/apple_silicon_yolo/flowunit/tracknet_post/CMakeLists.txt`
- Create: `src/demo/apple_silicon_yolo/flowunit/tracknet_post/tracknet_post_test.cc`

Pure logic exposed as `TrackNetExtractCentroid(...)` for testing; the flowunit is a thin wrapper around it + an OpenCV draw call.

- [ ] **Step 1: Write the failing test**

`tracknet_post_test.cc`:

```cpp
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

extern "C" bool TrackNetExtractCentroid(const float *heatmap,
                                        int h, int w,
                                        float score_thr,
                                        float mask_ratio,
                                        float *cx_out,
                                        float *cy_out,
                                        float *peak_out);

namespace {

std::vector<float> GaussianHeatmap(int h, int w, float cx, float cy, float sigma) {
  std::vector<float> hm(h * w, 0.f);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      float dx = x - cx;
      float dy = y - cy;
      hm[y * w + x] = std::exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
    }
  }
  return hm;
}

}  // namespace

TEST(ExtractCentroid, RecoversGaussianPeakWithinOnePixel) {
  const int H = 288, W = 512;
  auto hm = GaussianHeatmap(H, W, /*cx=*/123.4f, /*cy=*/77.8f, /*sigma=*/5.f);
  float cx = 0, cy = 0, peak = 0;
  ASSERT_TRUE(TrackNetExtractCentroid(hm.data(), H, W, 0.3f, 0.5f, &cx, &cy, &peak));
  EXPECT_NEAR(cx, 123.4f, 1.0f);
  EXPECT_NEAR(cy, 77.8f, 1.0f);
  EXPECT_GT(peak, 0.99f);
}

TEST(ExtractCentroid, FlatBelowThresholdReturnsFalse) {
  const int H = 288, W = 512;
  std::vector<float> hm(H * W, 0.1f);
  float cx = 0, cy = 0, peak = 0;
  EXPECT_FALSE(TrackNetExtractCentroid(hm.data(), H, W, 0.3f, 0.5f, &cx, &cy, &peak));
}
```

- [ ] **Step 2: Implement the header**

`tracknet_post_flowunit.h`:

```cpp
#ifndef MODELBOX_FLOWUNIT_TRACKNET_POST_CPU_H_
#define MODELBOX_FLOWUNIT_TRACKNET_POST_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/buffer.h>
#include <modelbox/flowunit.h>

#include <opencv2/opencv.hpp>

constexpr const char *FLOWUNIT_NAME = "tracknet_post";
constexpr const char *FLOWUNIT_TYPE = "cpu";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: TrackNet heatmap post-process: peak -> centroid -> draw circle.\n"
    "\t@Inputs:  heatmaps (float CHW 3x288x512), source_frame (uint8 BGR HxWx3)\n"
    "\t@Outputs: frame_out (uint8 BGR HxWx3) with red ball circle\n";

extern "C" bool TrackNetExtractCentroid(const float *heatmap,
                                        int h, int w,
                                        float score_thr,
                                        float mask_ratio,
                                        float *cx_out,
                                        float *cy_out,
                                        float *peak_out);

class TrackNetPostFlowUnit : public modelbox::FlowUnit {
 public:
  TrackNetPostFlowUnit() = default;
  ~TrackNetPostFlowUnit() override = default;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;
  modelbox::Status Close() override;
  modelbox::Status Process(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

 private:
  int net_h_{288};
  int net_w_{512};
  int heatmap_channel_{2};
  float score_thr_{0.3f};
  float mask_ratio_{0.5f};
  int circle_radius_{6};
  cv::Scalar circle_color_{0, 0, 255};  // BGR red
  bool draw_heatmap_{false};
};

#endif  // MODELBOX_FLOWUNIT_TRACKNET_POST_CPU_H_
```

- [ ] **Step 3: Implement the .cc**

`tracknet_post_flowunit.cc`:

```cpp
#include "tracknet_post_flowunit.h"

#include <algorithm>
#include <sstream>

extern "C" bool TrackNetExtractCentroid(const float *heatmap,
                                        int h, int w,
                                        float score_thr,
                                        float mask_ratio,
                                        float *cx_out,
                                        float *cy_out,
                                        float *peak_out) {
  float peak = 0.f;
  int n = h * w;
  for (int i = 0; i < n; ++i) {
    if (heatmap[i] > peak) peak = heatmap[i];
  }
  if (peak_out) *peak_out = peak;
  if (peak < score_thr) return false;

  float thr = peak * mask_ratio;
  double wx = 0, wy = 0, ws = 0;
  for (int y = 0; y < h; ++y) {
    const float *row = heatmap + y * w;
    for (int x = 0; x < w; ++x) {
      float v = row[x];
      if (v >= thr) {
        wx += static_cast<double>(x) * v;
        wy += static_cast<double>(y) * v;
        ws += v;
      }
    }
  }
  if (ws <= 0) return false;
  *cx_out = static_cast<float>(wx / ws);
  *cy_out = static_cast<float>(wy / ws);
  return true;
}

namespace {

cv::Scalar ParseBGR(const std::string &s, cv::Scalar def) {
  std::stringstream ss(s);
  int b = 0, g = 0, r = 0;
  char comma;
  if (!(ss >> b >> comma >> g >> comma >> r)) return def;
  return cv::Scalar(b, g, r);
}

}  // namespace

modelbox::Status TrackNetPostFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  net_h_ = opts->GetInt32("net_h", net_h_);
  net_w_ = opts->GetInt32("net_w", net_w_);
  heatmap_channel_ = opts->GetInt32("heatmap_channel", heatmap_channel_);
  score_thr_ = opts->GetFloat("score_thr", score_thr_);
  mask_ratio_ = opts->GetFloat("mask_ratio", mask_ratio_);
  circle_radius_ = opts->GetInt32("circle_radius", circle_radius_);
  draw_heatmap_ = opts->GetBool("draw_heatmap", draw_heatmap_);
  circle_color_ = ParseBGR(opts->GetString("circle_color", "0,0,255"), circle_color_);
  return modelbox::STATUS_OK;
}

modelbox::Status TrackNetPostFlowUnit::Close() { return modelbox::STATUS_OK; }

modelbox::Status TrackNetPostFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto hm_list = data_ctx->Input("heatmaps");
  auto src_list = data_ctx->Input("source_frame");
  auto out_list = data_ctx->Output("frame_out");
  if (!hm_list || !src_list || !out_list) {
    return {modelbox::STATUS_FAULT, "ports missing"};
  }
  if (hm_list->Size() != src_list->Size()) {
    return {modelbox::STATUS_FAULT, "heatmaps/source_frame size mismatch"};
  }

  const int H = net_h_;
  const int W = net_w_;
  const size_t plane = static_cast<size_t>(H) * static_cast<size_t>(W);

  for (size_t i = 0; i < hm_list->Size(); ++i) {
    auto hm = hm_list->At(i);
    auto src = src_list->At(i);

    int src_w = 0, src_h = 0;
    src->Get("width", src_w);
    src->Get("height", src_h);
    if (src_w <= 0 || src_h <= 0) {
      return {modelbox::STATUS_FAULT, "source_frame missing width/height meta"};
    }

    cv::Mat frame(src_h, src_w, CV_8UC3,
                  const_cast<void *>(src->ConstData()));
    cv::Mat draw = frame.clone();

    const auto *hmap = static_cast<const float *>(hm->ConstData());
    const float *channel = hmap + static_cast<size_t>(heatmap_channel_) * plane;

    float cx_lr = 0, cy_lr = 0, peak = 0;
    bool ok = TrackNetExtractCentroid(channel, H, W, score_thr_, mask_ratio_,
                                      &cx_lr, &cy_lr, &peak);
    if (ok) {
      float cx = cx_lr * static_cast<float>(src_w) / static_cast<float>(W);
      float cy = cy_lr * static_cast<float>(src_h) / static_cast<float>(H);
      cv::circle(draw, cv::Point(static_cast<int>(cx), static_cast<int>(cy)),
                 circle_radius_, circle_color_, cv::FILLED, cv::LINE_AA);
    }

    if (draw_heatmap_) {
      cv::Mat hm_lr(H, W, CV_32FC1, const_cast<float *>(channel));
      cv::Mat hm_resized;
      cv::resize(hm_lr, hm_resized, cv::Size(src_w, src_h));
      cv::Mat hm_u8;
      hm_resized.convertTo(hm_u8, CV_8UC1, 255.0);
      cv::Mat hm_color;
      cv::applyColorMap(hm_u8, hm_color, cv::COLORMAP_JET);
      cv::addWeighted(draw, 0.7, hm_color, 0.3, 0, draw);
    }

    auto out = std::make_shared<modelbox::Buffer>(GetBindDevice());
    auto status = out->Build(static_cast<size_t>(src_w) * src_h * 3);
    if (!status) return status;
    std::memcpy(out->MutableData(), draw.data,
                static_cast<size_t>(src_w) * src_h * 3);
    out->Set("width", src_w);
    out->Set("height", src_h);
    out->Set("pix_fmt", std::string("bgr"));
    out_list->PushBack(out);
  }
  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(TrackNetPostFlowUnit, FlowUnitDesc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Image");
  desc.AddFlowUnitInput({"heatmaps"});
  desc.AddFlowUnitInput({"source_frame"});
  desc.AddFlowUnitOutput({"frame_out"});
  desc.SetFlowType(modelbox::NORMAL);
  desc.SetDescription(FLOWUNIT_DESC);
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
```

(Engineer: cross-check the registration macro names against `yolo_track_post_flowunit.cc`.)

- [ ] **Step 4: Write the TOML**

`tracknet_post.toml`:

```toml
[base]
name = "tracknet_post"
device = "cpu"
version = "1.0.0"
description = "TrackNet heatmap post-process: peak -> centroid -> draw"
type = "flowunit"
entry = "libmodelbox-flowunit-cpu-tracknet-post.so"

[config]
net_h = 288
net_w = 512
heatmap_channel = 2
score_thr = 0.3
mask_ratio = 0.5
circle_radius = 6
circle_color = "0,0,255"
draw_heatmap = false

[input]
[input.input1]
name = "heatmaps"
type = "float"
[input.input2]
name = "source_frame"
type = "uint8"

[output]
[output.output1]
name = "frame_out"
type = "uint8"
```

- [ ] **Step 5: Write CMakeLists.txt**

Same recipe as Task 2 step 5. Output library name: `modelbox-flowunit-cpu-tracknet-post`. Add a `tracknet_post_test` gtest executable. Link OpenCV.

- [ ] **Step 6: Run the unit tests**

```bash
cd build && make tracknet_post_test -j$(sysctl -n hw.ncpu)
./src/demo/apple_silicon_yolo/flowunit/tracknet_post/tracknet_post_test
```

Expected: 2 tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/demo/apple_silicon_yolo/flowunit/tracknet_post/
git commit -m "flowunit: tracknet_post (heatmap centroid + circle draw)"
```

---

## Task 4: `tracknet_deep` Core ML inference wrapper flowunit

**Files:**
- Create: `src/demo/apple_silicon_yolo/flowunit/tracknet_deep/tracknet_deep.toml`
- Create: `src/demo/apple_silicon_yolo/flowunit/tracknet_deep/CMakeLists.txt`

This is a "virtual" Core ML flowunit — pure config + the `.mlpackage`. No C++ code; the `coreml_inference` flowunit on the `apple_silicon` device handles loading.

- [ ] **Step 1: Verify the .mlpackage exists locally**

```bash
test -f assets/macos/tracknet/tracknet_deep.mlpackage/Manifest.json && echo OK
```

If missing, run Task 1 step 4 first.

- [ ] **Step 2: Write the TOML**

`tracknet_deep.toml`:

```toml
[base]
name = "tracknet_deep"
device = "apple_silicon"
version = "1.0.0"
description = "TrackNetDeep ball tracker (Core ML, ANE+GPU+CPU)"
entry = "./tracknet_deep.mlpackage"
type = "inference"
virtual_type = "coreml"

[input]
[input.input1]
name = "frames"
type = "float"

[output]
[output.output1]
name = "heatmaps"
type = "float"
```

- [ ] **Step 3: Write CMakeLists.txt (mirrors yolo_pose_detect)**

```cmake
cmake_minimum_required(VERSION 3.10)

set(FLOWUNIT_NAME "tracknet_deep")
set(FLOWUNIT_PATH ${CMAKE_CURRENT_BINARY_DIR}/${FLOWUNIT_NAME})
set(FLOWUNIT_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/${FLOWUNIT_NAME}.toml)
set(MODEL_SRC ${CMAKE_SOURCE_DIR}/assets/macos/tracknet/tracknet_deep.mlpackage)
set(MODEL_DST ${FLOWUNIT_PATH}/tracknet_deep.mlpackage)
set(MODEL_STAMP ${MODEL_DST}/Manifest.json)

file(MAKE_DIRECTORY ${FLOWUNIT_PATH})
configure_file(${FLOWUNIT_CONFIG} ${FLOWUNIT_PATH}/${FLOWUNIT_NAME}.toml @ONLY)

add_custom_command(
    OUTPUT ${MODEL_STAMP}
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${MODEL_DST}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${MODEL_SRC} ${MODEL_DST}
    DEPENDS ${MODEL_SRC}/Manifest.json
    VERBATIM)
add_custom_target(${FLOWUNIT_NAME}_model ALL DEPENDS ${MODEL_STAMP})

install(DIRECTORY ${FLOWUNIT_PATH}
    DESTINATION ${DEMO_APPLE_SILICON_YOLO_FLOWUNIT_DIR}
    COMPONENT demo USE_SOURCE_PERMISSIONS)
```

- [ ] **Step 4: Wire all three new flowunits into the parent CMake**

Open `src/demo/apple_silicon_yolo/flowunit/CMakeLists.txt`. Add (matching position with the existing `add_subdirectory(yolo_*_detect)` lines):

```cmake
add_subdirectory(tracknet_frame_stacker)
add_subdirectory(tracknet_post)
add_subdirectory(tracknet_deep)
```

- [ ] **Step 5: Build everything**

```bash
cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug && make -j$(sysctl -n hw.ncpu)
```

Expected: builds clean; `build/src/demo/apple_silicon_yolo/flowunit/tracknet_deep/tracknet_deep/Manifest.json` exists.

- [ ] **Step 6: Commit**

```bash
git add src/demo/apple_silicon_yolo/flowunit/tracknet_deep/ \
        src/demo/apple_silicon_yolo/flowunit/CMakeLists.txt
git commit -m "flowunit: tracknet_deep CoreML wrapper + wire subdirs"
```

---

## Task 5: Graph TOML

**Files:**
- Create: `src/demo/apple_silicon_yolo/graph/apple_silicon_tracknet.toml.in`
- Modify: `src/demo/apple_silicon_yolo/graph/CMakeLists.txt`

- [ ] **Step 1: Write the graph**

`apple_silicon_tracknet.toml.in`:

```toml
[graph]
format = "graphviz"
graphconf = """digraph apple_silicon_tracknet {
    node [shape=Mrecord];
    queue_size = 16
    batch_size = 1

    input1[type=input, device=cpu]
    decoder[type=flowunit, flowunit=video_decoder, device=cpu, pix_fmt=bgr, queue_size=16]
    resize[type=flowunit, flowunit=resize, device=cpu, image_width=512, image_height=288]
    normalize[type=flowunit, flowunit=normalize, device=cpu, standard_deviation_inverse="0.00392157,0.00392157,0.00392157"]
    stacker[type=flowunit, flowunit=tracknet_frame_stacker, device=cpu]
    infer[type=flowunit, flowunit=tracknet_deep, device=apple_silicon, batch_size=1]
    post[type=flowunit, flowunit=tracknet_post, device=cpu, score_thr=0.3]
    encoder[type=flowunit, flowunit=video_encoder, device=cpu, format=mp4]
    output1[type=output, device=cpu]

    input1 -> decoder:in_video_url
    decoder:out_video_frame -> resize:in_image
    decoder:out_video_frame -> post:source_frame
    resize:out_image -> normalize:in_image
    normalize:out_image -> stacker:frame
    stacker:stacked -> infer:frames
    infer:heatmaps -> post:heatmaps
    post:frame_out -> encoder:in_video_frame
    encoder:out_video_url -> output1
}"""

[driver]
skip-default = false
dir = ["@DEMO_APPLE_SILICON_YOLO_FLOWUNIT_DIR@"]
```

(Engineer: cross-check the `[driver]` block against `apple_silicon_yolo_pose.toml.in` and copy that exact form. The `@...@` substitutions must match the configure_file substitutions used by the parent CMake.)

- [ ] **Step 2: Wire it into graph/CMakeLists.txt**

Add a `configure_file` line for the new `.toml.in` matching the existing yolo entries.

- [ ] **Step 3: Build, then inspect**

```bash
cd build && make -j$(sysctl -n hw.ncpu)
cat build/src/demo/apple_silicon_yolo/graph/apple_silicon_tracknet.toml | head -40
```

Expected: configured TOML exists with `@`-vars resolved.

- [ ] **Step 4: Commit**

```bash
git add src/demo/apple_silicon_yolo/graph/
git commit -m "graph: apple_silicon_tracknet TOML"
```

---

## Task 6: Runner script + end-to-end smoke

**Files:**
- Create: `scripts/macos/run-apple-silicon-tracknet.sh`

- [ ] **Step 1: Write the runner**

Read `scripts/macos/run-apple-silicon-yolo.sh` first, then write a sibling that:
- Accepts `INPUT_VIDEO` (positional 1) and `OUTPUT_VIDEO` (positional 2).
- Sets `DYLD_LIBRARY_PATH` and `MODELBOX_DRIVER_DIR` exactly the way the yolo runner does.
- Generates a configured TOML via `sed` substitution of `${input}` / `${output}` placeholders into the graph (or invokes `modelbox-tool` with `--input` / `--output` if that's what the yolo runner does — match its idiom).
- Defaults `INPUT_VIDEO` to a known tennis sample under `assets/macos/samples/tennis_short.mp4` (engineer: drop a 5–10 second tennis clip there before running, or pass an explicit path).

- [ ] **Step 2: Smoke-test end to end**

```bash
chmod +x scripts/macos/run-apple-silicon-tracknet.sh
./scripts/macos/run-apple-silicon-tracknet.sh /path/to/tennis_clip.mp4 /tmp/tracknet_out.mp4
```

Verification (per `superpowers:verification-before-completion`):

```bash
test -s /tmp/tracknet_out.mp4 && echo "MP4 exists, non-empty"
ffprobe -v error -count_frames -show_entries stream=nb_read_frames,width,height /tmp/tracknet_out.mp4
ffprobe -v error -count_frames -show_entries stream=nb_read_frames /path/to/tennis_clip.mp4
```

Expected: the two `nb_read_frames` values match. Open `/tmp/tracknet_out.mp4` visually and confirm a red dot follows the ball on at least some frames.

If frames don't match → check Open Question 1 (parallel-path ordering). If no dot ever appears → temporarily set `draw_heatmap=true` in `tracknet_post.toml` and rerun to see whether the heatmap is meaningful (model OK, threshold/centroid wrong) or noise (conversion or input-channel-order bug).

- [ ] **Step 3: Commit**

```bash
git add scripts/macos/run-apple-silicon-tracknet.sh
git commit -m "scripts/macos: apple_silicon TrackNet runner + smoke"
```

---

## Self-Review

**Spec coverage:**
- Variant choice (TrackNetDeep, plain RGB) → Task 1.
- Conversion script on `pose` → Task 1.
- `tracknet_frame_stacker` → Task 2.
- `tracknet_post` (centroid + draw) → Task 3.
- `tracknet_deep` Core ML wrapper → Task 4.
- Graph TOML → Task 5.
- Runner + smoke test → Task 6.
- Conversion sanity check + stacker unit test + post unit test + e2e smoke → Tasks 1, 2, 3, 6.

All spec sections covered.

**Open questions surfaced in Task 6 (parallel-path ordering, normalize-vs-bake-in-stacker)** are deliberately left as runtime contingencies — verify on first run, switch tactics if needed. Both are spec-acknowledged non-goals for the happy path.

**Type / name consistency check:**
- `TrackNetStackFrames` signature consistent across header and test (Task 2).
- `TrackNetExtractCentroid` signature consistent across header and test (Task 3).
- Flowunit names `tracknet_frame_stacker`, `tracknet_post`, `tracknet_deep` consistent across TOMLs, graph, CMakeLists.
- Library names `modelbox-flowunit-cpu-tracknet-{frame-stacker,post}` consistent between TOML `entry` and CMake target.
- Heatmap channel default = 2 (newest frame) consistent in spec, `tracknet_post.toml`, header default.

**Engineer notes embedded inline:**
- Macro names (`MODELBOX_FLOWUNIT`, `MODELBOX_DRIVER_FLOWUNIT`) — verify against `yolo_track_post_flowunit.cc` before building.
- CMake link recipes — copy from `yolo_track_post/CMakeLists.txt`.
- Runner script idiom — copy from `run-apple-silicon-yolo.sh`.
- `[driver]` block in graph TOML — copy from `apple_silicon_yolo_pose.toml.in`.

These four "verify against existing file" notes are deliberate: ModelBox's macro/cmake plumbing has evolved over the codebase, and rather than guess at obsolete names I'm telling the implementer to mirror the closest working sibling. That keeps the plan honest without leaving placeholders in the actual code-output steps.
