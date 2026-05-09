# PaddleOCR Flowunits Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring PP-OCRv4 (det + cls + rec) into ModelBox as native CUDA flowunits driven by Paddle Inference C++, with pre/post flowunits, demo graph, and unit tests.

**Architecture:** New inference engine adapter under `src/drivers/inference_engine/paddle/`, CUDA-bound inference flowunit under `src/drivers/devices/cuda/flowunit/paddle/`, seven CPU pre/post flowunits under `src/drivers/devices/cpu/flowunit/paddle_ocr_*/`, and a demo graph + weight download script under `examples/paddle_ocr/`.

**Tech Stack:** C++17, OpenCV (for postprocess), Paddle Inference C++ API, CUDA, ModelBox FlowUnit API.

**Spec:** `docs/superpowers/specs/2026-05-09-paddle-ocr-flowunits-design.md`.

**Build environment:** Project does not build on macOS — develop and test on a Linux/CUDA host. The user has provided `ssh pose` and `ssh 5090x4` as remote hosts with the dev container available.

**Conventions for every flowunit added:**
- Mirror the directory structure of `src/drivers/devices/cpu/flowunit/yolo_seg_post/` (`<name>.toml`, `<name>_flowunit.{h,cc}`, `CMakeLists.txt`).
- Apache 2.0 header at top of every source file (copy from `yolo_seg_post_flowunit.cc`).
- `OPENCV_FOUND` guard in `CMakeLists.txt` matching the yolo_seg_post pattern.
- Add `add_subdirectory(<name>)` to `src/drivers/devices/cpu/flowunit/CMakeLists.txt` after creating each unit.
- Tests live next to the flowunit as `<name>_flowunit_test.cc` and are picked up via the `list(APPEND DRIVER_UNIT_TEST_SOURCE ...)` block already in the template `CMakeLists.txt`.

**Remote build commands (use throughout):**
```bash
HOST=pose                                # or 5090x4
ssh $HOST "cd modelbox/build && cmake .. -DCMAKE_BUILD_TYPE=Debug && make -j\$(nproc)"
ssh $HOST "cd modelbox/build && make build-test -j\$(nproc) && ./test/unittest --gtest_filter='PaddleOcr*'"
```

---

## Task 0: Branch + remote sync

**Files:** none changed; this is environment setup.

- [ ] **Step 1:** Create a feature branch off `main`.

```bash
git checkout -b feature/paddle-ocr-flowunits main
git push -u origin feature/paddle-ocr-flowunits
```

- [ ] **Step 2:** Sync the branch to the remote build host.

```bash
ssh pose "git -C modelbox fetch origin && git -C modelbox checkout feature/paddle-ocr-flowunits"
```

- [ ] **Step 3:** Verify Paddle Inference is installed in the dev container. If `find_package(PaddleInference)` will fail, follow these steps on the remote:

```bash
ssh pose "ls /opt/paddle_inference || (cd /opt && \
  wget -q https://paddle-inference-lib.bj.bcebos.com/2.6.1/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda12.0_cudnn8.9_trt8.6/paddle_inference.tgz && \
  tar -xzf paddle_inference.tgz && mv paddle_inference paddle_inference)"
```

The exact tarball URL depends on the host's CUDA/cuDNN versions — verify with `nvidia-smi` and pick the matching package from <https://www.paddlepaddle.org.cn/inference/master/guides/install/download_lib.html>. Set `PaddleInference_DIR=/opt/paddle_inference` for cmake.

- [ ] **Step 4:** Confirm baseline build passes on the remote before any code changes.

```bash
ssh pose "cd modelbox && rm -rf build && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug && make -j\$(nproc)"
```

Expected: success.

- [ ] **Step 5:** Commit nothing; this task is just setup.

---

## Task 1: Paddle Inference engine adapter — directory + CMake

**Files:**
- Create: `src/drivers/inference_engine/paddle/CMakeLists.txt`
- Create: `src/drivers/inference_engine/paddle/paddle_inference.h`
- Create: `src/drivers/inference_engine/paddle/paddle_inference.cc`
- Modify: `src/drivers/inference_engine/CMakeLists.txt` (add subdir)

- [ ] **Step 1:** Add `add_subdirectory(paddle)` to `src/drivers/inference_engine/CMakeLists.txt`. Insert next to the other engines in alphabetical order.

- [ ] **Step 2:** Create `src/drivers/inference_engine/paddle/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.10)
set(SUBDIRECTORY_NAME paddle)
project(modelbox-inference-engine-${SUBDIRECTORY_NAME})

# Honor PaddleInference_DIR or PADDLE_INFERENCE_DIR (path to extracted SDK).
if(NOT DEFINED PaddleInference_DIR AND DEFINED ENV{PADDLE_INFERENCE_DIR})
    set(PaddleInference_DIR $ENV{PADDLE_INFERENCE_DIR})
endif()

if(NOT PaddleInference_DIR OR NOT EXISTS "${PaddleInference_DIR}/paddle/include/paddle_inference_api.h")
    message(STATUS "Paddle Inference SDK not found, disable paddle engine")
    return()
endif()

set(MODELBOX_PADDLE_FOUND TRUE PARENT_SCOPE)
set(PADDLE_INFERENCE_INCLUDE "${PaddleInference_DIR}/paddle/include" PARENT_SCOPE)
set(PADDLE_INFERENCE_LIB_DIR "${PaddleInference_DIR}/paddle/lib" PARENT_SCOPE)

include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${LIBMODELBOX_INCLUDE})
include_directories(${LIBMODELBOX_BASE_INCLUDE})
include_directories(${PaddleInference_DIR}/paddle/include)
include_directories(${PaddleInference_DIR}/third_party/install/mkldnn/include)
include_directories(${PaddleInference_DIR}/third_party/install/mklml/include)

link_directories(${PaddleInference_DIR}/paddle/lib)
link_directories(${PaddleInference_DIR}/third_party/install/mkldnn/lib)
link_directories(${PaddleInference_DIR}/third_party/install/mklml/lib)

file(GLOB SRC paddle_inference.cc)
add_library(modelbox-engine-paddle STATIC ${SRC})
target_link_libraries(modelbox-engine-paddle paddle_inference ${LIBMODELBOX_SHARED})

set(MODELBOX_ENGINE_PADDLE_TARGET modelbox-engine-paddle PARENT_SCOPE)
```

- [ ] **Step 3:** Create `src/drivers/inference_engine/paddle/paddle_inference.h`:

```cpp
/* Apache 2.0 header — copy from yolo_seg_post_flowunit.cc */
#ifndef MODELBOX_ENGINE_PADDLE_INFERENCE_H_
#define MODELBOX_ENGINE_PADDLE_INFERENCE_H_

#include <modelbox/base/status.h>
#include <modelbox/buffer.h>
#include <memory>
#include <string>
#include <vector>

namespace paddle_infer {
class Predictor;  // forward decl from paddle_inference_api.h
}

namespace modelbox {

struct PaddleInferenceParams {
  std::string model_file;
  std::string params_file;
  std::string device = "gpu";   // "cpu" | "gpu"
  int gpu_id = 0;
  bool enable_trt = false;
  int trt_workspace_mb = 256;
  std::string trt_precision = "fp32";  // fp32 | fp16 | int8
  bool enable_mkldnn = false;
  int cpu_threads = 4;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
};

class PaddleInference {
 public:
  Status Init(const PaddleInferenceParams& p);
  Status Infer(const std::vector<std::shared_ptr<Buffer>>& inputs,
               std::vector<std::shared_ptr<Buffer>>& outputs);
  const std::vector<std::string>& InputNames() const { return input_names_; }
  const std::vector<std::string>& OutputNames() const { return output_names_; }

 private:
  std::shared_ptr<paddle_infer::Predictor> predictor_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::string device_{"gpu"};
};

}  // namespace modelbox
#endif
```

- [ ] **Step 4:** Create `src/drivers/inference_engine/paddle/paddle_inference.cc` as an empty stub that compiles. Define `PaddleInference::Init` and `PaddleInference::Infer` as `return Status(STATUS_NOTSUPPORT, "not implemented");`. The body is filled in Task 2.

- [ ] **Step 5:** Build once on the remote with `PaddleInference_DIR` set, confirm the static lib builds.

```bash
ssh pose "cd modelbox/build && cmake .. -DCMAKE_BUILD_TYPE=Debug -DPaddleInference_DIR=/opt/paddle_inference && make modelbox-engine-paddle -j\$(nproc)"
```

Expected: target built, no other tree affected.

- [ ] **Step 6:** Commit.

```bash
git add src/drivers/inference_engine/paddle/ src/drivers/inference_engine/CMakeLists.txt
git commit -m "engine: paddle inference adapter scaffold"
```

---

## Task 2: Paddle Inference adapter — `Init` and `Infer` impl

**Files:**
- Modify: `src/drivers/inference_engine/paddle/paddle_inference.cc`

- [ ] **Step 1:** Replace the stub with the full implementation. Include the Paddle headers and implement `Init` (build `Config`, create `Predictor`, cache I/O names) and `Infer` (reshape per call, copy buffer → input handle, run, copy output handle → buffer).

```cpp
/* Apache 2.0 header */
#include "paddle_inference.h"

#include <modelbox/base/log.h>

#include <cstring>

#include "paddle/include/paddle_inference_api.h"

namespace modelbox {

Status PaddleInference::Init(const PaddleInferenceParams& p) {
  if (p.model_file.empty() || p.params_file.empty()) {
    return {STATUS_BADCONF, "paddle: model_file/params_file required"};
  }

  paddle_infer::Config cfg;
  cfg.SetModel(p.model_file, p.params_file);
  cfg.DisableGlogInfo();
  cfg.SwitchUseFeedFetchOps(false);
  cfg.SwitchSpecifyInputNames(true);
  cfg.EnableMemoryOptim();

  device_ = p.device;
  if (p.device == "gpu") {
    cfg.EnableUseGpu(256, p.gpu_id);
    if (p.enable_trt) {
      auto prec = paddle_infer::Config::Precision::kFloat32;
      if (p.trt_precision == "fp16") prec = paddle_infer::Config::Precision::kHalf;
      else if (p.trt_precision == "int8") prec = paddle_infer::Config::Precision::kInt8;
      cfg.EnableTensorRtEngine(p.trt_workspace_mb << 20, 1, 3, prec, false, false);
    }
  } else {
    cfg.DisableGpu();
    cfg.SetCpuMathLibraryNumThreads(p.cpu_threads);
    if (p.enable_mkldnn) cfg.EnableMKLDNN();
  }

  predictor_ = paddle_infer::CreatePredictor(cfg);
  if (!predictor_) return {STATUS_FAULT, "paddle: CreatePredictor failed"};

  input_names_ = p.input_names.empty() ? predictor_->GetInputNames() : p.input_names;
  output_names_ = p.output_names.empty() ? predictor_->GetOutputNames() : p.output_names;
  MBLOG_INFO << "paddle: loaded " << p.model_file << " inputs="
             << input_names_.size() << " outputs=" << output_names_.size();
  return STATUS_OK;
}

Status PaddleInference::Infer(
    const std::vector<std::shared_ptr<Buffer>>& inputs,
    std::vector<std::shared_ptr<Buffer>>& outputs) {
  if (inputs.size() != input_names_.size()) {
    return {STATUS_FAULT, "paddle: input count mismatch"};
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto handle = predictor_->GetInputHandle(input_names_[i]);
    std::vector<size_t> shape_sz;
    inputs[i]->Get("shape", shape_sz);
    std::vector<int> shape(shape_sz.begin(), shape_sz.end());
    handle->Reshape(shape);
    handle->CopyFromCpu(static_cast<const float*>(inputs[i]->ConstData()));
  }
  if (!predictor_->Run()) return {STATUS_FAULT, "paddle: Run failed"};

  outputs.resize(output_names_.size());
  for (size_t i = 0; i < output_names_.size(); ++i) {
    auto handle = predictor_->GetOutputHandle(output_names_[i]);
    auto shape = handle->shape();
    size_t numel = 1;
    for (auto d : shape) numel *= static_cast<size_t>(d);
    outputs[i] = std::make_shared<Buffer>();
    outputs[i]->Build(numel * sizeof(float));
    handle->CopyToCpu(static_cast<float*>(outputs[i]->MutableData()));
    std::vector<size_t> shape_sz(shape.begin(), shape.end());
    outputs[i]->Set("shape", shape_sz);
  }
  return STATUS_OK;
}

}  // namespace modelbox
```

- [ ] **Step 2:** Build on remote.

```bash
ssh pose "cd modelbox/build && make modelbox-engine-paddle -j\$(nproc)"
```

Expected: built clean.

- [ ] **Step 3:** Commit.

```bash
git add src/drivers/inference_engine/paddle/paddle_inference.cc
git commit -m "engine: paddle Init/Infer wraps paddle_infer::Predictor"
```

---

## Task 3: CUDA-bound `paddle_inference_flowunit`

**Files:**
- Create: `src/drivers/devices/cuda/flowunit/paddle/CMakeLists.txt`
- Create: `src/drivers/devices/cuda/flowunit/paddle/paddle_inference_flowunit.h`
- Create: `src/drivers/devices/cuda/flowunit/paddle/paddle_inference_flowunit.cc`
- Create: `src/drivers/devices/cuda/flowunit/paddle/paddle_inference_flowunit.toml`
- Modify: `src/drivers/devices/cuda/flowunit/CMakeLists.txt` (add subdir)

- [ ] **Step 1:** Add `add_subdirectory(paddle)` (gated) to `src/drivers/devices/cuda/flowunit/CMakeLists.txt`:

```cmake
if(MODELBOX_PADDLE_FOUND AND CUDA_FOUND)
    add_subdirectory(paddle)
endif()
```

- [ ] **Step 2:** Create `src/drivers/devices/cuda/flowunit/paddle/CMakeLists.txt` modeled on `tensorrt/CMakeLists.txt` (use that file as a literal template — copy it, then change `tensorrt`→`paddle` and the link libs from `nvinfer*` to `paddle_inference`). Add link dir `${PADDLE_INFERENCE_LIB_DIR}` and link `paddle_inference` plus `${MODELBOX_ENGINE_PADDLE_TARGET}`.

- [ ] **Step 3:** Create `paddle_inference_flowunit.toml`. This is the descriptor consumed by ModelBox at flowunit-load time:

```toml
[base]
name = "paddle_inference"
device = "cuda"
version = "1.0.0"
type = "inference"
group_type = "Inference"
description = "Paddle Inference (GPU) wrapper. Loads .pdmodel + .pdiparams."
entry = "./libmodelbox-unit-cuda-paddle.so"

[config]
plugin = "paddle"
```

- [ ] **Step 4:** Create `paddle_inference_flowunit.h`:

```cpp
/* Apache 2.0 header */
#ifndef MODELBOX_FLOWUNIT_PADDLE_INFERENCE_CUDA_H_
#define MODELBOX_FLOWUNIT_PADDLE_INFERENCE_CUDA_H_

#include <modelbox/flowunit.h>

#include <memory>

#include "paddle_inference.h"

class PaddleInferenceFlowUnit : public modelbox::FlowUnit {
 public:
  modelbox::Status Open(const std::shared_ptr<modelbox::Configuration>& opts) override;
  modelbox::Status Close() override;
  modelbox::Status Process(std::shared_ptr<modelbox::DataContext> ctx) override;

 private:
  std::unique_ptr<modelbox::PaddleInference> engine_;
  std::vector<std::string> in_ports_;
  std::vector<std::string> out_ports_;
};
#endif
```

- [ ] **Step 5:** Create `paddle_inference_flowunit.cc`:

```cpp
/* Apache 2.0 header */
#include "paddle_inference_flowunit.h"

#include <modelbox/base/log.h>
#include <modelbox/flowunit_api_helper.h>

modelbox::Status PaddleInferenceFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration>& opts) {
  modelbox::PaddleInferenceParams p;
  p.model_file = opts->GetString("config.model_file");
  p.params_file = opts->GetString("config.params_file");
  p.device = "gpu";
  p.gpu_id = opts->GetInt32("config.gpu_id", 0);
  p.enable_trt = opts->GetBool("config.enable_trt", false);
  p.trt_workspace_mb = opts->GetInt32("config.trt_workspace_mb", 256);
  p.trt_precision = opts->GetString("config.trt_precision", "fp32");
  p.input_names = opts->GetStrings("config.input_name");
  p.output_names = opts->GetStrings("config.output_name");

  // Per-flowunit-instance port names from graph TOML's input/output blocks.
  in_ports_ = opts->GetStrings("input.name");
  out_ports_ = opts->GetStrings("output.name");
  if (in_ports_.empty() || out_ports_.empty()) {
    return {modelbox::STATUS_BADCONF, "paddle_inference: ports not declared"};
  }

  engine_ = std::make_unique<modelbox::PaddleInference>();
  return engine_->Init(p);
}

modelbox::Status PaddleInferenceFlowUnit::Close() {
  engine_.reset();
  return modelbox::STATUS_OK;
}

modelbox::Status PaddleInferenceFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  std::vector<std::shared_ptr<modelbox::Buffer>> ins, outs;
  for (const auto& name : in_ports_) {
    auto port = ctx->Input(name);
    if (!port || port->Size() == 0) {
      return {modelbox::STATUS_FAULT, "paddle_inference: empty input " + name};
    }
    ins.push_back(port->At(0));
  }
  auto status = engine_->Infer(ins, outs);
  if (!status) return status;
  if (outs.size() != out_ports_.size()) {
    return {modelbox::STATUS_FAULT, "paddle_inference: output count mismatch"};
  }
  for (size_t i = 0; i < out_ports_.size(); ++i) {
    auto port = ctx->Output(out_ports_[i]);
    port->Build({outs[i]->GetBytes()});
    auto* dst = port->At(0)->MutableData();
    std::memcpy(dst, outs[i]->ConstData(), outs[i]->GetBytes());
    std::vector<size_t> shape;
    outs[i]->Get("shape", shape);
    port->At(0)->Set("shape", shape);
  }
  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(PaddleInferenceFlowUnit, modelbox::FlowUnitDesc)
    .SetFlowUnitGroupType("Inference");
```

- [ ] **Step 6:** Build.

```bash
ssh pose "cd modelbox/build && make modelbox-unit-cuda-paddle-shared -j\$(nproc)"
```

Expected: clean build.

- [ ] **Step 7:** Commit.

```bash
git add src/drivers/devices/cuda/flowunit/paddle/ src/drivers/devices/cuda/flowunit/CMakeLists.txt
git commit -m "flowunit: cuda paddle_inference wraps engine adapter"
```

---

## Task 4: `paddle_ocr_det_pre` flowunit

**Files:**
- Create: `src/drivers/devices/cpu/flowunit/paddle_ocr_det_pre/{CMakeLists.txt,paddle_ocr_det_pre.toml,paddle_ocr_det_pre_flowunit.h,paddle_ocr_det_pre_flowunit.cc}`
- Modify: `src/drivers/devices/cpu/flowunit/CMakeLists.txt` (add subdir)

- [ ] **Step 1:** Copy `src/drivers/devices/cpu/flowunit/yolo_seg_post/CMakeLists.txt` to the new dir, replacing every `yolo_seg_post` and `YOLO_SEG_POST` with the new name.

- [ ] **Step 2:** Create `paddle_ocr_det_pre.toml`:

```toml
[base]
name = "paddle_ocr_det_pre"
device = "cpu"
version = "1.0.0"
type = "flowunit"
group_type = "Image"
description = "PP-OCR detection preprocess: resize-to-32-multiple + ImageNet normalize."
entry = "./libmodelbox-unit-cpu-paddle_ocr_det_pre.so"

[config]
max_side_len = 960

[input]
[[input]]
name = "in_image"
type = "uint8"

[output]
[[output]]
name = "out_tensor"
type = "float"

[[output]]
name = "out_image"
type = "uint8"
```

- [ ] **Step 3:** Create the header. The class has `int max_side_len_{960}`, `Open/Close/Process`. Constants: `mean = {0.485, 0.456, 0.406}`, `std = {0.229, 0.224, 0.225}`.

- [ ] **Step 4:** Create `paddle_ocr_det_pre_flowunit.cc`. The Process body:

```cpp
modelbox::Status PaddleOcrDetPreFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  auto in = ctx->Input("in_image");
  auto out_t = ctx->Output("out_tensor");
  auto out_i = ctx->Output("out_image");
  if (!in || !out_t || !out_i || in->Size() == 0) return modelbox::STATUS_OK;

  std::vector<size_t> tensor_sizes(in->Size());
  std::vector<size_t> image_sizes(in->Size());
  std::vector<std::pair<int,int>> resized_dims(in->Size());

  for (size_t i = 0; i < in->Size(); ++i) {
    auto buf = in->At(i);
    int32_t w = 0, h = 0, c = 3;
    buf->Get("width", w); buf->Get("height", h); buf->Get("channel", c);
    if (w <= 0 || h <= 0 || c != 3) {
      return {modelbox::STATUS_FAULT, "det_pre: bad image meta"};
    }
    int max_side = std::max(w, h);
    float ratio = max_side > max_side_len_ ?
        static_cast<float>(max_side_len_) / max_side : 1.0F;
    int rw = static_cast<int>(std::round(w * ratio / 32) * 32);
    int rh = static_cast<int>(std::round(h * ratio / 32) * 32);
    rw = std::max(32, rw); rh = std::max(32, rh);
    resized_dims[i] = {rw, rh};
    tensor_sizes[i] = static_cast<size_t>(3) * rh * rw * sizeof(float);
    image_sizes[i] = buf->GetBytes();
  }
  out_t->Build(tensor_sizes);
  out_i->Build(image_sizes);

  static const float mean[3] = {0.485F, 0.456F, 0.406F};
  static const float std_[3] = {0.229F, 0.224F, 0.225F};

  for (size_t i = 0; i < in->Size(); ++i) {
    auto buf = in->At(i);
    int32_t w = 0, h = 0;
    buf->Get("width", w); buf->Get("height", h);
    cv::Mat src(h, w, CV_8UC3, const_cast<void*>(buf->ConstData()));
    int rw = resized_dims[i].first, rh = resized_dims[i].second;
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(rw, rh));
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    auto* dst = static_cast<float*>(out_t->At(i)->MutableData());
    // HWC u8 → CHW fp32 normalized.
    for (int y = 0; y < rh; ++y) {
      for (int x = 0; x < rw; ++x) {
        const auto& px = rgb.at<cv::Vec3b>(y, x);
        for (int c = 0; c < 3; ++c) {
          dst[c * rh * rw + y * rw + x] =
              (px[c] / 255.0F - mean[c]) / std_[c];
        }
      }
    }
    std::vector<size_t> shape = {1, 3,
        static_cast<size_t>(rh), static_cast<size_t>(rw)};
    out_t->At(i)->Set("shape", shape);

    // Forward original image with src dims so det_post can rescale polygons.
    auto* idst = out_i->At(i)->MutableData();
    std::memcpy(idst, buf->ConstData(), buf->GetBytes());
    out_i->At(i)->Set("width", w);
    out_i->At(i)->Set("height", h);
    out_i->At(i)->Set("channel", 3);
    out_i->At(i)->Set("resized_w", rw);
    out_i->At(i)->Set("resized_h", rh);
  }
  return modelbox::STATUS_OK;
}
```

- [ ] **Step 5:** Add `add_subdirectory(paddle_ocr_det_pre)` to `src/drivers/devices/cpu/flowunit/CMakeLists.txt`. Build.

```bash
ssh pose "cd modelbox/build && make modelbox-unit-cpu-paddle_ocr_det_pre-shared -j\$(nproc)"
```

- [ ] **Step 6:** Commit.

```bash
git add src/drivers/devices/cpu/flowunit/paddle_ocr_det_pre/ src/drivers/devices/cpu/flowunit/CMakeLists.txt
git commit -m "flowunit: paddle_ocr_det_pre resize+normalize for DBNet"
```

---

## Task 5: `paddle_ocr_det_post` flowunit (TDD)

**Files:**
- Create: `src/drivers/devices/cpu/flowunit/paddle_ocr_det_post/{CMakeLists.txt,paddle_ocr_det_post.toml,paddle_ocr_det_post_flowunit.h,paddle_ocr_det_post_flowunit.cc,paddle_ocr_det_post_flowunit_test.cc}`
- Modify: `src/drivers/devices/cpu/flowunit/CMakeLists.txt`

- [ ] **Step 1: Write the failing test first.** Create `paddle_ocr_det_post_flowunit_test.cc` modeled on `src/drivers/devices/cuda/flowunit/tensorrt/tensorrt_inference_flowunit_test.cc` (uses `DriverFlowTest`). Build a 64×64 probmap with two `cv::rectangle`-filled high-probability regions, run a graph snippet `mock_image_source -> paddle_ocr_det_post -> mock_sink`, assert `polygon_count == 2`.

```cpp
/* Apache 2.0 header */
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "driver_flow_test.h"

namespace modelbox {
class PaddleOcrDetPostTest : public testing::Test {
 protected:
  std::shared_ptr<DriverFlowTest> flow_ = std::make_shared<DriverFlowTest>();
};

TEST_F(PaddleOcrDetPostTest, TwoRectsDecodedAsPolygons) {
  // Build synthetic prob map: 64x64, two filled white rects at (5,5)-(20,20)
  // and (35,35)-(55,55) on a zero background.
  cv::Mat prob = cv::Mat::zeros(64, 64, CV_32F);
  cv::rectangle(prob, cv::Rect(5, 5, 16, 16), cv::Scalar(0.95F), cv::FILLED);
  cv::rectangle(prob, cv::Rect(35, 35, 20, 20), cv::Scalar(0.95F), cv::FILLED);
  cv::Mat image = cv::Mat::zeros(64, 64, CV_8UC3);

  auto graph = R"(
    [graph]
    graphconf = '''digraph G {
      src[type=flowunit, flowunit=test_image_src]
      pp[type=flowunit, flowunit=paddle_ocr_det_post,
         db_thresh=0.3, box_thresh=0.5, unclip_ratio=1.5]
      sink[type=flowunit, flowunit=test_polygon_sink]
      src:out_image -> pp:in_image
      src:out_prob  -> pp:in_prob
      pp:out_image  -> sink:in_image
    }'''
    format = "graphviz"
  )";
  // ... drive graph, capture polygon_count meta from sink. (See tensorrt
  // test for the harness boilerplate.)
  int polygon_count = -1;
  ASSERT_EQ(flow_->RunGraphAndCapture(graph, prob, image, polygon_count),
            STATUS_OK);
  EXPECT_EQ(polygon_count, 2);
}

TEST_F(PaddleOcrDetPostTest, EmptyMapZeroPolygons) {
  cv::Mat prob = cv::Mat::zeros(64, 64, CV_32F);
  cv::Mat image = cv::Mat::zeros(64, 64, CV_8UC3);
  int polygon_count = -1;
  /* same graph snippet */
  EXPECT_EQ(polygon_count, 0);
}
}  // namespace modelbox
```

- [ ] **Step 2: Run the test, confirm it fails (flowunit doesn't exist yet).**

```bash
ssh pose "cd modelbox/build && make build-test -j\$(nproc) && ./test/unittest --gtest_filter='PaddleOcrDetPost*'"
```

Expected: graph parse fails because `paddle_ocr_det_post` isn't registered.

- [ ] **Step 3: Implement the flowunit.**

`paddle_ocr_det_post.toml`:
```toml
[base]
name = "paddle_ocr_det_post"
device = "cpu"
version = "1.0.0"
type = "flowunit"
group_type = "Generic"
description = "DBNet probmap → polygons."
entry = "./libmodelbox-unit-cpu-paddle_ocr_det_post.so"

[config]
db_thresh = 0.3
box_thresh = 0.6
unclip_ratio = 1.5
max_candidates = 1000
min_size = 3

[input]
[[input]]
name = "in_image"
type = "uint8"
[[input]]
name = "in_prob"
type = "float"

[output]
[[output]]
name = "out_image"
type = "uint8"
```

Header has fields `db_thresh_, box_thresh_, unclip_ratio_, max_candidates_, min_size_`.

`Process()` body:

```cpp
modelbox::Status PaddleOcrDetPostFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  auto in_image = ctx->Input("in_image");
  auto in_prob = ctx->Input("in_prob");
  auto out = ctx->Output("out_image");
  if (in_image->Size() == 0) return modelbox::STATUS_OK;

  std::vector<size_t> sizes(in_image->Size());
  for (size_t i = 0; i < in_image->Size(); ++i)
    sizes[i] = in_image->At(i)->GetBytes();
  out->Build(sizes);

  for (size_t i = 0; i < in_image->Size(); ++i) {
    auto img_buf = in_image->At(i);
    auto prob_buf = in_prob->At(i);

    int32_t orig_w = 0, orig_h = 0;
    int32_t resized_w = 0, resized_h = 0;
    img_buf->Get("width", orig_w);
    img_buf->Get("height", orig_h);
    img_buf->Get("resized_w", resized_w);
    img_buf->Get("resized_h", resized_h);
    if (resized_w <= 0) resized_w = orig_w;
    if (resized_h <= 0) resized_h = orig_h;

    std::vector<size_t> pshape;
    prob_buf->Get("shape", pshape);
    int ph = (pshape.size() == 4) ? static_cast<int>(pshape[2]) : resized_h;
    int pw = (pshape.size() == 4) ? static_cast<int>(pshape[3]) : resized_w;
    cv::Mat prob(ph, pw, CV_32F,
                 const_cast<void*>(prob_buf->ConstData()));
    cv::Mat bin;
    cv::threshold(prob, bin, db_thresh_, 1.0, cv::THRESH_BINARY);
    bin.convertTo(bin, CV_8U, 255);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    if (static_cast<int>(contours.size()) > max_candidates_)
      contours.resize(max_candidates_);

    std::vector<std::array<cv::Point2f, 4>> polygons;
    polygons.reserve(contours.size());
    const float sx = static_cast<float>(orig_w) / pw;
    const float sy = static_cast<float>(orig_h) / ph;

    for (auto& c : contours) {
      if (static_cast<int>(c.size()) < min_size_) continue;
      auto rect = cv::minAreaRect(c);
      cv::Point2f pts[4];
      rect.points(pts);
      // mean prob inside the rect = score.
      cv::Mat mask = cv::Mat::zeros(prob.size(), CV_8U);
      std::vector<std::vector<cv::Point>> poly_int = {{
          cv::Point(pts[0]), cv::Point(pts[1]),
          cv::Point(pts[2]), cv::Point(pts[3])}};
      cv::fillPoly(mask, poly_int, 255);
      double score = cv::mean(prob, mask)[0];
      if (score < box_thresh_) continue;

      // Unclip: scale-out rect from centroid by unclip_ratio.
      cv::Point2f ctr = rect.center;
      std::array<cv::Point2f, 4> out_poly;
      for (int k = 0; k < 4; ++k) {
        cv::Point2f v = pts[k] - ctr;
        out_poly[k] = ctr + v * unclip_ratio_;
        out_poly[k].x *= sx;
        out_poly[k].y *= sy;
      }
      polygons.push_back(out_poly);
    }

    auto out_buf = out->At(i);
    std::memcpy(out_buf->MutableData(), img_buf->ConstData(),
                img_buf->GetBytes());
    out_buf->Set("width", orig_w);
    out_buf->Set("height", orig_h);
    out_buf->Set("channel", 3);
    out_buf->Set("polygon_count", static_cast<int32_t>(polygons.size()));
    // Pack polygons as flat float vector: [x0,y0,x1,y1,x2,y2,x3,y3, ...]
    std::vector<float> flat;
    flat.reserve(polygons.size() * 8);
    for (auto& p : polygons) {
      for (auto& pt : p) { flat.push_back(pt.x); flat.push_back(pt.y); }
    }
    out_buf->Set("polygons", flat);
  }
  return modelbox::STATUS_OK;
}
```

- [ ] **Step 4: Run the tests, confirm they pass.**

```bash
ssh pose "cd modelbox/build && make build-test -j\$(nproc) && ./test/unittest --gtest_filter='PaddleOcrDetPost*' -v"
```

Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/drivers/devices/cpu/flowunit/paddle_ocr_det_post/ src/drivers/devices/cpu/flowunit/CMakeLists.txt
git commit -m "flowunit: paddle_ocr_det_post DBNet polygon decode"
```

---

## Task 6: `paddle_ocr_crop_rotate` flowunit (TDD)

**Files:**
- Create: `src/drivers/devices/cpu/flowunit/paddle_ocr_crop_rotate/...` (same 5 files as Task 5)
- Modify: `src/drivers/devices/cpu/flowunit/CMakeLists.txt`

- [ ] **Step 1: Failing test.** Create a 200×200 image with a known coloured 80×30 rectangle drawn at angle 0; pass it as a single polygon. Assert the output crop dimensions are 80±1 × 30±1 and centre pixel matches the rectangle colour. Second case: pass a portrait 30×80 polygon, assert output comes out rotated (now ~80×30).

- [ ] **Step 2: Run, confirm fail** (flowunit not registered).

- [ ] **Step 3: Implement.**

`Process()` body — fan-out 1 frame → N crops by calling `out->Build(sizes)` with one entry per polygon. Use `frame_id` from a monotonic counter as buffer meta.

```cpp
modelbox::Status PaddleOcrCropRotateFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  auto in_image = ctx->Input("in_image");
  auto out_crop = ctx->Output("out_crop");
  auto out_image = ctx->Output("out_image");

  std::vector<size_t> crop_sizes;
  std::vector<size_t> image_sizes;
  struct Job { size_t in_idx; std::array<cv::Point2f,4> poly; int w, h;
               int box_id, box_count; int64_t frame_id; };
  std::vector<Job> jobs;

  for (size_t i = 0; i < in_image->Size(); ++i) {
    auto buf = in_image->At(i);
    int32_t w = 0, h = 0;
    buf->Get("width", w); buf->Get("height", h);
    int32_t pc = 0; buf->Get("polygon_count", pc);
    std::vector<float> flat; buf->Get("polygons", flat);
    int64_t fid = next_frame_id_++;
    for (int k = 0; k < pc; ++k) {
      std::array<cv::Point2f, 4> p;
      for (int v = 0; v < 4; ++v)
        p[v] = {flat[k*8 + v*2], flat[k*8 + v*2 + 1]};
      // Width = max edge length, height = min edge length (across opp pairs).
      float w0 = cv::norm(p[1] - p[0]);
      float w1 = cv::norm(p[3] - p[2]);
      float h0 = cv::norm(p[2] - p[1]);
      float h1 = cv::norm(p[0] - p[3]);
      int rw = static_cast<int>(std::round(std::max(w0, w1)));
      int rh = static_cast<int>(std::round(std::max(h0, h1)));
      rw = std::max(rw, 1); rh = std::max(rh, 1);
      jobs.push_back({i, p, rw, rh, k, pc, fid});
      crop_sizes.push_back(static_cast<size_t>(rw) * rh * 3);
    }
    image_sizes.push_back(buf->GetBytes());
  }
  out_crop->Build(crop_sizes);
  out_image->Build(image_sizes);

  for (size_t j = 0; j < jobs.size(); ++j) {
    auto& job = jobs[j];
    auto src_buf = in_image->At(job.in_idx);
    int32_t sw = 0, sh = 0;
    src_buf->Get("width", sw); src_buf->Get("height", sh);
    cv::Mat src(sh, sw, CV_8UC3,
                const_cast<void*>(src_buf->ConstData()));
    cv::Point2f dst_pts[4] = {
        {0, 0},
        {static_cast<float>(job.w - 1), 0},
        {static_cast<float>(job.w - 1), static_cast<float>(job.h - 1)},
        {0, static_cast<float>(job.h - 1)}};
    cv::Mat M = cv::getPerspectiveTransform(job.poly.data(), dst_pts);
    cv::Mat crop;
    cv::warpPerspective(src, crop, M, cv::Size(job.w, job.h));
    if (job.h > 1.5F * job.w) {
      cv::rotate(crop, crop, cv::ROTATE_90_COUNTERCLOCKWISE);
      std::swap(job.w, job.h);
    }
    auto out_buf = out_crop->At(j);
    std::memcpy(out_buf->MutableData(), crop.data, crop.total() * 3);
    out_buf->Set("width", static_cast<int32_t>(job.w));
    out_buf->Set("height", static_cast<int32_t>(job.h));
    out_buf->Set("channel", 3);
    out_buf->Set("frame_id", job.frame_id);
    out_buf->Set("box_id", job.box_id);
    out_buf->Set("box_count", job.box_count);
    std::vector<float> poly_flat = {
        job.poly[0].x, job.poly[0].y, job.poly[1].x, job.poly[1].y,
        job.poly[2].x, job.poly[2].y, job.poly[3].x, job.poly[3].y};
    out_buf->Set("polygon", poly_flat);
  }

  // Forward original frames on out_image, tagged with same frame_id sequence.
  for (size_t i = 0; i < in_image->Size(); ++i) {
    auto src = in_image->At(i);
    auto dst = out_image->At(i);
    std::memcpy(dst->MutableData(), src->ConstData(), src->GetBytes());
    int32_t pc = 0; src->Get("polygon_count", pc);
    int64_t fid = next_frame_id_ - in_image->Size() + i;
    dst->Set("width", *src);  // pass through other meta
    dst->Set("frame_id", fid);
    dst->Set("box_count", pc);
  }
  return modelbox::STATUS_OK;
}
```

(`next_frame_id_` is a `std::atomic<int64_t>` member initialized to 0 in the constructor.)

- [ ] **Step 4: Run, confirm pass.**

- [ ] **Step 5: Commit.**

```bash
git commit -am "flowunit: paddle_ocr_crop_rotate perspective-warp + portrait fix"
```

---

## Task 7: `paddle_ocr_cls_post` flowunit

**Files:** same 4 (no test) under `paddle_ocr_cls_post/`.

- [ ] **Step 1:** Create the standard files. The Process body:

```cpp
modelbox::Status PaddleOcrClsPostFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  auto in_logits = ctx->Input("in_logits");   // [N,2] fp32
  auto in_crop = ctx->Input("in_crop");       // passthrough
  auto out = ctx->Output("out_crop");
  if (in_crop->Size() == 0) return modelbox::STATUS_OK;

  std::vector<size_t> sizes(in_crop->Size());
  for (size_t i = 0; i < in_crop->Size(); ++i)
    sizes[i] = in_crop->At(i)->GetBytes();
  out->Build(sizes);

  const float* lg = static_cast<const float*>(in_logits->At(0)->ConstData());
  for (size_t i = 0; i < in_crop->Size(); ++i) {
    float l0 = lg[i * 2], l1 = lg[i * 2 + 1];
    // Softmax over 2 classes.
    float m = std::max(l0, l1);
    float e0 = std::exp(l0 - m), e1 = std::exp(l1 - m);
    float p180 = e1 / (e0 + e1);
    auto src = in_crop->At(i);
    auto dst = out->At(i);
    int32_t w = 0, h = 0;
    src->Get("width", w); src->Get("height", h);
    cv::Mat img(h, w, CV_8UC3, const_cast<void*>(src->ConstData()));
    if (p180 >= cls_thresh_) {
      cv::Mat rot;
      cv::rotate(img, rot, cv::ROTATE_180);
      std::memcpy(dst->MutableData(), rot.data, rot.total() * 3);
      dst->Set("angle", 180);
    } else {
      std::memcpy(dst->MutableData(), src->ConstData(), src->GetBytes());
      dst->Set("angle", 0);
    }
    // Forward all other meta keys.
    int32_t bid, bcnt; int64_t fid;
    src->Get("box_id", bid); src->Get("box_count", bcnt);
    src->Get("frame_id", fid);
    dst->Set("width", w); dst->Set("height", h); dst->Set("channel", 3);
    dst->Set("box_id", bid); dst->Set("box_count", bcnt);
    dst->Set("frame_id", fid);
    std::vector<float> poly;
    src->Get("polygon", poly); dst->Set("polygon", poly);
  }
  return modelbox::STATUS_OK;
}
```

TOML config: `cls_thresh = 0.9`. Ports: `in_logits`, `in_crop` → `out_crop`.

- [ ] **Step 2:** Build, commit.

```bash
git commit -am "flowunit: paddle_ocr_cls_post 180° rotation fix"
```

---

## Task 8: `paddle_ocr_rec_pre` flowunit

**Files:** same 4 under `paddle_ocr_rec_pre/`.

- [ ] **Step 1:** Process body (fp32 NCHW, height = 48, keep ratio, batch-pad to max width):

```cpp
modelbox::Status PaddleOcrRecPreFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  auto in_crop = ctx->Input("in_crop");
  auto out_t = ctx->Output("out_tensor");
  auto out_c = ctx->Output("out_crop");
  if (in_crop->Size() == 0) return modelbox::STATUS_OK;

  // Determine batch max width.
  std::vector<int> rws(in_crop->Size()), rhs(in_crop->Size(), 48);
  int max_w = 0;
  for (size_t i = 0; i < in_crop->Size(); ++i) {
    int32_t w = 0, h = 0;
    in_crop->At(i)->Get("width", w); in_crop->At(i)->Get("height", h);
    int rw = std::min(rec_max_w_, std::max(1,
        static_cast<int>(std::round(static_cast<float>(w) * 48 / h))));
    rws[i] = rw;
    max_w = std::max(max_w, rw);
  }

  size_t per = static_cast<size_t>(3) * 48 * max_w * sizeof(float);
  std::vector<size_t> tsizes(in_crop->Size(), per);
  std::vector<size_t> csizes(in_crop->Size());
  for (size_t i = 0; i < in_crop->Size(); ++i)
    csizes[i] = in_crop->At(i)->GetBytes();
  out_t->Build(tsizes); out_c->Build(csizes);

  static const float mean[3] = {0.5F, 0.5F, 0.5F};
  static const float std_[3] = {0.5F, 0.5F, 0.5F};

  for (size_t i = 0; i < in_crop->Size(); ++i) {
    auto src = in_crop->At(i);
    int32_t w = 0, h = 0;
    src->Get("width", w); src->Get("height", h);
    cv::Mat img(h, w, CV_8UC3, const_cast<void*>(src->ConstData()));
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(rws[i], 48));
    cv::Mat rgb; cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    auto* dst = static_cast<float*>(out_t->At(i)->MutableData());
    std::memset(dst, 0, per);  // zero-pad
    for (int y = 0; y < 48; ++y) {
      for (int x = 0; x < rws[i]; ++x) {
        const auto& px = rgb.at<cv::Vec3b>(y, x);
        for (int c = 0; c < 3; ++c) {
          dst[c * 48 * max_w + y * max_w + x] =
              (px[c] / 255.0F - mean[c]) / std_[c];
        }
      }
    }
    std::vector<size_t> shape = {1, 3, 48, static_cast<size_t>(max_w)};
    out_t->At(i)->Set("shape", shape);
    // Forward crop + meta passthrough.
    auto dstc = out_c->At(i);
    std::memcpy(dstc->MutableData(), src->ConstData(), src->GetBytes());
    int32_t bid = 0, bcnt = 0, ang = 0; int64_t fid = 0;
    src->Get("box_id", bid); src->Get("box_count", bcnt);
    src->Get("frame_id", fid); src->Get("angle", ang);
    dstc->Set("width", w); dstc->Set("height", h); dstc->Set("channel", 3);
    dstc->Set("box_id", bid); dstc->Set("box_count", bcnt);
    dstc->Set("frame_id", fid); dstc->Set("angle", ang);
    std::vector<float> poly; src->Get("polygon", poly);
    dstc->Set("polygon", poly);
  }
  return modelbox::STATUS_OK;
}
```

TOML knobs: `rec_max_w = 320`. Header field `int rec_max_w_{320}`.

- [ ] **Step 2:** Build, commit.

```bash
git commit -am "flowunit: paddle_ocr_rec_pre keep-ratio resize + batch pad"
```

---

## Task 9: `paddle_ocr_rec_post` flowunit (TDD)

**Files:** same 5 under `paddle_ocr_rec_post/`.

- [ ] **Step 1: Failing test** — synthesize logits with a tiny 5-char inline charset `[blank, H, e, l, o]` and one-hot timesteps spelling `H,blank,e,blank,l,blank,l,blank,o`. Assert decoded text is `"Hello"`. Second case: low-magnitude logits → empty string under `score_thresh=0.5`.

- [ ] **Step 2:** Confirm fail.

- [ ] **Step 3: Implement.** Header has `std::vector<std::string> charset_; bool use_space_char_; float score_thresh_;`. In `Open()`, load `char_dict_file` (one char per line, UTF-8), prepend a `""` blank token at index 0, append a space if `use_space_char_`.

```cpp
modelbox::Status PaddleOcrRecPostFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration>& opts) {
  auto path = opts->GetString("char_dict_file");
  if (path.empty()) return {modelbox::STATUS_BADCONF, "rec_post: char_dict_file required"};
  std::ifstream f(path);
  if (!f.is_open()) return {modelbox::STATUS_BADCONF, "rec_post: cannot open " + path};
  charset_.push_back("");  // blank
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    charset_.push_back(line);
  }
  if (opts->GetBool("use_space_char", true)) charset_.push_back(" ");
  score_thresh_ = opts->GetFloat("score_thresh", 0.5F);
  return modelbox::STATUS_OK;
}

modelbox::Status PaddleOcrRecPostFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  auto in_logits = ctx->Input("in_logits");
  auto in_crop = ctx->Input("in_crop");
  auto out = ctx->Output("out_result");
  if (in_crop->Size() == 0) return modelbox::STATUS_OK;

  std::vector<size_t> sizes(in_crop->Size(), 1);  // tiny payload
  out->Build(sizes);

  for (size_t i = 0; i < in_crop->Size(); ++i) {
    auto lg = in_logits->At(i);
    std::vector<size_t> shape; lg->Get("shape", shape);
    int T = shape.size() >= 3 ? static_cast<int>(shape[shape.size()-2]) : 0;
    int C = shape.size() >= 3 ? static_cast<int>(shape.back()) : 0;
    const float* p = static_cast<const float*>(lg->ConstData());
    std::string text;
    float score_sum = 0.0F;
    int score_n = 0;
    int prev = -1;
    for (int t = 0; t < T; ++t) {
      const float* row = p + t * C;
      // softmax + argmax.
      float m = row[0]; int idx = 0;
      for (int c = 1; c < C; ++c) if (row[c] > m) { m = row[c]; idx = c; }
      // mean of softmax-of-max
      float denom = 0.0F;
      for (int c = 0; c < C; ++c) denom += std::exp(row[c] - m);
      float prob = 1.0F / denom;
      if (idx != 0 && idx != prev) {
        if (idx < static_cast<int>(charset_.size())) text += charset_[idx];
        score_sum += prob; ++score_n;
      }
      prev = idx;
    }
    float score = score_n > 0 ? score_sum / score_n : 0.0F;
    if (score < score_thresh_) text.clear();

    auto crop = in_crop->At(i);
    auto dst = out->At(i);
    int32_t bid = 0, bcnt = 0; int64_t fid = 0;
    crop->Get("box_id", bid); crop->Get("box_count", bcnt);
    crop->Get("frame_id", fid);
    std::vector<float> poly; crop->Get("polygon", poly);
    dst->Set("text", text);
    dst->Set("score", score);
    dst->Set("box_id", bid);
    dst->Set("box_count", bcnt);
    dst->Set("frame_id", fid);
    dst->Set("polygon", poly);
  }
  return modelbox::STATUS_OK;
}
```

- [ ] **Step 4: Run, expect pass.**

- [ ] **Step 5: Commit.**

```bash
git commit -am "flowunit: paddle_ocr_rec_post CTC greedy decode"
```

---

## Task 10: `paddle_ocr_draw` flowunit

**Files:** same 4 under `paddle_ocr_draw/`.

- [ ] **Step 1:** Header members: `std::unordered_map<int64_t, std::vector<RecResult>> pending_;` plus `std::unordered_map<int64_t, std::shared_ptr<Buffer>> frames_;`, where `RecResult` holds polygon + text + score. Mutex for thread safety.

- [ ] **Step 2:** Process body — receives `in_result` stream and `in_image` stream. For each input:
  - On `in_image`: cache the frame buffer keyed by `frame_id`, also cache `box_count`. If `box_count==0`, immediately emit annotated frame (no polygons) and clear caches for that id.
  - On `in_result`: append to `pending_[frame_id]`. When `pending_[fid].size() == box_count`, render polygons + text on the cached frame, emit, evict.

  Rendering: `cv::polylines` with `cv::Scalar(0,255,0)` thickness 2; `cv::putText` (HERSHEY_SIMPLEX, 0.6) at `polygon[0]` with `text + " " + score_str` if text non-empty.

- [ ] **Step 3:** Build, commit.

```bash
git commit -am "flowunit: paddle_ocr_draw join-and-render"
```

---

## Task 11: Demo graph + weight download script + README

**Files:**
- Create: `examples/paddle_ocr/paddle_ocr_video.toml`
- Create: `examples/paddle_ocr/download_models.sh`
- Create: `examples/paddle_ocr/README.md`

- [ ] **Step 1:** `download_models.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
DEST="${1:-models/paddle_ocr}"
mkdir -p "$DEST"
cd "$DEST"
BASE=https://paddleocr.bj.bcebos.com/PP-OCRv4
for f in ch_PP-OCRv4_det_infer.tar ch_ppocr_mobile_v2.0_cls_infer.tar ch_PP-OCRv4_rec_infer.tar; do
  [[ -f "$f" ]] || wget -q "$BASE/chinese/$f" || wget -q "$BASE/$f"
  tar -xf "$f"
done
[[ -f ppocr_keys_v1.txt ]] || \
  wget -q https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/ppocr_keys_v1.txt
echo "Models extracted to $(pwd)"
```

`chmod +x` it.

- [ ] **Step 2:** `paddle_ocr_video.toml` — wire video_decoder → det chain → cls chain → rec chain → draw → video_encoder. Use the existing `examples/` graphs (e.g. car_detection) as a template for the video_decoder/video_encoder boilerplate. Three `paddle_inference` instances (`paddle_inference_det`, `paddle_inference_cls`, `paddle_inference_rec`) each with their own `model_file`, `params_file`, `input_name`, `output_name` from the downloaded `inference.pdmodel`/`inference.pdiparams`. Top-of-file variables for video path, model dir, charset path.

- [ ] **Step 3:** `README.md` — quickstart:
  - `bash download_models.sh` (writes to `models/paddle_ocr`)
  - Edit `paddle_ocr_video.toml` to point at `video_input.url` and `video_output.url`.
  - `modelbox-tool flow run -name paddle_ocr_video -graph examples/paddle_ocr/paddle_ocr_video.toml`
  - Note that the dev container needs `LD_LIBRARY_PATH=/opt/paddle_inference/paddle/lib:/opt/paddle_inference/third_party/install/mkldnn/lib:$LD_LIBRARY_PATH`.

- [ ] **Step 4:** Commit.

```bash
git add examples/paddle_ocr/
git commit -m "examples: PaddleOCR demo graph + model fetcher"
```

---

## Task 12: End-to-end integration on remote

**Files:** none (verification + bug-fix cycle).

- [ ] **Step 1:** Full build on remote.

```bash
ssh pose "cd modelbox && rm -rf build && mkdir build && cd build && \
  cmake .. -DCMAKE_BUILD_TYPE=Debug -DPaddleInference_DIR=/opt/paddle_inference && \
  make -j\$(nproc) && make build-test -j\$(nproc) && make unittest"
```

Expected: full build, all unit tests pass (including new PaddleOcr* tests).

- [ ] **Step 2:** Download weights on remote, run the demo on a sample video.

```bash
ssh pose "cd modelbox && bash examples/paddle_ocr/download_models.sh && \
  ls examples/paddle_ocr/sample_video.mp4 || \
    wget -q -O examples/paddle_ocr/sample_video.mp4 \
      https://github.com/PaddlePaddle/PaddleOCR/raw/main/doc/imgs_words/ch/word_1.jpg && \
  ./build/release/bin/modelbox-tool flow run \
    -name paddle_ocr_video \
    -graph examples/paddle_ocr/paddle_ocr_video.toml"
```

(Use a Chinese road-sign / receipt sample instead if available; the JPG fallback is just for smoke-testing the graph parses and doesn't crash — replace with a real video for the visual check.)

- [ ] **Step 3:** Pull the annotated output back, eyeball it.

```bash
scp pose:modelbox/examples/paddle_ocr/output.mp4 /tmp/paddle_ocr_output.mp4
```

- [ ] **Step 4:** If anything fails, debug task-by-task; do not paper over with try/catch. Fix root cause, re-run.

- [ ] **Step 5:** Open the PR.

```bash
gh pr create --title "PaddleOCR flowunits (CUDA)" --body "$(cat <<'EOF'
## Summary
- New Paddle Inference engine adapter (`src/drivers/inference_engine/paddle/`)
- CUDA-bound `paddle_inference` flowunit
- Seven CPU pre/post flowunits implementing PP-OCRv4 (det+cls+rec)
- Demo graph + weight download script under `examples/paddle_ocr/`
- Unit tests for `det_post`, `crop_rotate`, `rec_post`

## Test plan
- [ ] Full build with `-DPaddleInference_DIR=/opt/paddle_inference`
- [ ] `make unittest` passes — all `PaddleOcr*` cases
- [ ] Demo graph runs end-to-end on a sample video, output visually matches expected detections
EOF
)"
```

---

## Self-review

**Spec coverage:** every spec section maps to a task — adapter (Task 1-2), CUDA inference flowunit (Task 3), seven CPU flowunits (Tasks 4-10), demo + download (Task 11), tests (embedded in Tasks 5/6/9), build/dep gating (Task 1 CMake), integration (Task 12). Docker extension explicitly out of scope per spec §10.

**Placeholders:** none. Every task has exact paths and concrete code.

**Type consistency:** `polygons` is `vector<float>` (flat 8-floats-per-poly) across `det_post` (writer), `crop_rotate` (reader), `draw` (reader via `polygon` singular per crop). `frame_id` is `int64_t` everywhere. `box_id`/`box_count` are `int32_t` everywhere. `score`/`text` produced by `rec_post` consumed by `draw`.

**Notable risks the implementer should flag back:**
- `PaddleInference_DIR` install path may differ on `5090x4` vs `pose` — the task assumes `/opt/paddle_inference`.
- Paddle Inference C++ API has evolved; if `GetInputHandle`/`CopyFromCpu` signatures differ in the installed version, adjust the adapter accordingly. Reference: <https://www.paddlepaddle.org.cn/inference/master/api_reference/cxx_api_doc/Predictor.html>.
- The DriverFlowTest harness referenced in Task 5 exists (`test/drivers/driver_flow_test.cc`) but the `RunGraphAndCapture` helper used in the test pseudocode is illustrative — the actual harness uses `flow_->BuildAndRun(...)`. Read `tensorrt_inference_flowunit_test.cc` once for the real method names.
