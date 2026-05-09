# PaddleOCR Flowunits for ModelBox (CUDA)

**Date:** 2026-05-09
**Status:** Approved (design)
**Target:** Linux x86_64 + CUDA. Will not build on macOS — same restriction as the rest of the engine (`CLAUDE.md`).

## 1. Goal

Bring the full PP-OCRv4 pipeline (text detection → angle classification → text recognition) into ModelBox as native flowunits driven by Paddle Inference C++ on NVIDIA GPU. End user gets a runnable graph that decodes a video, OCRs every frame, and re-encodes an annotated video.

## 2. Non-goals

- ONNX-based fallback path (not in scope; can be layered later via existing OpenVINO/TensorRT engines).
- Ascend / Rockchip / Intel-GPU device variants (later).
- Auto-fetching weights at build time, or extending `docker/prepare_for_dev.sh` to install Paddle Inference (follow-up).
- Multi-language model selection logic in a single graph (one graph = one charset).

## 3. Layout

```
src/drivers/inference_engine/paddle/
    CMakeLists.txt                         # find_package(PaddleInference CONFIG)
    paddle_inference.{h,cc}                # adapter wrapping paddle_infer::Predictor
    paddle_inference_flowunit.{h,cc}       # ModelBox flowunit shell

src/drivers/devices/cuda/flowunit/paddle/
    CMakeLists.txt                         # gated on MODELBOX_PADDLE_FOUND + CUDA_FOUND
    paddle_inference_flowunit.{h,cc}       # CUDA-bound instantiation, mirrors tensorrt_inference_flowunit

src/drivers/devices/cpu/flowunit/
    paddle_ocr_det_pre/                    # resize-to-mult-of-32 + normalize
    paddle_ocr_det_post/                   # DBNet probmap → polygons
    paddle_ocr_crop_rotate/                # perspective-crop + portrait rotate
    paddle_ocr_cls_post/                   # 180° rotation fix
    paddle_ocr_rec_pre/                    # H=48 keep-ratio + batch-pad
    paddle_ocr_rec_post/                   # CTC greedy decode
    paddle_ocr_draw/                       # polylines + putText join unit

examples/paddle_ocr/
    paddle_ocr_video.toml                  # full demo graph
    download_models.sh                     # fetch PP-OCRv4 det/cls/rec + charset
    README.md

test/drivers/cpu/flowunit/
    paddle_ocr_det_post_test.cc
    paddle_ocr_rec_post_test.cc
    paddle_ocr_crop_rotate_test.cc
```

Each new flowunit dir mirrors the structure of an existing `yolo_*_post` directory: `<name>.toml` descriptor, `<name>_flowunit.{h,cc}`, `CMakeLists.txt`, registered with the parent's `add_subdirectory`.

## 4. Data flow

```
video_decoder
    └─► paddle_ocr_det_pre ─► paddle_inference[det] ─► paddle_ocr_det_post
                                                            │
                                       (frame + polygons)   ▼
                                               paddle_ocr_crop_rotate  ── fan-out: 1 frame → N crops
                                                            │
                                                            ▼
                                              paddle_inference[cls] ─► paddle_ocr_cls_post
                                                            │
                                                            ▼
                                                  paddle_ocr_rec_pre
                                                            │
                                                            ▼
                                              paddle_inference[rec] ─► paddle_ocr_rec_post
                                                            │
                                                            ▼
                                                  paddle_ocr_draw  ── join: N crops → 1 annotated frame
                                                            │
                                                            ▼
                                                      video_encoder
```

Stream semantics:

- `paddle_ocr_det_post` attaches `polygons: vector<array<Point2f,4>>` and `polygon_count: int` to the frame buffer meta and forwards the original frame.
- `paddle_ocr_crop_rotate` expands the stream: emits `polygon_count` crop buffers, each tagged with `frame_id` (monotonic per input frame), `box_id` (0..N-1), `box_count` (== `polygon_count`), and the source polygon. The original frame is forwarded on a side port to `paddle_ocr_draw`.
- `paddle_ocr_draw` joins on `frame_id`, accumulating `box_count` results before emitting an annotated frame downstream.
- This fan-out / fan-in pattern reuses the `MatchStream` machinery already used by yolo demo graphs — no scheduler changes.

Buffer payload conventions:

- Pre flowunits emit raw fp32 NCHW `Tensor` buffers (Paddle's expected format).
- The Paddle inference flowunit accepts/emits ModelBox tensor buffers; tensor names are read from the loaded model and may be overridden by TOML keys `input_name` / `output_name`.
- Post flowunits attach decoded results as buffer meta (`polygons`, `angle`, `text`, `score`) so downstream units don't reparse tensors.

## 5. Paddle Inference adapter

`src/drivers/inference_engine/paddle/paddle_inference.h`:

```cpp
struct PaddleInferenceParams {
  std::string model_file;       // .pdmodel
  std::string params_file;      // .pdiparams
  std::string device = "gpu";   // "cpu" | "gpu"
  int gpu_id = 0;
  bool enable_trt = false;
  int trt_workspace_mb = 256;
  std::string trt_precision = "fp32";   // fp32 | fp16 | int8
  bool enable_mkldnn = false;           // cpu only
  int cpu_threads = 4;                  // cpu only
  std::vector<std::string> input_names;   // optional override
  std::vector<std::string> output_names;  // optional override
};

class PaddleInference {
 public:
  Status Init(const PaddleInferenceParams& p);
  Status Infer(const std::vector<std::shared_ptr<ModelBoxTensor>>& inputs,
               std::vector<std::shared_ptr<ModelBoxTensor>>& outputs);
 private:
  std::shared_ptr<paddle_infer::Predictor> predictor_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
};
```

The adapter handles:
- Predictor creation via `paddle_infer::CreatePredictor(Config)`.
- Reshape per-call from buffer dims (Paddle Inference supports dynamic shapes natively).
- Copying ModelBox tensors → Paddle input handles → output handles → ModelBox tensors. The copy is zero-copy when the buffer's device matches (i.e. CUDA→GPU), full copy otherwise.
- Lifetime: one predictor per flowunit instance, held for the flowunit's lifetime.

The `paddle_inference_flowunit` (cuda dir) is a thin wrapper that:
- Reads TOML keys mapped 1:1 to `PaddleInferenceParams` plus the standard ModelBox flowunit keys (name, type, group_type, device, queue_size, batch_size).
- Forces `device=gpu` and reads `gpu_id` from the device context if not explicitly set.
- Is instantiated three times in the demo graph (det / cls / rec) with different model paths.

## 6. CPU postprocess contracts

### `paddle_ocr_det_post`
- **In:** probmap tensor `[1, 1, H, W]` fp32 on `in_prob`; original frame on `in_image` (passthrough).
- **Out:** frame buffer on `out_image` with meta `polygons` and `polygon_count`.
- **Algorithm:** `cv::threshold(prob, bin, db_thresh)` → `cv::findContours` → for each contour, min-area rect + mean-prob score; drop if `score < box_thresh`; unclip via Vatti polygon offset (Clipper2 if vendored under `thirdparty/`, else fallback to scalar dilation of the rect by `unclip_ratio`).
- **TOML knobs:** `db_thresh=0.3`, `box_thresh=0.6`, `unclip_ratio=1.5`, `max_candidates=1000`, `min_size=3`.

### `paddle_ocr_crop_rotate`
- **In:** frame + polygons.
- **Out:** N crop buffers tagged `frame_id`, `box_id`, `box_count`, `polygon`.
- **Algorithm:** `cv::getPerspectiveTransform` from polygon → axis-aligned rect with width = max edge length, height = min edge length; `cv::warpPerspective`. If `h > 1.5 * w`, rotate the crop 90° CCW so text reads left-to-right.

### `paddle_ocr_cls_post`
- **In:** classifier logits `[N, 2]` + crops (passthrough).
- **Out:** crops (rotated 180° in-place when predicted angle = 180° and `score >= cls_thresh`) with meta `angle` attached.
- **TOML knobs:** `cls_thresh=0.9`.

### `paddle_ocr_rec_post`
- **In:** recognition logits `[N, T, C]` fp32 (C = `len(charset) + 1`, blank at index 0) + crop meta passthrough.
- **Out:** result buffers with meta `text: string`, `score: float`, plus inherited `frame_id`, `box_id`, `box_count`, `polygon`.
- **Algorithm:** greedy CTC — argmax along C, collapse repeats, drop blanks; score = mean of per-step max softmax over kept timesteps. Charset loaded once at Init from `char_dict_file`.
- **TOML knobs:** `char_dict_file` (required path to e.g. `ppocr_keys_v1.txt`), `use_space_char=true`, `score_thresh=0.5` (text below threshold becomes empty string).

### `paddle_ocr_draw`
- **In:** result buffers + original frames, joined on `frame_id` using `box_count` to know when a frame is fully assembled.
- **Out:** annotated frame; `cv::polylines` for polygons; `cv::putText` for text (FreeType backend if its CMake target is found, else ASCII fallback for non-Latin charsets).

## 7. Pre-processing flowunits

### `paddle_ocr_det_pre`
- Resize the input frame so both sides are multiples of 32 (DBNet requirement) while keeping aspect ratio under a configurable `max_side_len=960`.
- Normalize: `(pixel/255 - mean) / std` with `mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`.
- Emit fp32 NCHW tensor on the model's input port plus the original frame on `out_image` (forwarded so `det_post` sees both).

### `paddle_ocr_rec_pre`
- Per-crop: resize to height 48, width = `round(w * 48 / h)`, capped at `rec_max_w=320`.
- Batch: pad widths to the max in the current batch with zero pixels.
- Same normalization constants as `det_pre`.

## 8. Demo graph

`examples/paddle_ocr/paddle_ocr_video.toml` wires:

```
video_input → video_decoder → paddle_ocr_det_pre →
paddle_inference[det] → paddle_ocr_det_post →
paddle_ocr_crop_rotate →
paddle_inference[cls] → paddle_ocr_cls_post →
paddle_ocr_rec_pre →
paddle_inference[rec] → paddle_ocr_rec_post →
paddle_ocr_draw → video_encoder → video_output
```

Every model path, charset path, and threshold is exposed at the top of the TOML. The README documents how to run `download_models.sh` first.

## 9. Tests

Three gtest files under `test/drivers/cpu/flowunit/`, all using the existing `MockFlow` pattern:

1. **`paddle_ocr_det_post_test.cc`** — synthesize a 1×1×64×64 probmap with two `cv::rectangle`d regions; assert `polygon_count == 2` and polygons enclose the regions. Edge cases: empty probmap → 0 polygons; all-low probability → 0 polygons after `box_thresh` cull.
2. **`paddle_ocr_rec_post_test.cc`** — synthesize logits that one-hot encode `H,e,l,l,o` (with an explicit blank between the doubled `l`s) using a tiny inline charset; assert decoded `"Hello"` and score ≈ 1.0. Edge case: uniformly low logits → empty string per `score_thresh`.
3. **`paddle_ocr_crop_rotate_test.cc`** — 200×200 image with a known coloured rectangle polygon; assert crop dimensions and corner pixel colours. Second case: portrait polygon → crop comes out rotated.

The Paddle inference flowunits and `det_pre`/`rec_pre` are not unit-tested — they're thin wrappers; correctness is exercised end-to-end via the demo graph.

Tests are added to `test/CMakeLists.txt` and gated on the same CMake variable as their flowunits.

## 10. Build & dependencies

- `inference_engine/paddle/CMakeLists.txt` calls `find_package(PaddleInference CONFIG)`. On success it exports `MODELBOX_PADDLE_FOUND=TRUE` to parent scope. On failure, the subdir produces no targets and prints a status message — same degradation pattern as `tensorrt`, `mindspore`, etc.
- `devices/cuda/flowunit/paddle/CMakeLists.txt` is gated on `MODELBOX_PADDLE_FOUND AND CUDA_FOUND`.
- CPU pre/post flowunits depend only on OpenCV + `libmodelbox` (already required). They always build, so users can hand-roll graphs that produce/consume OCR tensors even without Paddle (e.g. via ONNX through OpenVINO).
- `examples/paddle_ocr/download_models.sh` fetches PP-OCRv4 det/cls/rec inference tarballs from the upstream PaddleOCR release URL (default to `models/paddle_ocr/`, overridable). No auto-fetch at build time.
- Installing Paddle Inference into `docker/Dockerfile.cuda.develop.ubuntu` is **out of scope** here; users install it manually into the dev container until a follow-up extends `docker/prepare_for_dev.sh`.

## 11. Open follow-ups (not in this spec)

- ONNX path through OpenVINO/TensorRT for non-Paddle deployments.
- Per-device variants (Ascend, Rockchip).
- Multi-language: a charset selector flowunit that picks `ppocr_keys_*.txt` per crop (e.g. by language detector).
- `docker/prepare_for_dev.sh` extension to install Paddle Inference automatically.
