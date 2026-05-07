# TrackNet on Apple Silicon — Design

Date: 2026-05-07
Branch base: `main`
Upstream model: https://github.com/bcinno-dev/tracknet
Pretrained weights: `pose:/root/autodl-fs/repos/tracknet/exp_deep/TrackNet_best.pt`

## Goal

A standalone ModelBox graph that decodes a tennis video, runs **TrackNetDeep** via Core ML on Apple Silicon (ANE/GPU/CPU), draws the predicted ball position on each frame, and encodes an MP4. Parallel to the existing `apple_silicon_yolo*` demos — separate TOML, separate flowunits, no entanglement with `det_then_pose`.

## Variant

- Architecture: `TrackNetDeep` (4 encoder levels, ~46.7M params).
- Input mode: plain RGB, 9 channels (3 frames × 3).
- Reason: best F1@10 (0.9515) on the published checkpoints with the simplest preprocessing path.
- Out of scope for v1: small `TrackNet`, `bg_concat`, `seq5`.

## Deliverables (in-tree)

- `scripts/macos/convert_tracknet_coreml.py` — `.pt` → `.mlpackage` conversion. Run on `ssh pose`.
- `src/demo/apple_silicon_yolo/flowunit/tracknet_frame_stacker/` — stateful 3-frame ring buffer.
- `src/demo/apple_silicon_yolo/flowunit/tracknet_post/` — heatmap → centroid + draw on source-resolution frame.
- `src/demo/apple_silicon_yolo/flowunit/tracknet_deep/` — Core ML inference flowunit wrapper (`virtual_type = "coreml"`, ships the `.mlpackage`).
- `src/demo/apple_silicon_yolo/graph/apple_silicon_tracknet.toml.in`.
- `scripts/macos/run-apple-silicon-tracknet.sh` — convenience runner mirroring `run-apple-silicon-yolo.sh`.

Reused, untouched:
- `coreml_inference` flowunit on the `apple_silicon` device.
- Existing `video_decoder`, `resize`, `normalize`, `video_encoder` flowunits used by the YOLO demos.

## Data flow

```
input.mp4
   │
   ▼
video_decoder (cpu, pix_fmt=bgr) ────┐
   │                                  │ bypass to post for source-resolution draw
   ▼                                  │
resize (cpu, 512×288)                 │
   │                                  │
   ▼                                  │
normalize (cpu, ×1/255, no mean)      │
   │                                  │
   ▼                                  │
tracknet_frame_stacker (cpu)          │
   │  fp32 (9, 288, 512), [t-2,t-1,t] │
   ▼                                  │
tracknet_deep (apple_silicon, coreml) │
   │  fp32 (3, 288, 512) heatmaps     │
   ▼                                  │
tracknet_post (cpu) ◀─────────────────┘
   │  uint8 BGR (H, W, 3) with red ball circle
   ▼
video_encoder (cpu, mp4)
   │
   ▼
output.mp4
```

Per-frame correspondence: stacker emits one stacked tensor per input (sliding step=1), inference emits one heatmap-triplet per stacked input, post draws on one source frame per triplet. Frame rate end-to-end is preserved.

## Tensor contracts

| Edge | Shape | Dtype | Layout | Meta |
|---|---|---|---|---|
| `decoder → resize` | (H, W, 3) | uint8 | HWC BGR | width, height |
| `resize → normalize` | (288, 512, 3) | uint8 | HWC BGR | — |
| `normalize → stacker` | (3, 288, 512) | fp32 | CHW [0,1] BGR | — |
| `stacker → tracknet_deep` | (9, 288, 512) | fp32 | CHW concat[t-2,t-1,t] | — |
| `tracknet_deep → post` | (3, 288, 512) | fp32 | CHW per-frame heatmaps [0,1] | — |
| `decoder → post (bypass)` | (H, W, 3) | uint8 | HWC BGR | width, height |
| `post → encoder` | (H, W, 3) | uint8 | HWC BGR | width, height |

The model expects **BGR-or-RGB consistent with training**. Training reads frames via OpenCV (`cv2.imread` → BGR) and does not swap channels, so we keep BGR end-to-end. Confirmed against `dataset.py` during conversion-script development; note in conversion script header.

## Flowunit specs

### `tracknet_frame_stacker` (C++, `device=cpu`)

- Single-input single-primary-output flowunit. State: `prev1`, `prev2` as `std::vector<float>` (3·288·512 floats each); `frame_idx` counter. Cleared on `DataPre` / new session.
- Process step:
  1. `cur` = input tensor (3·288·512 floats).
  2. If `frame_idx == 0`: `out = [cur, cur, cur]`. If `frame_idx == 1`: `out = [prev1, prev1, cur]`. Else: `out = [prev2, prev1, cur]`.
  3. Shift: `prev2 ← prev1`; `prev1 ← cur`; `frame_idx++`.
  4. Allocate output `9·288·512·sizeof(float)`; three `memcpy`s.
- TOML config: none for v1; `seq_len=3`, `H=288`, `W=512` baked in.
- Multi-stream: state is per-flowunit-instance — out of scope; v1 assumes one stream per graph run.

### `tracknet_post` (C++, `device=cpu`)

- Inputs: `heatmaps` (fp32 3×288×512) and `source_frame` (uint8 BGR H×W×3) paired by stream order.
- Output: `frame_out` (uint8 BGR H×W×3) — drawn frame.
- Process step:
  1. Select channel `heatmap_channel` (default 2, the newest frame).
  2. Find peak; if `peak < score_thr` (default 0.3), forward source frame unmodified.
  3. Mask = pixels where `h > peak * mask_ratio` (default 0.5).
  4. Weighted centroid `(cx_lr, cy_lr)` over the masked region using heatmap values as weights.
  5. Scale: `cx = cx_lr * W / 512`, `cy = cy_lr * H / 288`.
  6. Draw filled circle (radius=`circle_radius`, color=`circle_color`) on the source frame.
  7. If `draw_heatmap` is true, alpha-blend a grayscale-mapped heatmap (resized to source res) at α=0.3 for debugging.
- TOML config: `score_thr=0.3`, `mask_ratio=0.5`, `circle_radius=6`, `circle_color="0,0,255"` (BGR), `draw_heatmap=false`, `heatmap_channel=2`.

### `tracknet_deep` (Core ML inference flowunit wrapper)

Mirrors `yolo_pose_detect/yolo_pose_detect.toml`:

```toml
[base]
name = "tracknet_deep"
device = "apple_silicon"
version = "1.0.0"
description = "TrackNetDeep ball-tracking inference on Apple Silicon via Core ML"
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

CMakeLists.txt for the flowunit dir: copy the `.mlpackage` into the build flowunit dir at install time; same idiom as `yolo_pose_detect/CMakeLists.txt`.

## Conversion script

`scripts/macos/convert_tracknet_coreml.py`. Run on `ssh pose`. Argparse:

- `--checkpoint` (default `/root/autodl-fs/repos/tracknet/exp_deep/TrackNet_best.pt`)
- `--variant` (`deep` | `small`, default `deep`)
- `--seq_len` (default 3)
- `--height 288 --width 512`
- `--out` (default `tracknet_deep.mlpackage`)
- `--compute_units` (default `ALL`)

Steps:

1. `git clone https://github.com/bcinno-dev/tracknet` to a tempdir and `sys.path.insert` it (no install) — `model.py` only needs torch. Avoids vendoring drift.
2. `model = TrackNetDeep(seq_len=3); model.load_state_dict(_unwrap(torch.load(ckpt, map_location='cpu')))`. Helper `_unwrap` accepts both raw `state_dict` and `{"model": state_dict, ...}` checkpoint shapes (project's `train.py` saves the latter).
3. `model.eval()`; example `torch.randn(1, 9, 288, 512)`; `traced = torch.jit.trace(model, ex)`.
4. `coremltools.convert(traced, inputs=[ct.TensorType(name='frames', shape=(1,9,288,512), dtype=np.float32)], outputs=[ct.TensorType(name='heatmaps', dtype=np.float32)], compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS14, compute_units=...)`.
5. Sanity check: 5 random fp32 inputs; assert per-pixel `|Δ| < 1e-2` between traced PyTorch and CoreML outputs. Fail loudly otherwise.
6. Save `.mlpackage`. Print byte size + sha256.

The script is self-contained — no ModelBox imports — so it can run on Linux+CUDA boxes (it doesn't need ANE).

## Graph TOML sketch

`src/demo/apple_silicon_yolo/graph/apple_silicon_tracknet.toml.in`:

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
```

Configured `apple_silicon_tracknet_configured.toml` mirrors the YOLO `configured/` pattern (concrete input/output paths). Generated at CMake configure time.

## Testing

- **Conversion sanity** (in `convert_tracknet_coreml.py`): `|Δ| < 1e-2` per-pixel between traced PyTorch and CoreML on N=5 random tensors.
- **Frame-stacker unit-ish test**: feed three fp32 frames with distinguishable values; assert output channel ordering and the padding behavior of the first two emissions.
- **`tracknet_post` unit-ish test**: synthetic Gaussian heatmap with peak at known `(cx, cy)`; assert centroid extraction within ±1 px after the 512×288 → source-res scale.
- **End-to-end smoke**: `scripts/macos/run-apple-silicon-tracknet.sh <input.mp4> <output.mp4>` on a short tennis clip. Pass: output MP4 exists, frame count == input frame count, at least one frame visibly has the ball circle drawn (manual check). Land under `test/function/apple_silicon_yolo/` parallel to existing smokes; not gated on CI (CI lacks Apple hardware).

## Open questions / non-goals

- **Source frame ↔ heatmap pairing.** Relies on ModelBox preserving stream order across the parallel `decoder → post` and `decoder → resize → … → post` paths. Existing demos depend on the same property. Verify on first run; fall back to attaching `frame_idx` on the buffer and explicit pair-by-key in `tracknet_post` if reordering shows up.
- **Normalize flowunit form.** TrackNet expects `/255.0` with no mean subtraction. If the existing `normalize` flowunit doesn't accept "no mean", drop it and bake the divide into `tracknet_frame_stacker` (cheap; the input copy is already the hot loop there).
- **First-2-frames padding.** Channel 2 (newest) is always the meaningful output, even when the window is padded — so no special-case at `tracknet_post`.
- **Multi-stream.** Stacker state is per-flowunit-instance. Multiple parallel input streams would need state keyed by session/stream id. Out of scope for v1.
- **No JSON sidecar, no batching > 1, no temporal smoothing across detections, no fall-back to YOLO ball detection.** Explicitly deferred.
