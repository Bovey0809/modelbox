# Car Detection App on Apple Silicon (ModelBox + CoreML)

**Branch:** `feature/macos-coreml-integration`
**Status:** approved design, ready for implementation plan
**Date:** 2026-05-11

## Goal

Ship an end-to-end "car detection" demo that runs entirely on Apple
Silicon (CPU + GPU + ANE via Core ML) using ModelBox flowunits. The app
reads an mp4, runs a YOLO detector at 640×640, filters detections to the
COCO **vehicle** classes (`car`, `motorcycle`, `bus`, `truck`), draws
boxes on the source-resolution frames, and writes an annotated mp4.

This is a thin variant of the existing `apple_silicon_yolo` detection
demo — we change *only* the class filter and the graph wiring, not the
device, decoder, encoder, resize, or Core ML inference flowunits.

## Non-goals

- No tracking, counting, zone-crossing, or speed estimation.
- No new device backend, no new inference flowunit, no new test binary.
- No changes to Linux/CUDA/Ascend paths.
- No retraining — uses the same Ultralytics-exported `.mlpackage` the
  existing det demo already builds (`yolo26n` / `yolov8n` head).

## Architecture

Pipeline (unchanged from `apple_silicon_yolo.toml.in` except for the
post-process flowunit's config):

```
video_input → videodemuxer → videodecoder (ffmpeg, bgr) →
image_resize (640×640) →
yolo_detect (apple_silicon, coreml, batch=1) →
yolo26_post (cpu, conf=0.35, class_allowlist="2,3,5,7") →
videoencoder (cpu, h264_videotoolbox, mp4)
```

Side branch: `videodecoder:out_video_frame` also feeds
`yolo26_post:in_image` so boxes are drawn on the original-resolution
frame (already wired in the existing det graph).

All flowunits except `yolo26_post` are reused verbatim from the
`apple_silicon_yolo` demo. Expected throughput: ~450 fps end-to-end on
M4 Pro (matches the measured numbers in `docs/macos-coreml.md`), so the
30 fps / 12.5 fps source videos run faster than realtime by ~15×.

## Components

### Modified: `src/drivers/devices/cpu/flowunit/yolo26_post/`

Add one optional config key `class_allowlist` to `Yolo26PostFlowUnit`:

- **Type:** comma-separated list of integers, e.g. `"2,3,5,7"`.
- **Default:** empty → keep all classes (preserves existing behavior
  for every consumer of this flowunit, including the five other
  `apple_silicon_yolo` task graphs).
- **Semantics:** after sigmoid+argmax class selection inside
  `Yolo26PostFlowUnit::Decode`, drop any detection whose `label` is
  not in the allowlist. Filter runs *before* NMS so NMS doesn't waste
  work on filtered-out classes.
- **Validation:** `Open()` parses the CSV; reject non-integer tokens
  or values outside `[0, num_classes_-1]` with
  `modelbox::STATUS_BADCONF`.
- **Storage:** `std::unordered_set<int> class_allowlist_` member.
  `Decode()` checks `class_allowlist_.empty() || class_allowlist_.count(label)`.

Header diff (in `yolo26_post_flowunit.h`):

```cpp
+ #include <unordered_set>
  ...
+ std::unordered_set<int> class_allowlist_;
```

Implementation diff (in `yolo26_post_flowunit.cc`):

- `Open`: parse `class_allowlist` from `opts->GetString("class_allowlist", "")`,
  split on `,`, `std::stoi` each token, range-check, insert into set.
- `Decode`: after computing `label`, `continue` if not in allowlist.

Estimated size: ~25 lines of code including parsing.

### New: `src/demo/apple_silicon_yolo/graph/apple_silicon_car_detection.toml.in`

Copy of `apple_silicon_yolo.toml.in` with two changes:

1. `video_input.source_url` → `@DEMO_VIDEO_DIR@/car-detection.mp4`
   (the bundled Intel sample, copied to `${MODELBOX_ROOT}/demo/video/`
   by the demo's existing CMake install rule).
2. `yolo26_post` line gains `class_allowlist="2,3,5,7"` and (optional)
   bumps `conf_threshold` to `0.4` to suppress junk far-off boxes.
3. `videoencoder.default_dest_url` →
   `/tmp/apple_silicon_car_detection_result.mp4`.

The CMake glue follows the same `configure_file(...)` pattern the
existing five task graphs already use. Install path mirrors
`apple_silicon_yolo.toml` so `modelbox-tool flow -run` can find it
under `/usr/local/share/modelbox/demo/apple_silicon_yolo/graph/`.

### New: `examples/car_detection/`

Out-of-tree convenience scaffold for the README/Quick-start. Layout:

```
examples/car_detection/
├── README.md                       # quick-start, both sample videos
├── run.sh                          # one-liner around modelbox-tool flow -run
└── videos/
    ├── car-detection.mp4           # 768×432 @ 12.5 fps, 30s, 2.7 MB
    └── person-bicycle-car-detection.mp4  # 768×432 @ 12 fps, 54s, 5.8 MB
```

Sample videos are from
[`intel-iot-devkit/sample-videos`](https://github.com/intel-iot-devkit/sample-videos)
(Apache-2.0). Already downloaded and committed.

The `examples/` tree is **not** built on macOS (per
`docs/macos-coreml.md`: "Skipped on macOS: … examples"), so this
directory is documentation + sample assets only — the runnable graph
lives under `src/demo/apple_silicon_yolo/`.

### New: smoke test (optional, in same PR if cheap)

`test/function/apple_silicon_yolo/test_apple_silicon_car_detection.py`
mirroring the existing `test_apple_silicon_yolo.py`:

- run the car-detection graph;
- assert output mp4 exists and `ffprobe` reports `nb_frames ≥ 0.95 ×`
  input frames (encoder may drop a tail frame);
- parse the per-frame stdout drawing log (if available) and assert at
  least one frame has a detection with `label ∈ {2,3,5,7}` and none
  with `label ∉ {2,3,5,7}`.

If the existing post-process flowunit doesn't expose detections in a
machine-readable form, skip the label-set assertion — the visual
inspection in the README covers it.

## Data flow

1. **video_input** emits a URL.
2. **videodemuxer** (cpu, ffmpeg) splits h264 packets.
3. **videodecoder** (cpu, ffmpeg) decodes to BGR uint8 frames. Feeds
   *both* the resize branch (for inference) and the post branch (for
   drawing on the original frame).
4. **image_resize** (cpu, opencv) scales to 640×640 BGR.
5. **yolo_detect** (apple_silicon, coreml) runs the `.mlpackage`. Core
   ML uses `MLComputeUnitsAll` so the prediction dispatches across CPU
   + GPU + ANE.
6. **yolo26_post** (cpu) decodes anchor-free head, applies the
   vehicle-class allowlist, runs NMS, draws boxes on the original BGR
   frame, emits the annotated frame.
7. **videoencoder** (cpu, h264_videotoolbox) re-encodes to mp4 on disk.

## Error handling

Inherits the existing det demo's error semantics. The only new failure
mode is in `Yolo26PostFlowUnit::Open`:

- Malformed `class_allowlist` (non-integer, out-of-range, empty after
  trimming a stray comma) → `STATUS_BADCONF` with a message naming
  the bad token. Graph build fails fast; nothing downstream runs.

## Testing

| Layer        | Test                                                                  |
| ------------ | --------------------------------------------------------------------- |
| Build        | `make -j` on macOS picks up the new graph TOML and modified flowunit. |
| Graph parse  | `modelbox-tool flow -info <graph>` exits 0.                           |
| End-to-end   | `modelbox-tool flow -run` on both sample videos produces a non-empty mp4 with frame count ≈ input. |
| Visual       | Spot-check 2–3 frames per video: vehicles boxed, persons/bikes/lights *not* boxed. |
| Regression   | Existing five `apple_silicon_yolo` task graphs still pass `test_apple_silicon_yolo.py` (unchanged config → unchanged behavior). |

The existing CI runs Linux-only; on-Mac validation is manual per
`docs/macos-coreml.md` (no macOS CI).

## Risks / open questions

1. **Filter-before-NMS interaction.** YOLO26's argmax happens per
   anchor over all 80 classes; if a near-tie between `car` and
   `truck` swaps after filtering, NMS could keep two boxes on the
   same vehicle. Mitigation: keep all four vehicle classes in the
   allowlist (the user-confirmed scope) so the swap stays within the
   allowed set. If false-double-boxing shows up in practice, the fix
   is to run class-agnostic NMS — out of scope for this spec.
2. **`yolo26n.mlpackage` already has all 80 COCO classes.** No
   retraining or model swap needed. The allowlist is purely a runtime
   filter on inference output.
3. **Bundled video is side-view, not aerial.** The original spec
   draft pointed at Intel's `car-detection.mp4` (top-down aerial
   parking-lot footage), but YOLOv8n COCO is trained on street-level
   side-view imagery and emits near-zero confidence on aerial vehicle
   shots (verified empirically: best top-down car detection score was
   0.016). The committed bundle is the side-view `highway.mp4` clip
   from `MikhailTodes/traffic_counter`, which yields 0.85–0.92
   confidence scores on visible cars and trucks.

## Out of scope (explicit non-deliverables)

- Reorganizing `yolo26_post` into a "class filter" subflowunit.
- Exposing the allowlist through any other graph's config in this PR.
- Adding macOS CI.
- Any change to `feature/paddle-ocr-flowunits` (current working
  branch) — work lands on `feature/macos-coreml-integration`.
