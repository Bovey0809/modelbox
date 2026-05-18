# TrackNet flyingball dataset

Frame-level dataset used to train the `tracknet_deep` model whose ONNX export
drives the tennis_action graph's ball-detection branch.

## Production weights

The `tracknet_deep_ort` flowunit loads `tracknet_deep.onnx` (187 MB, monolithic
— ORT can't follow external-data sidecars in the modelbox loader path). Current
production ONNX is the fine-tuned `exp16_finetune_2026-05-17` checkpoint
(epoch 9, val `f1@10=0.9379`).

| Where | Path |
| --- | --- |
| Source repo + experiments (pose) | `/autodl-fs/data/repos/tracknet/` |
| Fine-tune checkpoint (best) | `/autodl-fs/data/repos/tracknet/exp16_finetune_2026-05-17/TrackNet_best.pt` |
| Fine-tune ONNX (inlined) | `/autodl-fs/data/repos/tracknet/exp16_finetune_2026-05-17/tracknet_deep_inline.onnx` |
| Pre-fine-tune backup | `/root/autodl-tmp/modelbox_tennis/assets/tennis_action/tracknet_deep.onnx.pre-2026-05-17.bak` |
| Production canonical (live) | `/root/autodl-tmp/modelbox_tennis/assets/tennis_action/tracknet_deep.onnx` |
| Flowunit runtime location | `/usr/local/share/modelbox/demo/tennis_action/flowunit/tracknet_deep_ort/tracknet_deep.onnx` |

## Datasets (pose server)

| Version | Path | Frames | Notes |
| --- | --- | --- | --- |
| `flyingball_2026-03-25` | `/autodl-fs/data/datasets/westc/flyingball_2026-03-25/` | 52,692 | Baseline; ZQW project-3 `flyingball` tracks + project-19 Wimbledon. |
| `flyingball_2026-05-17` | `/autodl-fs/data/datasets/westc/flyingball_2026-05-17/` | 53,316 | Adds Kickball test clips (project-1) + project-3 `ball` tracks that overlap existing jpegs. **Used for current production weights.** |

Symlinked at `/root/autodl-fs/datasets/flyingball/` for the old path.

## Schema

`annotations.json` (top level):

```
dataset_name, created_at, label, dedup_rule, frame_alignment_note,
coordinate_note, sources{src -> meta}, stats{...}, images[]
```

Each `images[i]`:

```
image_path, source, video_name, frame_index, width, height,
video_frame_count, annotations[ {label, track_id, bbox_relative,
bbox_pixel, label_studio_frame_index, label_studio_time_sec,
floor_offset, ...} ]
```

`splits.json`: `{train: [video_name, ...], val: [...]}`.

## Frame alignment

`frame_index = label_studio_frame_index - floor_offset`. For all sources in
both versions `floor_offset=2`, verified by ffmpeg md5 comparison of
`-ss T` vs `select=eq(n,N)` extracts (see source README).

## Conversion tooling

Lives in the TrackNet repo on pose only (not tracked here):

- `/autodl-fs/data/repos/tracknet/tools/ls_to_flyingball.py` — generic
  label-studio → flyingball fragment converter; probes `floor_offset`
  per video via ffmpeg md5 if requested.
- `/autodl-fs/data/repos/tracknet/tools/project3_overlay.py` — project-3
  special case: source mp4s are not on this box, so `ball`-track bboxes
  are emitted only for `(video, frame_index)` pairs whose jpeg already
  exists (i.e. previously extracted for a `flyingball` track).
- `/autodl-fs/data/repos/tracknet/tools/build_dataset.py` — merges base
  + per-source fragments, symlinks shared jpeg trees from
  `flyingball_2026-03-25`, regenerates `splits.json` keeping prior
  train/val assignments.
- `/autodl-fs/data/repos/tracknet/tools/export_onnx.py` — exports a
  fine-tuned `tracknet_deep` checkpoint to the production `[batch, 9,
  288, 512] -> [batch, 3, 288, 512]` layout that
  `src/demo/tennis_action/graph/...` expects.

## Diff `2026-05-17` vs `2026-03-25`

- `Kickball_test_video/` (project_id=1, 10 LS tasks, 8 videos with
  usable frames, ~624 jpegs).
- `2025-03-20-ZQW-SplitVideo/`: +365 multi-ball bboxes overlaid onto
  209 existing jpegs (frames where `ball` and `flyingball` tracks
  coincide).
- ~12.3 k project-3 `ball`-track frames remain extractable only with the
  original ZQW mp4s, which are not on pose (only the previously
  extracted jpegs are). Tracked as the limiting blocker — fetch the
  rclone zip if a future pass needs them.
