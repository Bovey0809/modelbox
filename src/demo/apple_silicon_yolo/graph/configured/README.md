# Configured graph TOMLs (read-only reference)

These six files are **snapshots** of the per-task DAG TOMLs after CMake's
`configure_file` has substituted the `@VAR@` placeholders in the
`.toml.in` templates one directory up. They use the canonical install
prefix:

* `@DEMO_APPLE_SILICON_YOLO_FLOWUNIT_DIR@` → `/usr/local/share/modelbox/demo/apple_silicon_yolo/flowunit`
* `@DEMO_VIDEO_DIR@`                       → `/opt/modelbox/demo/video`

After `sudo make install`, the same files land at
`/usr/local/share/modelbox/demo/apple_silicon_yolo/graph/`. These copies
are checked in so the DAG topology is browsable directly on GitHub
without cloning + building.

The build system regenerates the live versions from the `.in` templates
each configure; **do not edit these snapshots by hand** — edit the
matching `.toml.in` template and rebuild.

## Files

| file | task | model | post-processor |
| --- | --- | --- | --- |
| `apple_silicon_yolo.toml`        | detection      | `yolov8n.mlpackage` (post-NMS [1, 300, 6])  | `yolo26_post`     |
| `apple_silicon_yolo_obb.toml`    | oriented bbox  | `yolov8n-obb.mlpackage` ([1, 20, 8400])     | `yolo_obb_post`   |
| `apple_silicon_yolo_pose.toml`   | pose / kpts    | `yolov8n-pose.mlpackage` ([1, 56, 8400])    | `yolo_pose_post`  |
| `apple_silicon_yolo_seg.toml`    | segmentation   | `yolov8n-seg.mlpackage` (2 outputs)         | `yolo_seg_post`   |
| `apple_silicon_yolo_cls.toml`    | classification | `yolov8n-cls.mlpackage` ([1, 1000])         | `yolo_cls_post`   |
| `apple_silicon_yolo_track.toml`  | det + tracking | `yolov8n.mlpackage` (reused, post-NMS)      | `yolo_track_post` |

All six graphs share the same upstream pipeline:

```
video_input → demuxer → decoder (cpu/ffmpeg/bgr)
             → resize (cpu/opencv-mb, 640x640 — or 224x224 for cls)
             → yolo_*_detect (apple_silicon / coreml / .mlpackage)
             → yolo_*_post (cpu/opencv-mb)
             → video_encoder (cpu/h264_videotoolbox/mp4)
```

`videodecoder:out_video_frame` forks: one branch goes through resize +
inference, the other feeds the original-resolution frame straight into
the post-processor so bounding boxes / overlays are drawn at full
resolution.
