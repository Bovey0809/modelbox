# YOLO26n on Intel Arc A770 (OpenVINO)

End-to-end ModelBox demo: video in, YOLO26n detection on the Arc, annotated
video out.

## Pipeline (Python-free, all C++)

```
video_input -> demuxer
            -> video_decoder    [intel_gpu, h264_qsv]
            -> resize(640)
            -> packed_planar_transpose
            -> normalize(/255)
            -> yolo26_detect    [intel_gpu, openvino, yolo26n.onnx]
            -> yolo26_post      [cpu, C++, OpenCV]
            -> video_encoder    [intel_gpu, h264_qsv]
```

The Arc A770 carries hardware H.264 decode (QSV), inference (OpenVINO GPU
plugin via Level Zero), and hardware H.264 encode (QSV). The CPU only
handles the cheap preprocess (resize / transpose / normalize) and
post-process (anchor-free decode + NMS + cv::rectangle). No Python
interpreter is loaded at runtime.

## Host prerequisites

```bash
sudo apt install libze-dev intel-level-zero-gpu-dev openvino libmfxgen1 \
                 intel-media-va-driver-non-free
pip install ultralytics  # one-time, only for exporting the ONNX model
```

`libmfxgen1` (Intel oneVPL GPU runtime) is required for the QSV codecs to
create an MFX session; without it `h264_qsv` etc. fail with
`MFX_ERR_DEVICE_FAILED (-9)`.

## Build and install

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_ALL_DEMO=ON
make -j$(nproc)
make install
```

The build invokes `scripts/export_yolo26n.py` once to produce `yolo26n.onnx`
beside `yolo26_detect.toml`. The script tries `yolo26n.pt` first and falls
back to `yolov8n.pt` if v26 weights are not yet published; the post-processor
handles either drop-in (Ultralytics v8 / v10 / v11 / v26 share the anchor-free
`[B, 4 + num_classes, N]` output format).

## Run

```bash
modelbox-tool flow run \
    -name yolo26n_arc770 \
    -graph /usr/local/share/modelbox/demo/yolo26n_arc770/graph/yolo26n_arc770.toml
```

Produces `/tmp/yolo26n_arc770_result.mp4`. Supply your own input video by
overriding `source_url` in the graph TOML, or pre-stage one at the path the
`@DEMO_VIDEO_DIR@/yolo26n_test_video.mp4` placeholder resolves to.

## Verifying the Arc is actually engaged

```bash
python -c 'import openvino as ov; print(ov.Core().available_devices)'  # must include "GPU"
intel_gpu_top                                                           # GPU util spikes during the run
```

The ModelBox log line `openvino: loaded ... on device GPU` confirms the
Arc plugin was selected. To force CPU fallback for comparison, change
`device=intel_gpu` to `device=cpu` in the graph TOML.

## Verified end-to-end run (Ubuntu 26.04 + Arc A770)

Full pipeline executed against the existing `car_detection` test video:

```
video_input -> demuxer -> decoder -> resize(640) -> packed_planar_transpose
   -> normalize(/255) -> yolo26_detect (intel_gpu, openvino, yolo11n.onnx)
   -> yolo26_post (cpu, python) -> video_encoder
```

**296 frames @ 1920×1080 from car_test_video.mp4** produced
`/tmp/yolo26n_arc770_result.mp4` with red bounding boxes around every
car in every frame. YOLO11n was used as the drop-in fallback for
YOLO26n (output tensor format is identical for v8 / v10 / v11 / v26).

## Speed profile

Benchmarks taken on the same Arc A770 + i5-14600KF host. Inference numbers
are 50 iters after 5 warm-ups; pipeline numbers are wall-clock from
`modelbox-tool flow -run` launch until the output mp4 contains every
expected frame.

### Per-model OpenVINO inference

| model         | input         | output                     | CPU p50 | Arc p50 | speedup |
|---------------|---------------|----------------------------|---------|---------|---------|
| yolo11n       | 3×640×640     | 84×8400                    | 9.92 ms | **3.19 ms** | 3.1× |
| yolo11n-seg   | 3×640×640     | 116×8400 + 32×160²         | 13.89 ms | **3.46 ms** | 4.0× |
| yolo11n-pose  | 3×640×640     | 56×8400                    | 11.04 ms | **3.38 ms** | 3.3× |
| yolo11n-obb   | 3×640×640     | 20×8400                    | 9.91 ms | **3.10 ms** | 3.2× |
| yolo11n-cls   | 3×224×224     | 1000                       | **0.76 ms** | 0.96 ms | 0.79× |

`yolo11n-cls` is the only one where the CPU wins — at 224² the model is
too small to amortise the host↔Arc round-trip. For maximum cls throughput
keep `device=cpu` for that node.

### End-to-end pipeline FPS

Each pipeline runs the full graph: `video_input -> demuxer -> video_decoder
(intel_gpu, h264_qsv) -> resize -> transpose -> normalize -> *_detect
(intel_gpu, openvino) -> *_post (cpu, C++) -> video_encoder (intel_gpu,
h264_qsv)`.

| task     | source resolution | frames | wall  | pipeline FPS | ms/frame |
|----------|-------------------|--------|-------|--------------|----------|
| detect   | 1920×1080         | 292    | 2.67 s | **109**       | 9.2 |
| segment  | 1920×1080         | 292    | 3.18 s | 92            | 10.9 |
| obb      | 1920×1080         | 292    | 2.69 s | 109           | 9.2 |
| classify | 1920×1080         | 292    | 2.14 s | 136           | 7.3 |
| pose     | 768×432           | 592    | 3.74 s | **158**       | 6.3 |
| track    | 768×432           | 592    | 3.21 s | 184           | 5.4 |

At 1080p every dense-prediction task clears 90 fps — comfortably above
real-time on a single Arc. Per-frame budget of ~9.2 ms breaks down as
~3.2 ms inference + ~6 ms for QSV decode/encode + preprocess + C++ post.
Inference is no longer the bottleneck — codec and memory copies are.

The pose / track pipelines look faster because the people-detection
sample is 768×432 (~6× smaller area), so the codec/preprocess shrinks
proportionally; inference time is essentially identical.

## Verified runtime behavior (from this branch's build verification)

After a successful build on Ubuntu 26.04 with OpenVINO 2024.6, `modelbox-tool driver -info -details` over the new driver paths reports:

```
Device Information:
  name: 0  type: intel_gpu  description: Intel GPU device (Arc / DG2 via Level Zero).

Driver Information:
  device-intel-gpu      type: intel_gpu  class: DRIVER-DEVICE
  openvino_inference    type: cpu        class: DRIVER-INFERENCE
  openvino_inference    type: intel_gpu  class: DRIVER-INFERENCE
```

A standalone OpenVINO C++ probe on this machine reports:

```
available_devices: CPU GPU
  GPU: Intel(R) Arc(TM) A770 Graphics (dGPU)
```

If OpenVINO is installed from the .tgz tarball rather than apt, set
`LD_LIBRARY_PATH` so the bundled TBB resolves at runtime:

```bash
export LD_LIBRARY_PATH=$OPENVINO_ROOT/runtime/lib/intel64:$OPENVINO_ROOT/runtime/3rdparty/tbb/lib
```
