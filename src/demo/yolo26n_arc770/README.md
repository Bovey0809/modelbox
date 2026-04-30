# YOLO26n on Intel Arc A770 (OpenVINO)

End-to-end ModelBox demo: video in, YOLO26n detection on the Arc, annotated
video out.

## Pipeline

```
video_input -> demuxer -> decoder -> resize(640) -> packed_planar_transpose
   -> normalize(/255) -> yolo26_detect (intel_gpu, openvino) -> yolo26_post
   -> video_encoder
```

`yolo26_detect` is a TOML-only flow unit that loads `yolo26n.onnx` via the
new OpenVINO inference engine and dispatches to OpenVINO's `GPU` plugin
(Level Zero -> Intel Arc).

## Host prerequisites

```bash
sudo apt install libze-dev intel-level-zero-gpu-dev openvino
pip install ultralytics
```

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

Standalone OpenVINO benchmark on the same model:

| device          | infer time |
|-----------------|------------|
| Arc A770 (GPU)  | **3.55 ms / iter** |
| i5-14600KF (CPU)| 10.24 ms / iter |

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
