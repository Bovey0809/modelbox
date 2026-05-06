# Apple Silicon YOLO demo

YOLO object detection running on an Apple Silicon Mac (M-series) via Core ML.
This is the macOS counterpart to the `yolo26n_arc770` demo: same DAG shape,
same Ultralytics post-processor, swapped device + inference engine.

## Pipeline

```
video_input (cpu)
  -> video_demuxer (cpu)
  -> video_decoder (cpu, ffmpeg)
  -> resize / packed_planar_transpose / normalize (cpu)
  -> yolo_detect (apple_silicon, coreml, .mlpackage)
  -> yolo26_post (cpu, opencv draw)
  -> video_encoder (cpu, h264_videotoolbox)
```

The decoder/encoder stay on the cpu device. Apple's `h264_videotoolbox`
codec is built into homebrew ffmpeg and gives hardware-accelerated H.264
without needing a separate device driver — same as how the Arc770 path
uses `h264_qsv`.

## Build prerequisites

```bash
brew install cmake ffmpeg openssl@3 boost opencv pkg-config
pip install ultralytics coremltools
```

Build:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
         -DWITH_ALL_DEMO=ON \
         -DWITH_WEBUI=OFF \
         -DWITH_JAVA=OFF \
         -DPYTHONE_DISABLED=ON \
         -DOPENSSL_ROOT_DIR=$(brew --prefix openssl@3) \
         -DOpenCV_DIR=$(brew --prefix opencv)/lib/cmake/opencv4
make -j$(sysctl -n hw.ncpu)
sudo make install
```

The build runs `scripts/export_yolo_coreml.py` once and produces
`yolov8n.mlpackage` next to `yolo_detect.toml`.

## Run

```bash
modelbox-tool driver -info -details
# Look for: type: apple_silicon  /  coreml_inference  type: apple_silicon

modelbox-tool flow run -name apple_silicon_yolo \
  -graph /usr/local/share/modelbox/demo/apple_silicon_yolo/graph/apple_silicon_yolo.toml
ls -lh /tmp/apple_silicon_yolo_result.mp4
```

Smoke test:

```bash
python test/function/apple_silicon_yolo/test_apple_silicon_yolo.py \
  /usr/local/share/modelbox/demo/apple_silicon_yolo
```
