# macOS / Apple Silicon support

ModelBox builds and runs on Apple Silicon Macs (M-series) with a new
`apple_silicon` device backed by Core ML. The integration mirrors the
Intel Arc A770 + OpenVINO path so existing demos and post-processors
port over without code changes.

## What works on Darwin

| Component                                | Status     | Notes                                       |
| ---------------------------------------- | ---------- | ------------------------------------------- |
| `libmodelbox-base` / `libmodelbox`       | builds     | Darwin OS adapter under `arch/darwin/`      |
| `modelbox-tool` (CLI)                    | builds     | server + manager are Linux-only             |
| `device-cpu` + cpu flowunits             | builds     | uses `h264_videotoolbox` for hw video I/O   |
| `device-apple-silicon`                   | new        | one device id; CPU + GPU + ANE via Core ML  |
| `inference_engine/coreml/`               | new        | `.mlpackage` / `.mlmodel` loader            |
| `coreml_inference` flowunit              | new        | `device=apple_silicon`, `virtual_type=coreml` |
| `apple_silicon_yolo` demo + smoke test   | new        | mirrors `yolo26n_arc770`                    |
| Linux/aarch64 + Linux/x86_64 builds      | unaffected | every change gated by `__APPLE__` / `APPLE` |

Skipped on macOS: `manager` daemon (signalfd / capabilities), `server`
+ `serving` (bake in `SIGPWR`/`SIGSTKFLT`), `develop`, `python`, `java`,
`examples` (Linux-only setup scripts), `package` (deb/rpm).

## Quick start

```bash
brew install cmake ffmpeg openssl@3 boost opencv graphviz pkg-config
python3 -m venv ~/yolovenv
~/yolovenv/bin/pip install ultralytics coremltools

git checkout feature/macos-coreml-integration
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
         -DWITH_ALL_DEMO=ON -DWITH_WEBUI=OFF -DWITH_JAVA=OFF \
         -DPYTHONE_DISABLED=ON \
         -DOPENSSL_ROOT_DIR=$(brew --prefix openssl@3) \
         -DOpenCV_DIR=$(brew --prefix opencv)/lib/cmake/opencv4 \
         -DPYTHON_EXECUTABLE=$HOME/yolovenv/bin/python
make -j$(sysctl -n hw.ncpu)
sudo make install
```

## Verify

```bash
modelbox-tool driver -info
# Expect at least:
#   DRIVER-DEVICE     device-apple-silicon  apple_silicon  ...
#   DRIVER-INFERENCE  coreml_inference      apple_silicon  ...
#   DRIVER-VIRTUAL    inference (libmodelbox-engine-coreml)

cp /opt/modelbox/demo/video/yolo26n_test_video.mp4 /tmp/   # or any 640x640 test mp4
modelbox-tool flow -run \
  /usr/local/share/modelbox/demo/apple_silicon_yolo/graph/apple_silicon_yolo.toml
ls -lh /tmp/apple_silicon_yolo_result.mp4
```

Smoke test:

```bash
python test/function/apple_silicon_yolo/test_apple_silicon_yolo.py \
  /usr/local/share/modelbox/demo/apple_silicon_yolo
```

## Performance

Measured on an M4 Pro MacBook Pro (Darwin 26.4.1, 14-core CPU, integrated
GPU + ANE, 24 GB unified memory) running the demo graph against
`/Users/houbowei/Downloads/Feishu20260424-154315.mp4` (1280x720 h264,
3137 frames, 104.6 s, 12 MB).

```
TOTAL flow wall time:    7.78 s for 3137 frames  =  403 fps end-to-end

flowunit                              calls  frames  cpu_ms  ms/frm  span_s
───────────────────────────────────────────────────────────────────────────
yolo_detect (apple_silicon, coreml)      99    3137  7674.7   2.45    7.69
videoencoder (cpu, h264_videotoolbox)    99    3137  4613.1   1.47    7.50
videodecoder (cpu, ffmpeg)              100    3138  3931.6   1.25    7.48
image_resize (cpu, opencv-mb)            99    3137   657.2   0.21    7.52
yolo_post    (cpu, opencv-mb draw)       99    3137   507.6   0.16    7.62
videodemuxer (cpu, ffmpeg)             3138    3138   254.5   0.08    7.29
video_input  (cpu)                        1       1     0.0   0.05    0.00
───────────────────────────────────────────────────────────────────────────
sum CPU time                                   3138 17638.8 ms
parallelism (sum CPU / wall):  2.27x
```

* **Critical path is inference.** `yolo_detect` occupies a 7.69 s span out
  of the 7.78 s total wall window — every other stage finishes inside the
  inference window. The next-longest stage is videoencoder at 1.47 ms/frame.
* **CoreML 2.45 ms/frame vs Ultralytics' 6.2 ms/frame** on the same
  `.mlpackage`: modelbox dispatches `Process()` calls into the device's
  thread pool so multiple frames are in flight, and `MLComputeUnitsAll`
  spreads them across CPU + GPU + ANE simultaneously. Ultralytics' Python
  predict loop is strictly sequential.
* **Sequential single-stream latency** is ~5.5 ms/frame
  (decode 1.25 + resize 0.21 + infer 2.45 + post 0.16 + encode 1.47).
  Pipelining yields 403 fps measured = 2.27x the sequential 180 fps;
  inference is the bottleneck, decode/encode are well under 2 ms/frame.

### Reproduce

Add a `[profile]` block to the graph TOML and re-run:

```toml
[profile]
profile = true
trace = true
session = true
dir = "/tmp/mb_profile"
```

```bash
PROFILE_PATH=/tmp/mb_profile \
  modelbox-tool flow -run /usr/local/share/modelbox/demo/apple_silicon_yolo/graph/apple_silicon_yolo.toml
```

This emits `/tmp/mb_profile/trace_<ts>.json` (Chrome-tracing format).
Drop the file on `chrome://tracing` (or `https://ui.perfetto.dev`) for
an interactive flame view. The aggregate table above comes from a small
Python script that parses the JSON and groups events by `tid` (flowunit)
— the modelbox `Profiler` class records one `TraceSlice` per `Process()`
call, batched by `FlowUnitGroup::StartTrace` (so each trace event's
`args.batch_size` is the number of `Process()` calls aggregated, not a
within-call batch).

### Async vs sync

* **Pipeline is async.** Modelbox's `DeviceExecute()` dispatches each
  flowunit's `Process()` call onto the device's thread pool. NORMAL
  flowunits (`image_resize`, `yolo_detect`, `yolo26_post`) scale wider
  because each call is independent; STREAM flowunits (`video_demuxer`,
  `video_decoder`, `video_encoder`) serialize per-stream because they
  hold codec context across calls. Overlapping spans in the trace —
  yolo_detect's 7.69 s overlapping with videoencoder's 7.50 s,
  videodecoder's 7.48 s, etc. — are realized parallelism.
* **Per-call is sync.** The CoreML flowunit's `Infer()` blocks on
  `[model predictionFromFeatures:provider error:&err]`; Core ML
  pipelines that single inference internally across CPU/GPU/ANE but
  doesn't return until the prediction is complete. `MLComputeUnitsAll`
  is set in `coreml_inference.mm`'s Open path.

## Architecture notes

* **Memory model**: Apple Silicon's unified memory means CPU, GPU, and
  ANE share one physical pool. The first-pass `apple_silicon` memory
  manager `malloc`s host buffers; Core ML routes them to the chosen
  compute unit at `predictionFromFeatures:` time. A future iteration
  can switch to `IOSurface`-backed buffers or Core ML's
  `MLMultiArray.dataPointer`-aware allocator for true zero-copy.
* **Compute unit**: `MLComputeUnitsAll` lets Core ML pick CPU/GPU/ANE
  per layer. To force CPU-only or ANE-only, pass `compute_units` in the
  flowunit's `[base]` block (Core ML accepts the standard
  `MLComputeUnits*` enum strings).
* **Driver scanner**: macOS's Objective-C runtime aborts on
  `fork()`-without-`exec()` when Cocoa frameworks have run any
  `+initialize` in the parent. The scanner runs in-process on Darwin.
* **Cross-platform shims**: see `src/libmodelbox/base/arch/darwin/`
  (host_statistics64 / sysctlbyname / pthread_setname_np /
  getifaddrs+AF_LINK) and the small set of `#ifdef __APPLE__` blocks
  in `src/modelbox/common/utils.cc`.

## Known caveats

* **brew opencv install\_names on macOS 26 (Tahoe)**: brew's opencv
  4.13 bottle ships with unsubstituted `@@HOMEBREW_PREFIX@@` placeholders
  in dylib install names. Cv-dependent flowunits (`resize`,
  `packed_planar_transpose`, `yolo26_post`, …) fail to dlopen at
  runtime. Workaround until the brew formula is fixed:

  ```bash
  for f in /opt/homebrew/Cellar/opencv/*/lib/*.dylib; do
      otool -L "$f" | awk '/@@HOMEBREW_PREFIX@@/{print $1}' | while read ref; do
          new=$(echo "$ref" | sed 's|@@HOMEBREW_PREFIX@@|/opt/homebrew|g')
          install_name_tool -change "$ref" "$new" "$f"
      done
      codesign --force --sign - "$f"
  done
  ```

  May require a single re-link of the modelbox dylibs after the patch.
  The `apple_silicon` device + Core ML engine + flowunit themselves are
  unaffected — they don't link OpenCV.

* **No `sudo`-free install**: the demo graph references
  `${MODELBOX_ROOT}/...` install paths. Run `sudo make install` or
  override paths in the graph TOML for a user-local layout.
