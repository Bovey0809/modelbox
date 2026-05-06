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
