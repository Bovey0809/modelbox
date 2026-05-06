# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ModelBox is a C++/Python AI application framework for pipelined inference across heterogeneous devices (CPU, CUDA/NVIDIA, Ascend, Rockchip RKNPU, Intel GPU, OpenVINO, MindSpore Lite, TensorFlow). Apps are described as graphs of "flowunits" connected by buffer streams; the engine schedules them in parallel with a thread pool.

Upstream docs: <http://modelbox-ai.com/modelbox-book/>.

## Build

The project does not build on macOS — it targets Ubuntu 18.04/20.04/22.04 and openEuler on x86_64/aarch64. Develop inside one of the `modelbox/modelbox-develop-*` Docker images (see `docker/README.md`); `docker/prepare_for_dev.sh` downloads the prebuilt binary deps the CMake `find_package`s expect.

Out-of-source builds are mandatory (the top-level `CMakeLists.txt` aborts otherwise):

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make package -j$(nproc)         # builds and packages
make build-test -j$(nproc)      # builds the gtest binary
make unittest                   # runs unit tests
```

CI invokes exactly this sequence (`.github/workflows/unit-test-pull-requests-on-device.yml`). On aarch64, CI sets `LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1` for `build-test` and unsets `LD_LIBRARY_PATH` before `make unittest`.

Notable CMake options (`CMake/Options.cmake`): `STANDALONE`, `WITH_WEBUI` (default ON), `WITH_JAVA`, `WITH_MINDSPORE`, `WITH_SECURE_C`, `TEST_COVERAGE`, `CLANG_TIDY`, `CLANG_TIDY_AS_ERROR`, `USE_CN_MIRROR`, `PYTHONE_DISABLED`. CI runs PRs across a matrix of `tensorrt`, `pytorch`, `tensorflow`, `ubuntu-d310p` self-hosted runners — a feature only compiles when the matching `find_package` succeeds, so missing CUDA/Ascend/etc. silently disables those drivers.

### Single test

The unit-test binary is one gtest executable produced by `build-test`. Filter with the standard gtest flag instead of rerunning `make unittest`:

```bash
./test/unittest --gtest_filter=YoloBoxFlowUnitTest.*
```

Test sources live under `test/{unit,function,drivers,manager,mock}` with assets in `test/assets`; `test/test_config.h.in` is configured into the build dir.

### Lint

`-DCLANG_TIDY=ON` enables clang-tidy during compilation; `-DCLANG_TIDY_AS_ERROR=ON` (used by CI) promotes warnings to errors. `.clang-tidy` enables `modernize-*`, `bugprone-*`, `concurrency-*`, `misc-*`, `readability-*`, `performance-*`, `portability-*`, `google-*` with selective opt-outs. `.clang-format` defines the C++ style.

## Architecture

The repo splits cleanly into the engine (`src/libmodelbox`), the runtime/server (`src/modelbox`), and pluggable drivers (`src/drivers`). Most day-to-day work happens in drivers.

### `src/libmodelbox` — engine + base library

- `base/` — primitives: `device/`, `drivers/` (driver loader/registry), `graph_manager/`, `mem/`, `thread_pool/`, `timer/`, `log/`, `status/`, `config/`. This is the shared substrate everything else links against.
- `engine/` — the pipeline runtime. `flow.cc`, `graph.cc`, `node.cc`, `port.cc`, `session.cc`, `data_context.cc`, `flowunit*.cc`, `match_stream.cc`, plus `scheduler/` and `dynamic_graph/`. A `Flow` parses a graph (TOML/Graphviz), constructs `Node`s wrapping `FlowUnit`s, and the scheduler pumps `Buffer`s along `Port`s.
- Public headers under `include/` are what flowunit authors include.

### `src/modelbox` — process-level glue

- `server/` — REST/HTTP control plane, `serving/` — model-serving entrypoints, `manager/` — process supervisor, `tool/` — `modelbox-tool` CLI, `common/` — shared helpers.

### `src/drivers` — the plugin layer (where most contributions land)

Drivers are loaded dynamically by the engine's driver manager.

- `drivers/devices/<backend>/` — one directory per hardware/runtime backend (`cpu`, `cuda`, `ascend`, `rockchip`, `intel_gpu`). Each has `core/` (device + memory adapter exposing the engine's device API) and `flowunit/` (per-backend flowunits — codecs, image ops, inference, sources/sinks). Adding a backend means implementing the device core and registering its flowunits.
- `drivers/inference_engine/<engine>/` — inference-runtime adapters (`tensorflow`, `mindspore`, `openvino`, `dlengine`). These are the model-loading flowunits referenced by the per-device flowunits via `find_package`.
- `drivers/common/flowunit/` — backend-agnostic flowunits (`image_process/`, `video_decode/`, `mean/`, `normalize/`, `image_rotate/`, `inference/`, `safe_http/`, `source_context/`, `hw_components/`, `driver_util/`).
- `drivers/common/{python,libs,devices}` — Python bindings (`modelbox_api`), shared helper libs (`fuse`, `file_requester`), and the cross-device `device_stream` utilities.
- `drivers/virtual/` — "virtual" drivers that wrap higher-level concepts (`inference`, `python`, `java`, `yolobox`) so users can declare them in graphs without writing C++.
- `drivers/graph_conf/graphviz/` — the Graphviz graph-config parser.

When adding a new flowunit: create a directory under the appropriate `flowunit/` (common vs. device-specific), add it to the parent `CMakeLists.txt`, and follow the pattern of an adjacent flowunit (e.g. the `yolo_*_post` flowunits under `src/drivers/devices/cpu/flowunit/` are good reference C++ post-processors). Tests for drivers live in `test/drivers/`.

### Other tree-level pieces

- `examples/` — runnable demo apps and graphs; built as part of the main CMake.
- `package/` — distro packaging (deb/rpm/tarball) integrated via CPack.
- `thirdparty/` — vendored or fetched deps wired in before `src/`.
- `docker/` — `Dockerfile.{cuda,ascend,rknnrt}.{develop,runtime}.{ubuntu,openeuler}` and the `prepare_for_*.sh` scripts that fetch prebuilt deps from the `modelbox-binary` GitHub release.

## Conventions

- C++ formatting via `.clang-format`; lint via `.clang-tidy`. Run a clang-tidy build before sending non-trivial C++ changes — CI rejects new clang-tidy warnings.
- Optional features must degrade gracefully when their `find_package` fails (the rest of the tree builds without TensorRT, Ascend, etc.).
- `docs/Design.md` and `docs/Goal.md` are the canonical internal design notes; consult them when changing engine semantics.
- `docs/macos-coreml.md` documents the Apple Silicon / Core ML port on the `feature/macos-coreml-integration` branch — what builds on Darwin, the Darwin OS adapter under `src/libmodelbox/base/arch/darwin/`, and the brew opencv install\_names workaround on macOS 26.
