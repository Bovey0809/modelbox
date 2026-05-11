# Car Detection on Apple Silicon — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a car-detection demo to the `apple_silicon_yolo` family that runs YOLOv8n via Core ML and filters detections to COCO vehicle classes (car/motorcycle/bus/truck).

**Architecture:** One new graph TOML reusing the existing `video_demuxer → video_decoder → resize → yolo_detect (coreml) → yolo26_post → video_encoder` pipeline; one tiny config knob (`class_allowlist`) added to `yolo26_post`; one bundled sample mp4 installed alongside the graph.

**Tech Stack:** C++17, OpenCV, modelbox flowunit API, Core ML (via existing `coreml_inference` engine), CMake `configure_file`, Python `unittest` for the smoke test. All work on branch `feature/macos-coreml-integration`.

**Spec:** `docs/superpowers/specs/2026-05-11-car-detection-apple-silicon-design.md`

**Parallelizable structure:** Tasks 2 and 3 are independent (flowunit C++ vs graph + video assets). They can be dispatched to two subagents in parallel. Task 4 (smoke test) must wait until both 2 and 3 are merged. Task 5 (docs) waits on 4.

---

## File Structure

| File | Action | Owner |
| --- | --- | --- |
| `src/drivers/devices/cpu/flowunit/yolo26_post/yolo26_post_flowunit.h` | modify | Task 2 |
| `src/drivers/devices/cpu/flowunit/yolo26_post/yolo26_post_flowunit.cc` | modify | Task 2 |
| `src/demo/apple_silicon_yolo/graph/apple_silicon_car_detection.toml.in` | create | Task 3 |
| `src/demo/apple_silicon_yolo/graph/CMakeLists.txt` | modify | Task 3 |
| `src/demo/apple_silicon_yolo/graph/car-detection.mp4` | create (binary asset) | Task 3 |
| `src/demo/apple_silicon_yolo/graph/configured/apple_silicon_car_detection.toml` | create (snapshot) | Task 3 |
| `test/function/apple_silicon_yolo/test_apple_silicon_car_detection.py` | create | Task 4 |
| `src/demo/apple_silicon_yolo/README.md` | modify | Task 5 |
| `docs/macos-coreml.md` | modify | Task 5 |

---

## Task 0: Prep workspace and switch to the CoreML branch

**Files:** none (git-only)

- [ ] **Step 1: Stash or discard the throwaway `examples/car_detection/` scratch dir on the current branch**

The user's earlier exploration created `examples/car_detection/videos/{car-detection,person-bicycle-car-detection}.mp4` on `feature/paddle-ocr-flowunits`. It is not part of the plan; remove it before switching.

Run:
```bash
cd /Users/houbowei/modelbox
git status --short
rm -rf examples/car_detection
git status --short  # confirm clean working tree (no staged changes besides committed spec)
```
Expected: nothing under `examples/car_detection/` left; `git status` shows no untracked files.

- [ ] **Step 2: Switch to the CoreML feature branch**

Run:
```bash
git checkout feature/macos-coreml-integration
git status --short
git log --oneline -3
```
Expected: branch switched cleanly, HEAD at `e60e533` or later. The previously-committed spec from `feature/paddle-ocr-flowunits` will NOT be visible here — that's fine, we'll cherry-pick it next.

- [ ] **Step 3: Cherry-pick the spec doc onto this branch**

Run:
```bash
git cherry-pick feature/paddle-ocr-flowunits -- docs/superpowers/specs/2026-05-11-car-detection-apple-silicon-design.md 2>/dev/null || \
git show feature/paddle-ocr-flowunits:docs/superpowers/specs/2026-05-11-car-detection-apple-silicon-design.md > docs/superpowers/specs/2026-05-11-car-detection-apple-silicon-design.md && \
  git add docs/superpowers/specs/2026-05-11-car-detection-apple-silicon-design.md && \
  git commit -m "docs: car detection on Apple Silicon (CoreML) design spec"
```
Expected: spec doc now exists at `docs/superpowers/specs/2026-05-11-car-detection-apple-silicon-design.md` on this branch with one new commit.

- [ ] **Step 4: Verify the apple_silicon_yolo demo tree is in place**

Run:
```bash
ls src/demo/apple_silicon_yolo/graph/
test -f src/drivers/devices/cpu/flowunit/yolo26_post/yolo26_post_flowunit.cc && echo "yolo26_post present"
```
Expected: `apple_silicon_yolo*.toml.in` files visible; `yolo26_post_flowunit.cc` reported present.

---

## Task 2: Add `class_allowlist` config to `yolo26_post`

**Files:**
- Modify: `src/drivers/devices/cpu/flowunit/yolo26_post/yolo26_post_flowunit.h`
- Modify: `src/drivers/devices/cpu/flowunit/yolo26_post/yolo26_post_flowunit.cc`

Independent of Task 3. Can be implemented in parallel.

- [ ] **Step 1: Update the header — add include and member**

Edit `src/drivers/devices/cpu/flowunit/yolo26_post/yolo26_post_flowunit.h`. Add `#include <unordered_set>` near the other includes (after `<opencv2/opencv.hpp>`), and add a new private member at the end of the private section, immediately after `float iou_threshold_{0.45F};`:

```cpp
  std::unordered_set<int> class_allowlist_;
```

- [ ] **Step 2: Parse `class_allowlist` in `Open()`**

Edit `src/drivers/devices/cpu/flowunit/yolo26_post/yolo26_post_flowunit.cc`. Replace the body of `Yolo26PostFlowUnit::Open()` with:

```cpp
modelbox::Status Yolo26PostFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  net_h_ = opts->GetInt32("net_h", 640);
  net_w_ = opts->GetInt32("net_w", 640);
  num_classes_ = opts->GetInt32("num_classes", 80);
  conf_threshold_ = opts->GetFloat("conf_threshold", 0.25F);
  iou_threshold_ = opts->GetFloat("iou_threshold", 0.45F);

  std::string allow = opts->GetString("class_allowlist", "");
  if (!allow.empty()) {
    std::stringstream ss(allow);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      // trim
      size_t a = tok.find_first_not_of(" \t");
      size_t b = tok.find_last_not_of(" \t");
      if (a == std::string::npos) continue;
      tok = tok.substr(a, b - a + 1);
      if (tok.empty()) continue;
      int v = 0;
      try {
        v = std::stoi(tok);
      } catch (const std::exception &) {
        return {modelbox::STATUS_BADCONF,
                "yolo26_post: class_allowlist token is not an int: " + tok};
      }
      if (v < 0 || v >= num_classes_) {
        return {modelbox::STATUS_BADCONF,
                "yolo26_post: class_allowlist value out of range [0, " +
                    std::to_string(num_classes_ - 1) + "]: " + tok};
      }
      class_allowlist_.insert(v);
    }
  }

  MBLOG_INFO << "yolo26_post (cpp): net=" << net_w_ << "x" << net_h_
             << " num_classes=" << num_classes_
             << " conf=" << conf_threshold_ << " iou=" << iou_threshold_
             << " class_allowlist=[" << allow << "]";
  return modelbox::STATUS_OK;
}
```

Also add at the top of the file (with the other `#include`s):

```cpp
#include <sstream>
#include <stdexcept>
#include <string>
```

- [ ] **Step 3: Apply the filter inside `Decode()`**

Edit `src/drivers/devices/cpu/flowunit/yolo26_post/yolo26_post_flowunit.cc`. In `Yolo26PostFlowUnit::Decode()`, immediately after the `if (best_score < conf_threshold_) { continue; }` line, add:

```cpp
    if (!class_allowlist_.empty() && class_allowlist_.count(best_label) == 0) {
      continue;
    }
```

This filters before NMS so NMS doesn't waste cycles on rejected classes.

- [ ] **Step 4: Build the flowunit**

Run from the repo root (assumes a prior CMake configure exists — if not, follow `docs/macos-coreml.md` quick-start first):

```bash
cd build
make -j$(sysctl -n hw.ncpu) modelbox-flowunit-yolo26-post 2>&1 | tail -30
```
Expected: target builds with no warnings, ends in `[100%] Built target modelbox-flowunit-yolo26-post`. If the target name differs, `cmake --build . --target help | grep -i yolo26` reveals it.

- [ ] **Step 5: Sanity-check existing det demo still works (regression guard)**

Install and run the unchanged `apple_silicon_yolo.toml` to confirm the empty-allowlist code path matches old behavior:

```bash
sudo make install
modelbox-tool flow -run \
  /usr/local/share/modelbox/demo/apple_silicon_yolo/graph/apple_silicon_yolo.toml
ls -lh /tmp/apple_silicon_yolo_result.mp4
```
Expected: the existing det demo produces a non-empty `/tmp/apple_silicon_yolo_result.mp4`. Open it; it should look identical to the pre-change run (all 80 classes drawn).

- [ ] **Step 6: Commit**

```bash
git add src/drivers/devices/cpu/flowunit/yolo26_post/yolo26_post_flowunit.h \
        src/drivers/devices/cpu/flowunit/yolo26_post/yolo26_post_flowunit.cc
git commit -m "flowunit: yolo26_post adds optional class_allowlist filter (pre-NMS)"
```

---

## Task 3: Bundle sample video + new graph TOML + CMake wiring

**Files:**
- Create: `src/demo/apple_silicon_yolo/graph/car-detection.mp4` (binary, 2.7 MB)
- Create: `src/demo/apple_silicon_yolo/graph/apple_silicon_car_detection.toml.in`
- Modify: `src/demo/apple_silicon_yolo/graph/CMakeLists.txt`
- Create: `src/demo/apple_silicon_yolo/graph/configured/apple_silicon_car_detection.toml`

Independent of Task 2. Can be implemented in parallel.

- [ ] **Step 1: Download the bundled sample video**

The Intel sample-videos repo ships a 768×432 / 12.5 fps / 30 s clip purpose-built for car detection. License: Apache-2.0.

```bash
cd /Users/houbowei/modelbox
curl -sL -o src/demo/apple_silicon_yolo/graph/car-detection.mp4 \
  https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4
ls -l src/demo/apple_silicon_yolo/graph/car-detection.mp4
file src/demo/apple_silicon_yolo/graph/car-detection.mp4
```
Expected: 2.7 MB, `ISO Media, MP4 v2`.

- [ ] **Step 2: Create the graph template**

Write `src/demo/apple_silicon_yolo/graph/apple_silicon_car_detection.toml.in`:

```toml
[driver]
dir = [
    "@DEMO_APPLE_SILICON_YOLO_FLOWUNIT_DIR@"
]
[flow]
desc = "Car detection on Apple Silicon via Core ML — vehicles-only filter"
[graph]
format = "graphviz"
graphconf = """digraph apple_silicon_car_detection {
    node [shape=Mrecord]
    video_input[type=flowunit, flowunit=video_input, device=cpu, deviceid=0, source_url="@DEMO_APPLE_SILICON_YOLO_GRAPH_DIR@/car-detection.mp4"]
    videodemuxer[type=flowunit, flowunit=video_demuxer, device=cpu, deviceid=0]
    videodecoder[type=flowunit, flowunit=video_decoder, device=cpu, deviceid=0, pix_fmt=bgr]
    image_resize[type=flowunit, flowunit=resize, device=cpu, deviceid=0, image_width=640, image_height=640]
    yolo_detect[type=flowunit, flowunit=yolo_detect, device=apple_silicon, deviceid=0, batch_size=1]
    yolo_post[type=flowunit, flowunit=yolo26_post, device=cpu, deviceid=0, conf_threshold=0.4, class_allowlist="2,3,5,7"]
    videoencoder[type=flowunit, flowunit=video_encoder, device=cpu, deviceid=0, encoder=h264_videotoolbox, format=mp4, default_dest_url="/tmp/apple_silicon_car_detection_result.mp4"]

    video_input:out_video_url -> videodemuxer:in_video_url
    videodemuxer:out_video_packet -> videodecoder:in_video_packet
    videodecoder:out_video_frame -> image_resize:in_image
    image_resize:out_image -> yolo_detect:input
    yolo_detect:output -> yolo_post:in_feat
    videodecoder:out_video_frame -> yolo_post:in_image
    yolo_post:out_data -> videoencoder:in_video_frame
}"""
```

The four allowlist class ids correspond to COCO `car=2, motorcycle=3, bus=5, truck=7`.

- [ ] **Step 3: Wire the new graph into CMake**

Edit `src/demo/apple_silicon_yolo/graph/CMakeLists.txt`. In the existing `foreach(GRAPH_BASE ...)` loop, append `apple_silicon_car_detection` to the list:

```cmake
foreach(GRAPH_BASE
        apple_silicon_yolo
        apple_silicon_yolo_obb
        apple_silicon_yolo_pose
        apple_silicon_yolo_seg
        apple_silicon_yolo_cls
        apple_silicon_yolo_track
        apple_silicon_car_detection)
```

Then immediately after the existing `foreach` block, add a separate install rule for the bundled video so it lands next to the graph at install time:

```cmake
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/car-detection.mp4
    DESTINATION ${DEMO_APPLE_SILICON_YOLO_GRAPH_DIR}
    COMPONENT demo
)
```

- [ ] **Step 4: Generate and commit the read-only configured snapshot**

The repo convention (see `src/demo/apple_silicon_yolo/graph/configured/README.md`) is to check in CMake-configured copies with `@VAR@` already substituted to the canonical install prefix. Create `src/demo/apple_silicon_yolo/graph/configured/apple_silicon_car_detection.toml` by hand, mirroring the substitutions:

* `@DEMO_APPLE_SILICON_YOLO_FLOWUNIT_DIR@` → `/usr/local/share/modelbox/demo/apple_silicon_yolo/flowunit`
* `@DEMO_APPLE_SILICON_YOLO_GRAPH_DIR@`    → `/usr/local/share/modelbox/demo/apple_silicon_yolo/graph`

Content:

```toml
[driver]
dir = [
    "/usr/local/share/modelbox/demo/apple_silicon_yolo/flowunit"
]
[flow]
desc = "Car detection on Apple Silicon via Core ML — vehicles-only filter"
[graph]
format = "graphviz"
graphconf = """digraph apple_silicon_car_detection {
    node [shape=Mrecord]
    video_input[type=flowunit, flowunit=video_input, device=cpu, deviceid=0, source_url="/usr/local/share/modelbox/demo/apple_silicon_yolo/graph/car-detection.mp4"]
    videodemuxer[type=flowunit, flowunit=video_demuxer, device=cpu, deviceid=0]
    videodecoder[type=flowunit, flowunit=video_decoder, device=cpu, deviceid=0, pix_fmt=bgr]
    image_resize[type=flowunit, flowunit=resize, device=cpu, deviceid=0, image_width=640, image_height=640]
    yolo_detect[type=flowunit, flowunit=yolo_detect, device=apple_silicon, deviceid=0, batch_size=1]
    yolo_post[type=flowunit, flowunit=yolo26_post, device=cpu, deviceid=0, conf_threshold=0.4, class_allowlist="2,3,5,7"]
    videoencoder[type=flowunit, flowunit=video_encoder, device=cpu, deviceid=0, encoder=h264_videotoolbox, format=mp4, default_dest_url="/tmp/apple_silicon_car_detection_result.mp4"]

    video_input:out_video_url -> videodemuxer:in_video_url
    videodemuxer:out_video_packet -> videodecoder:in_video_packet
    videodecoder:out_video_frame -> image_resize:in_image
    image_resize:out_image -> yolo_detect:input
    yolo_detect:output -> yolo_post:in_feat
    videodecoder:out_video_frame -> yolo_post:in_image
    yolo_post:out_data -> videoencoder:in_video_frame
}"""
```

- [ ] **Step 5: Configure CMake to verify TOML generation**

```bash
cd build
cmake . 2>&1 | tail -20
ls -l src/demo/apple_silicon_yolo/graph/apple_silicon_car_detection.toml
diff src/demo/apple_silicon_yolo/graph/apple_silicon_car_detection.toml \
     ../src/demo/apple_silicon_yolo/graph/configured/apple_silicon_car_detection.toml
```
Expected: `cmake .` succeeds, the generated TOML exists, and `diff` reports no differences (the configured snapshot matches the substituted template exactly).

- [ ] **Step 6: Commit**

```bash
git add src/demo/apple_silicon_yolo/graph/car-detection.mp4 \
        src/demo/apple_silicon_yolo/graph/apple_silicon_car_detection.toml.in \
        src/demo/apple_silicon_yolo/graph/CMakeLists.txt \
        src/demo/apple_silicon_yolo/graph/configured/apple_silicon_car_detection.toml
git commit -m "demo: apple_silicon_car_detection graph + bundled sample mp4"
```

---

## Task 4: End-to-end smoke test

**Files:**
- Create: `test/function/apple_silicon_yolo/test_apple_silicon_car_detection.py`

Depends on Task 2 (`class_allowlist` parser) and Task 3 (graph installed).

- [ ] **Step 1: Write the failing test**

Create `test/function/apple_silicon_yolo/test_apple_silicon_car_detection.py`:

```python
#!/usr/bin/env python3
#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""End-to-end smoke test for the apple_silicon_car_detection demo.

Skipped automatically when not on macOS, when the demo wasn't installed,
or when modelbox-tool isn't on PATH. Run manually:

    python test_apple_silicon_car_detection.py \
        /usr/local/share/modelbox/demo/apple_silicon_yolo
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import unittest


def _have(cmd):
    return shutil.which(cmd) is not None


class AppleSiliconCarDetectionSmoke(unittest.TestCase):
    DEMO_DIR = os.environ.get(
        "APPLE_SILICON_YOLO_DEMO_DIR",
        "/usr/local/share/modelbox/demo/apple_silicon_yolo",
    )
    GRAPH = os.path.join(DEMO_DIR, "graph", "apple_silicon_car_detection.toml")
    OUTPUT_MP4 = "/tmp/apple_silicon_car_detection_result.mp4"

    @classmethod
    def setUpClass(cls):
        if platform.system() != "Darwin":
            raise unittest.SkipTest("apple_silicon path requires macOS")
        if not _have("modelbox-tool"):
            raise unittest.SkipTest("modelbox-tool not on PATH")
        if not os.path.isfile(cls.GRAPH):
            raise unittest.SkipTest(f"graph not installed: {cls.GRAPH}")

    def test_runs_and_produces_mp4(self):
        if os.path.exists(self.OUTPUT_MP4):
            os.remove(self.OUTPUT_MP4)
        rc = subprocess.call(["modelbox-tool", "flow", "-run", self.GRAPH])
        self.assertEqual(rc, 0, "modelbox-tool flow -run exited non-zero")
        self.assertTrue(os.path.isfile(self.OUTPUT_MP4),
                        f"missing output mp4 at {self.OUTPUT_MP4}")
        self.assertGreater(os.path.getsize(self.OUTPUT_MP4), 10_000,
                           "output mp4 is suspiciously small (<10 KB)")

    def test_ffprobe_frame_count(self):
        if not _have("ffprobe"):
            self.skipTest("ffprobe not on PATH")
        # The bundled clip is 768x432 @ 12.5 fps for 30 s ≈ 377 frames.
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_frames", "-show_entries", "stream=nb_read_frames",
            "-of", "default=nokey=1:noprint_wrappers=1", self.OUTPUT_MP4
        ]).decode().strip()
        n = int(out)
        self.assertGreaterEqual(n, 350, f"output mp4 has only {n} frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("demo_dir", nargs="?", default=None)
    args, rest = parser.parse_known_args()
    if args.demo_dir:
        os.environ["APPLE_SILICON_YOLO_DEMO_DIR"] = args.demo_dir
    sys.argv = [sys.argv[0]] + rest
    unittest.main()
```

- [ ] **Step 2: Verify the test fails when the graph isn't installed**

Run before `sudo make install` to confirm the skip vs fail logic:

```bash
python3 test/function/apple_silicon_yolo/test_apple_silicon_car_detection.py /tmp/nonexistent
```
Expected: tests reported as skipped (`s`) — confirms `setUpClass` guard works.

- [ ] **Step 3: Install and run for real**

```bash
cd build && sudo make install && cd ..
python3 test/function/apple_silicon_yolo/test_apple_silicon_car_detection.py
```
Expected: two tests pass. `/tmp/apple_silicon_car_detection_result.mp4` exists, ≥350 frames.

- [ ] **Step 4: Visual spot-check (manual, not in the test)**

```bash
open /tmp/apple_silicon_car_detection_result.mp4
```
Confirm by eye: cars and trucks have red boxes; no boxes appear on roadside signs, pavement markings, or pedestrian-shaped artifacts. If everything's boxed identically to the unfiltered demo, the allowlist is silently broken — re-check Task 2 Step 3.

- [ ] **Step 5: Commit**

```bash
git add test/function/apple_silicon_yolo/test_apple_silicon_car_detection.py
git commit -m "test: apple_silicon_car_detection end-to-end smoke"
```

---

## Task 5: Documentation

**Files:**
- Modify: `src/demo/apple_silicon_yolo/README.md`
- Modify: `docs/macos-coreml.md`
- Modify: `src/demo/apple_silicon_yolo/graph/configured/README.md`

Depends on Task 4. Pure docs — no build required.

- [ ] **Step 1: Add a "Car detection variant" section to the demo README**

Open `src/demo/apple_silicon_yolo/README.md`. After the existing "Run" section, append:

```markdown
## Car detection variant

`apple_silicon_car_detection.toml` runs the same DAG but filters
detections to COCO vehicle classes only (`car=2`, `motorcycle=3`,
`bus=5`, `truck=7`) via the `yolo26_post` flowunit's `class_allowlist`
config. A 30-second 768×432 highway clip from
[`intel-iot-devkit/sample-videos`](https://github.com/intel-iot-devkit/sample-videos)
(Apache-2.0) is bundled and installed alongside the graph.

```bash
modelbox-tool flow -run \
  /usr/local/share/modelbox/demo/apple_silicon_yolo/graph/apple_silicon_car_detection.toml
open /tmp/apple_silicon_car_detection_result.mp4
```

To filter different classes in your own graph, set
`class_allowlist="<csv>"` on `yolo26_post`. Empty/missing = keep all
classes (default).
```

- [ ] **Step 2: Reference the variant from `docs/macos-coreml.md`**

In `docs/macos-coreml.md`, find the table that lists `apple_silicon_yolo demo + smoke test` (around line 18) and add a row underneath:

```markdown
| `apple_silicon_car_detection` graph + test | new        | vehicles-only filter via `yolo26_post`      |
```

- [ ] **Step 3: Update the configured/ snapshot index**

In `src/demo/apple_silicon_yolo/graph/configured/README.md`, append a row to the existing files table:

```markdown
| `apple_silicon_car_detection.toml` | detection (vehicles) | `yolov8n.mlpackage` (post-NMS [1, 300, 6]) | `yolo26_post` + class_allowlist |
```

- [ ] **Step 4: Commit**

```bash
git add src/demo/apple_silicon_yolo/README.md \
        docs/macos-coreml.md \
        src/demo/apple_silicon_yolo/graph/configured/README.md
git commit -m "docs: apple_silicon_car_detection variant + class_allowlist"
```

---

## Self-Review checklist (for the executing agent)

After the final commit, verify:

1. `git log feature/macos-coreml-integration -5` shows: spec doc → flowunit change → graph/asset commit → test → docs. Five commits, no merge commits.
2. `git status` is clean.
3. `cd build && make -j` succeeds with no new warnings.
4. `sudo make install` installs:
   - `/usr/local/share/modelbox/demo/apple_silicon_yolo/graph/apple_silicon_car_detection.toml`
   - `/usr/local/share/modelbox/demo/apple_silicon_yolo/graph/car-detection.mp4`
5. The Task 4 smoke test passes against the installed graph.
6. Re-running the original `apple_silicon_yolo.toml` produces visually identical output to before Task 2 (regression guard — empty allowlist must mean "keep all").

If any check fails, fix in a follow-up commit on the same branch; do not amend the existing commits.
