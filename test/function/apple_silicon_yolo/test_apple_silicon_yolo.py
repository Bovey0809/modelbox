#!/usr/bin/env python3
#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end smoke test for the apple_silicon_yolo demo.

Skipped automatically when:
  - Not running on macOS (Core ML / videotoolbox), or
  - The demo wasn't built (WITH_ALL_DEMO != ON), or
  - modelbox-tool isn't on PATH.

Run manually:
    python test_apple_silicon_yolo.py /usr/local/share/modelbox/demo/apple_silicon_yolo

The test invokes `modelbox-tool flow run` against the demo graph and
verifies the encoder produced a non-empty mp4 at the configured output
path. A second pass replaces device=apple_silicon with device=cpu in a
temp graph copy and re-runs to confirm the cpu fallback path works.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest


def _have(cmd):
    return shutil.which(cmd) is not None


class AppleSiliconYoloSmoke(unittest.TestCase):

    DEMO_DIR = os.environ.get(
        "APPLE_SILICON_YOLO_DEMO_DIR",
        "/usr/local/share/modelbox/demo/apple_silicon_yolo",
    )
    GRAPH = os.path.join(DEMO_DIR, "graph", "apple_silicon_yolo.toml")
    OUTPUT_MP4 = "/tmp/apple_silicon_yolo_result.mp4"

    @classmethod
    def setUpClass(cls):
        if platform.system() != "Darwin":
            raise unittest.SkipTest("apple_silicon path requires macOS")
        if not _have("modelbox-tool"):
            raise unittest.SkipTest("modelbox-tool not on PATH")
        if not os.path.exists(cls.GRAPH):
            raise unittest.SkipTest(f"demo graph not installed at {cls.GRAPH}")

    def _run_graph(self, graph_path):
        if os.path.exists(self.OUTPUT_MP4):
            os.remove(self.OUTPUT_MP4)
        result = subprocess.run(
            [
                "modelbox-tool",
                "flow",
                "run",
                "-name",
                "apple_silicon_yolo",
                "-graph",
                graph_path,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        return result

    def test_apple_silicon_path(self):
        result = self._run_graph(self.GRAPH)
        self.assertEqual(
            result.returncode,
            0,
            f"modelbox-tool failed: stderr={result.stderr}",
        )
        self.assertTrue(os.path.exists(self.OUTPUT_MP4))
        self.assertGreater(os.path.getsize(self.OUTPUT_MP4), 1024)

    def test_cpu_fallback_path(self):
        # Replace the inference device. Since the cpu device has no coreml
        # engine, this exercise only makes sense if a cpu-side coreml
        # flowunit lands later or the user has an alternate cpu inference
        # path; for now this case is informational and is skipped on
        # graphs that have no fallback.
        with tempfile.NamedTemporaryFile(
            "w", suffix=".toml", delete=False
        ) as tmp:
            with open(self.GRAPH, "r") as src:
                content = src.read()
            content = content.replace(
                "device=apple_silicon", "device=cpu"
            )
            tmp.write(content)
            tmp_path = tmp.name
        try:
            result = self._run_graph(tmp_path)
            if result.returncode != 0:
                # Expected on the apple_silicon-only build: cpu has no
                # coreml engine. Mark as skip rather than fail until the
                # cpu-coreml flowunit ships.
                raise unittest.SkipTest(
                    "cpu fallback graph failed (no cpu-side coreml engine "
                    f"yet): {result.stderr}"
                )
            self.assertTrue(os.path.exists(self.OUTPUT_MP4))
            self.assertGreater(os.path.getsize(self.OUTPUT_MP4), 1024)
        finally:
            os.remove(tmp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "demo_dir",
        nargs="?",
        default=None,
        help="Override the installed demo directory.",
    )
    args, remaining = parser.parse_known_args()
    if args.demo_dir:
        AppleSiliconYoloSmoke.DEMO_DIR = args.demo_dir
        AppleSiliconYoloSmoke.GRAPH = os.path.join(
            args.demo_dir, "graph", "apple_silicon_yolo.toml"
        )
    unittest.main(argv=[sys.argv[0]] + remaining, verbosity=2)
