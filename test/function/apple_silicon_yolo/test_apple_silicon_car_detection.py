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
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        rc = subprocess.call([
            "modelbox-tool", "flow", "run",
            "-name", "apple_silicon_car_detection",
            "-graph", self.GRAPH,
        ])
        self.assertEqual(rc, 0, "modelbox-tool flow run exited non-zero")
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
        AppleSiliconCarDetectionSmoke.DEMO_DIR = args.demo_dir
        AppleSiliconCarDetectionSmoke.GRAPH = os.path.join(
            args.demo_dir, "graph", "apple_silicon_car_detection.toml"
        )
    unittest.main(argv=[sys.argv[0]] + rest, verbosity=2)
