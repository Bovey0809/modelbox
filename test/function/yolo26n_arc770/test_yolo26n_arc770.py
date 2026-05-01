#!/usr/bin/env python3
#
# Copyright 2021 The Modelbox Project Authors. All Rights Reserved.
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

"""End-to-end smoke test for the yolo26n_arc770 demo.

Skipped automatically when:
  - The demo wasn't built (WITH_ALL_DEMO != ON), or
  - The Arc A770 / OpenVINO GPU plugin isn't available, or
  - modelbox-tool isn't on PATH.

Run manually:
    python test_yolo26n_arc770.py /usr/local/share/modelbox/demo/yolo26n_arc770

The test invokes `modelbox-tool flow run` against the demo graph, then
verifies that the encoder produced a non-empty mp4 at the configured
output path. A second pass replaces device=intel_gpu with device=cpu
in a temp graph copy and re-runs to confirm the cpu fallback path.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import unittest


def _have(cmd):
    return shutil.which(cmd) is not None


def _openvino_has_gpu():
    try:
        import openvino as ov  # noqa: WPS433

        return "GPU" in ov.Core().available_devices
    except Exception:  # noqa: BLE001
        return False


class Yolo26nArc770Smoke(unittest.TestCase):

    DEMO_DIR = os.environ.get(
        "YOLO26N_ARC770_DEMO_DIR",
        "/usr/local/share/modelbox/demo/yolo26n_arc770",
    )
    GRAPH = os.path.join(DEMO_DIR, "graph", "yolo26n_arc770.toml")
    OUTPUT_MP4 = "/tmp/yolo26n_arc770_result.mp4"

    @classmethod
    def setUpClass(cls):
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
                "yolo26n_arc770",
                "-graph",
                graph_path,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        return result

    def test_intel_gpu_path(self):
        if not _openvino_has_gpu():
            raise unittest.SkipTest("OpenVINO GPU plugin not available")
        result = self._run_graph(self.GRAPH)
        self.assertEqual(
            result.returncode,
            0,
            f"modelbox-tool failed: stderr={result.stderr}",
        )
        self.assertTrue(os.path.exists(self.OUTPUT_MP4))
        self.assertGreater(os.path.getsize(self.OUTPUT_MP4), 1024)

    def test_cpu_fallback_path(self):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".toml", delete=False
        ) as tmp:
            with open(self.GRAPH, "r") as src:
                content = src.read()
            content = content.replace("device=intel_gpu", "device=cpu")
            tmp.write(content)
            tmp_path = tmp.name
        try:
            result = self._run_graph(tmp_path)
            self.assertEqual(
                result.returncode,
                0,
                f"cpu fallback run failed: stderr={result.stderr}",
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
        Yolo26nArc770Smoke.DEMO_DIR = args.demo_dir
        Yolo26nArc770Smoke.GRAPH = os.path.join(
            args.demo_dir, "graph", "yolo26n_arc770.toml"
        )
    unittest.main(argv=[sys.argv[0]] + remaining, verbosity=2)
