# yolo26n_arc770 end-to-end smoke test

Verifies the full demo graph runs and produces output, both on the Intel Arc
(via the OpenVINO GPU plugin) and on the CPU fallback path.

This is a Python test rather than a gtest because:
- It exercises the *installed* graph + flowunits, not just compiled C++.
- It needs a working OpenVINO runtime and (for the Arc path) the GPU plugin
  + Level Zero stack at runtime, neither of which is available at compile
  time.
- It needs `modelbox-tool` on PATH, which only exists post-install.

## Run

```bash
# After `make install` from a build configured with -DWITH_ALL_DEMO=ON
sudo apt install libze-dev openvino
pip install ultralytics openvino

python test/function/yolo26n_arc770/test_yolo26n_arc770.py
# or, if the demo is installed somewhere non-standard:
python test/function/yolo26n_arc770/test_yolo26n_arc770.py /opt/modelbox/share/modelbox/demo/yolo26n_arc770
```

## Skip conditions (automatic)

- `modelbox-tool` not on PATH -> entire suite skipped
- Demo graph not installed at the expected path -> entire suite skipped
- `OpenVINO Core().available_devices` does not include `"GPU"` -> only the
  Arc test is skipped; the CPU fallback test still runs

## What it asserts

1. `modelbox-tool flow run` against the installed graph exits 0.
2. `/tmp/yolo26n_arc770_result.mp4` exists and is at least 1 KiB (i.e.
   the encoder produced real frames, not just a header).
3. Same assertions on a temp copy of the graph with `device=intel_gpu`
   replaced by `device=cpu`, to confirm the CPU OpenVINO plugin path.
