# PaddleOCR demo graph

End-to-end OCR pipeline (PP-OCRv4 detection -> direction classification -> recognition)
running on top of the ModelBox graph engine. Decodes a video stream on CUDA, runs the
three Paddle Inference models in sequence, draws the recognised text + boxes onto each
frame, and re-encodes to MP4. All custom flowunits live under
`src/drivers/devices/cpu/flowunit/paddle_ocr_*` and
`src/drivers/devices/cuda/flowunit/paddle/`.

## 1. Prerequisites

- Built ModelBox tree with the Paddle Inference driver enabled. Configure CMake with:
  ```
  cmake .. -DPaddleInference_DIR=/path/to/paddle_inference
  ```
- The Paddle Inference shared libs must be on the loader's path at run time:
  ```bash
  export PADDLE=/autodl-fs/data/paddle_inference_pkg/paddle_inference
  export LD_LIBRARY_PATH=$PADDLE/paddle/lib:$PADDLE/third_party/install/mkldnn/lib:$PADDLE/third_party/install/mklml/lib:$LD_LIBRARY_PATH
  ```
  (Adjust the prefix to wherever your Paddle SDK lives.)
- `wget`, `tar`, and ~250 MB of free disk for the model bundles.

## 2. Fetch models + charset

```bash
bash examples/paddle_ocr/download_models.sh
```

Default destination is `<repo_root>/models/paddle_ocr/`. Pass an explicit path as the
first argument to relocate. The script:

1. Downloads three PaddleOCR inference bundles (`ch_PP-OCRv4_det_infer`,
   `ch_ppocr_mobile_v2.0_cls_infer`, `ch_PP-OCRv4_rec_infer`).
2. Downloads `ppocr_keys_v1.txt` (Chinese + Latin + digits charset).
3. Writes a `paddle_inference` virtual-flowunit TOML inside each model directory so
   ModelBox can register each model as a graph-callable flowunit.

## 3. Edit the graph

Open `examples/paddle_ocr/paddle_ocr_video.toml` and adjust:

- `source_url` on the `video_input` node — point at the input video.
- `default_dest_url` on the `video_encoder` node — output MP4 path.
- The five `[[driver.dir]]` entries — point at your build tree and your
  `models/paddle_ocr/<model_name>` directories.
- `rec_post.char_dict_file` — absolute path to `ppocr_keys_v1.txt`.

## 4. Run

```bash
./build/release/bin/modelbox-tool flow run \
  -name paddle_ocr_video \
  -graph examples/paddle_ocr/paddle_ocr_video.toml
```

The output MP4 lands at the `default_dest_url` you set on `video_encoder`.

## Discovering the real tensor names

PaddleOCR exports occasionally name their inputs/outputs differently across releases.
If `det_infer`/`cls_infer`/`rec_infer` fail with port-mismatch errors, dump the real
names:

```bash
python - <<'PY'
import paddle.inference as pi
for m in ["ch_PP-OCRv4_det_infer", "ch_ppocr_mobile_v2.0_cls_infer", "ch_PP-OCRv4_rec_infer"]:
    cfg = pi.Config(f"models/paddle_ocr/{m}/inference.pdmodel",
                    f"models/paddle_ocr/{m}/inference.pdiparams")
    p = pi.create_predictor(cfg)
    print(m, p.get_input_names(), p.get_output_names())
PY
```

Update the corresponding edges in `paddle_ocr_video.toml`, and the
`[input.input1].name` / `[output.output1].name` fields inside each model's auto-generated
TOML. (Leaving `config.input_name` / `config.output_name` empty falls back to whatever
the Paddle model itself declares, so the explicit edge names in the graph are usually
the only thing that needs to match.)

## Known limits

- The recogniser ships with the Chinese + Latin + digits charset
  (`ppocr_keys_v1.txt`). For other scripts, download the matching
  `<lang>_dict.txt` from
  <https://github.com/PaddlePaddle/PaddleOCR/tree/main/ppocr/utils> and point
  `rec_post.char_dict_file` at it (and re-export a matching recogniser if the
  vocabulary differs from PP-OCRv4 Chinese).
- TensorRT acceleration is wired (`config.enable_trt=true`,
  `config.trt_precision="fp16"`/`"int8"`) but the first run will spend a minute
  building engines. Leave it off for the demo.
- The graph assumes a CUDA build. CPU-only Paddle inference works but you must
  edit the model TOMLs (`[base].device = "cpu"`) and the graph (`device=cpu` on
  every `*_infer` node).
- macOS is not a supported build target for ModelBox itself; develop inside the
  `modelbox/modelbox-develop-*` Docker images.
