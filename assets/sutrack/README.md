# SUTrack assets

The `sutrack_b224.onnx` (~340 MB) and `sutrack_b224.mlpackage/` (~170 MB) artifacts are
intentionally **not** tracked in git — they exceed GitHub's 100 MB file limit.

Regenerate locally before building the `sutrack` demo:

```bash
# 1. Download the upstream SUTrack b224 checkpoint (~1.2 GB)
python -c "from huggingface_hub import hf_hub_download; \
  hf_hub_download(repo_id='xche32/SUTrack', \
    filename='checkpoints/train/sutrack/sutrack_b224/SUTRACK_ep0180.pth.tar', \
    local_dir='$HOME/modelS/models/sutrack/checkpoints_hf')"

# 2. Symlink to the path config.json expects
mkdir -p $HOME/modelS/models/sutrack/checkpoints
ln -sf $HOME/modelS/models/sutrack/checkpoints_hf/checkpoints/train/sutrack/sutrack_b224/SUTRACK_ep0180.pth.tar \
       $HOME/modelS/models/sutrack/checkpoints/SUTRACK_ep0180.pth.tar

# 3. Export ONNX (~340 MB)
python $HOME/modelS/models/sutrack/deploy/export_onnx.py \
  --config $HOME/modelS/models/sutrack/config.json \
  --model sutrack_b224 --no-text \
  --output $(git rev-parse --show-toplevel)/assets/sutrack/sutrack_b224.onnx

# 4. (Apple Silicon only) Export CoreML mlpackage
python $HOME/modelS/models/sutrack/deploy/coreml/convert_torchscript_to_coreml.py \
  --input /tmp/sutrack_b224.ts.pt \
  --output $(git rev-parse --show-toplevel)/assets/sutrack/sutrack_b224.mlpackage
```

Then the cmake target `sutrack_infer_ort_model` / `sutrack_infer_openvino_model` /
`sutrack_infer_coreml_model` will pick the artifact up automatically.
