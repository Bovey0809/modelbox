# apple_silicon_yolo smoke test

End-to-end verification that the `src/demo/apple_silicon_yolo` graph runs
on an Apple Silicon Mac and produces a non-empty result video.

Auto-skipped on non-Darwin hosts and when modelbox-tool / the demo install
isn't available — same shape as `test/function/yolo26n_arc770/`.

```bash
python test_apple_silicon_yolo.py /usr/local/share/modelbox/demo/apple_silicon_yolo
```

The test does not assert detection accuracy — only that the DAG executes,
the Core ML inference produces output, and the encoder writes >1 KiB at
`/tmp/apple_silicon_yolo_result.mp4`.
