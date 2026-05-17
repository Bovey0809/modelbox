#!/usr/bin/env python3
"""Standalone TrackNet validation harness — no ModelBox.

Replicates the tennis_action graph's TrackNet branch exactly:

  video_decode (BGR HWC u8) -> resize 512x288 (HWC u8, BGR) ->
  normalize *= 1/255 (HWC float32, BGR) -> 3-frame sliding stacker
  (concat in order [prev2_flat, prev1_flat, cur_flat] along the flat axis,
  reshape to (1, 9, 288, 512)) -> ONNX TrackNet -> (1, 3, 288, 512)
  -> extract_centroid (matches src/demo/tennis_action/flowunit/
  tracknet_ball_emitter/tracknet_ball_emitter.py).

For each frame we record (frame_idx, cx, cy, peak). cx,cy are in source-
frame pixel coordinates; (-1, -1) means no detection.

Outputs per video under {out_dir}/{videoname}.{ext}:
  {videoname}.json   per-frame list + summary
  {videoname}_miss_frames/  10 evenly-spaced JPEGs from the longest miss run

Top-level {out_dir}/SUMMARY.md is written by the caller after a multi-
video run (see --videos for batch mode + --emit-summary).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort


# ----- Pipeline constants (must match the ModelBox graph). -----------------

NET_W = 512
NET_H = 288
PER_FRAME_FLOATS = 3 * NET_H * NET_W      # 442_368
STACK_FLOATS = 3 * PER_FRAME_FLOATS       # 1_327_104
SCORE_THR = 0.3
MASK_RATIO = 0.5
PEAK_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70]


def extract_centroid(heat: np.ndarray, score_thr: float,
                     mask_ratio: float) -> tuple[float, float, float]:
    """Verbatim port of tracknet_ball_emitter.extract_centroid.

    heat: CHW float32 (C >= 2). Returns (cx, cy, peak) in net-coords, or
    (-1.0, -1.0, peak) if peak < score_thr.
    """
    if heat.ndim != 3 or heat.shape[0] < 2:
        return (-1.0, -1.0, 0.0)
    fg = heat[1:].max(axis=0)  # max over foreground channels
    peak = float(fg.max())
    if peak < score_thr:
        return (-1.0, -1.0, peak)
    thr = peak * mask_ratio
    mask = fg >= thr
    ys, xs = np.where(mask)
    weights = fg[ys, xs].astype(np.float64)
    ws = weights.sum()
    if ws <= 0:
        peak_idx = int(np.argmax(fg))
        peak_y, peak_x = divmod(peak_idx, fg.shape[1])
        return (float(peak_x), float(peak_y), peak)
    cx = float((xs.astype(np.float64) * weights).sum() / ws)
    cy = float((ys.astype(np.float64) * weights).sum() / ws)
    return (cx, cy, peak)


def stack_three(prev2: np.ndarray | None, prev1: np.ndarray | None,
                cur: np.ndarray) -> np.ndarray:
    """Replicate TrackNetStackFrames exactly.

    Each frame is a flat float32 array of length PER_FRAME_FLOATS holding the
    HWC-flattened BGR-float values (since the in-graph normalize preserves
    HWC layout). The stacker writes [slot0, slot1, cur] linearly, with:

      have_prev2 && have_prev1 -> slot0 = prev2,  slot1 = prev1
      have_prev1 only          -> slot0 = prev1,  slot1 = prev1
      none                     -> slot0 = cur,    slot1 = cur

    Then reshape into (1, 9, 288, 512). This is the same byte layout the
    ModelBox graph hands the ONNX session: 3 frames of HWC-floats
    concatenated, reinterpreted by the ONNX model as 9 planar channels.
    """
    if prev2 is not None and prev1 is not None:
        slot0, slot1 = prev2, prev1
    elif prev1 is not None:
        slot0, slot1 = prev1, prev1
    else:
        slot0, slot1 = cur, cur
    stacked = np.concatenate([slot0, slot1, cur], axis=0)  # (9*H*W,)
    return stacked.reshape(1, 9, NET_H, NET_W).astype(np.float32, copy=False)


def preprocess_frame(bgr: np.ndarray) -> np.ndarray:
    """Resize + normalize one frame, returning a flat HWC float32 array."""
    # cv2.resize with INTER_LINEAR matches NppiResize default ("inter_linear").
    resized = cv2.resize(bgr, (NET_W, NET_H), interpolation=cv2.INTER_LINEAR)
    # uint8 -> float32, scale 1/255. HWC layout, BGR channel order.
    f = resized.astype(np.float32) * (1.0 / 255.0)
    return f.reshape(-1)


def build_session(onnx_path: str, use_gpu: bool) -> ort.InferenceSession:
    avail = ort.get_available_providers()
    if use_gpu and "CUDAExecutionProvider" in avail:
        providers: list[Any] = [
            ("CUDAExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CPUExecutionProvider"]
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    return sess


def ffprobe(video: str) -> dict[str, Any]:
    """Pull width, height, fps, nb_frames, duration via ffprobe -of json."""
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
            "-of", "json", video,
        ], stderr=subprocess.STDOUT, timeout=30)
        meta = json.loads(out)
        st = meta.get("streams", [{}])[0]
        w = int(st.get("width", 0))
        h = int(st.get("height", 0))
        rfr = st.get("r_frame_rate", "0/1")
        a, b = rfr.split("/")
        fps = (float(a) / float(b)) if float(b) else 0.0
        nb = int(st.get("nb_frames", 0)) if str(st.get("nb_frames", "0")).isdigit() else 0
        dur = float(st.get("duration", 0) or 0)
        return {"width": w, "height": h, "fps": fps, "nb_frames": nb, "duration": dur}
    except Exception as e:
        return {"width": 0, "height": 0, "fps": 0.0, "nb_frames": 0,
                "duration": 0.0, "ffprobe_err": str(e)}


def process_video(video: str, sess: ort.InferenceSession, out_dir: Path,
                  score_thr: float, miss_dump_n: int) -> dict[str, Any]:
    name = Path(video).name
    safe_stem = Path(video).stem
    json_path = out_dir / f"{name}.json"
    miss_dir = out_dir / f"{safe_stem}_miss_frames"

    probe = ffprobe(video)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return {"error": f"cannot open {video}", "video": name}
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or probe.get("fps", 0.0)

    scale_x = src_w / NET_W
    scale_y = src_h / NET_H

    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    per_frame_records: list[dict[str, float]] = []
    prev1: np.ndarray | None = None
    prev2: np.ndarray | None = None
    peak_hist_counts = [0] * len(PEAK_THRESHOLDS)
    valid_frames = 0

    # Track miss runs (contiguous frames where cx == -1) to find the longest.
    longest_run_len = 0
    longest_run_start = -1
    longest_run_end = -1  # inclusive
    cur_run_len = 0
    cur_run_start = -1

    t0 = time.time()
    frame_idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        cur = preprocess_frame(bgr)
        x = stack_three(prev2, prev1, cur)
        y = sess.run([out_name], {in_name: x})[0]  # (1, 3, 288, 512)
        heat = y[0]  # (3, 288, 512)
        cx_net, cy_net, peak = extract_centroid(heat, score_thr, MASK_RATIO)

        # Histogram of per-frame peaks (whether or not detection cleared thr).
        for ti, th in enumerate(PEAK_THRESHOLDS):
            if peak >= th:
                peak_hist_counts[ti] += 1

        if cx_net >= 0:
            cx = cx_net * scale_x
            cy = cy_net * scale_y
            valid_frames += 1
            # End any in-progress miss run.
            if cur_run_len > 0:
                if cur_run_len > longest_run_len:
                    longest_run_len = cur_run_len
                    longest_run_start = cur_run_start
                    longest_run_end = frame_idx - 1
                cur_run_len = 0
                cur_run_start = -1
        else:
            cx = -1.0
            cy = -1.0
            if cur_run_len == 0:
                cur_run_start = frame_idx
            cur_run_len += 1

        per_frame_records.append({
            "frame_idx": frame_idx,
            "cx": float(cx),
            "cy": float(cy),
            "peak": float(peak),
        })

        # Shift state (mirror stacker semantics).
        if prev1 is not None:
            prev2 = prev1
        prev1 = cur

        frame_idx += 1

    # Close any miss run that extended to the end of the video.
    if cur_run_len > longest_run_len:
        longest_run_len = cur_run_len
        longest_run_start = cur_run_start
        longest_run_end = frame_idx - 1

    cap.release()
    elapsed = time.time() - t0

    total = frame_idx
    valid_pct = (100.0 * valid_frames / total) if total else 0.0

    # Dump N evenly-spaced frames from inside the longest miss run.
    miss_frames_saved: list[int] = []
    if longest_run_len > 0 and miss_dump_n > 0:
        miss_dir.mkdir(parents=True, exist_ok=True)
        # Pick N indices spanning [start, end] inclusive.
        if longest_run_len <= miss_dump_n:
            idxs = list(range(longest_run_start, longest_run_end + 1))
        else:
            idxs = [
                int(round(longest_run_start + i * (longest_run_len - 1) / (miss_dump_n - 1)))
                for i in range(miss_dump_n)
            ]
        # Re-open the video and seek to each index for the dump.
        cap2 = cv2.VideoCapture(video)
        # Linear pass instead of seek (mp4 keyframe seek can be wonky).
        idx_set = sorted(set(idxs))
        cur_idx = 0
        target_iter = iter(idx_set)
        try:
            next_target: int | None = next(target_iter)
        except StopIteration:
            next_target = None
        while next_target is not None:
            ok, frame_bgr = cap2.read()
            if not ok:
                break
            if cur_idx == next_target:
                jpg = miss_dir / f"frame_{cur_idx:06d}.jpg"
                cv2.imwrite(str(jpg), frame_bgr,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                miss_frames_saved.append(cur_idx)
                try:
                    next_target = next(target_iter)
                except StopIteration:
                    next_target = None
            cur_idx += 1
        cap2.release()

    longest_miss_run_sec = (longest_run_len / fps) if fps else 0.0

    summary = {
        "video": name,
        "path": video,
        "total_frames": total,
        "valid_frames": valid_frames,
        "valid_pct": round(valid_pct, 3),
        "fps": round(fps, 4),
        "resolution": [src_w, src_h],
        "score_thr": score_thr,
        "peak_hist": [{"thr": t, "count": c}
                      for t, c in zip(PEAK_THRESHOLDS, peak_hist_counts)],
        "longest_miss_run": {
            "len_frames": longest_run_len,
            "len_seconds": round(longest_miss_run_sec, 3),
            "start_frame": longest_run_start,
            "end_frame": longest_run_end,
            "saved_frames": miss_frames_saved,
            "saved_dir": str(miss_dir) if miss_frames_saved else "",
        },
        "elapsed_seconds": round(elapsed, 2),
        "ffprobe": probe,
    }
    payload = {"summary": summary, "frames": per_frame_records}
    out_dir.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as f:
        json.dump(payload, f)
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True, help="Path to tracknet_deep.onnx")
    p.add_argument("--video", action="append", default=[],
                   help="Video file (repeatable)")
    p.add_argument("--out_dir", default="/tmp/tracknet_validate")
    p.add_argument("--score_thr", type=float, default=SCORE_THR)
    p.add_argument("--miss_dump_n", type=int, default=10)
    p.add_argument("--cpu", action="store_true",
                   help="Force CPUExecutionProvider")
    args = p.parse_args()

    if not args.video:
        print("no --video given", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sess = build_session(args.onnx, use_gpu=not args.cpu)
    print(f"providers: {sess.get_providers()}", flush=True)

    all_summaries = []
    for v in args.video:
        print(f"=== {v} ===", flush=True)
        s = process_video(v, sess, out_dir, args.score_thr, args.miss_dump_n)
        all_summaries.append(s)
        print(json.dumps({k: s[k] for k in ("video", "total_frames",
              "valid_frames", "valid_pct", "fps", "resolution",
              "longest_miss_run", "elapsed_seconds")}, indent=2), flush=True)

    # Drop a roll-up index for the SUMMARY.md generator on the local side.
    with (out_dir / "_index.json").open("w") as f:
        json.dump(all_summaries, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
