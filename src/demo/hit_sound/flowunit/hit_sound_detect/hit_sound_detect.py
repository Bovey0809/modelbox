#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
"""hit_sound_detect — Python flowunit running the tennis_audio_seg ONNX
classifier inside ModelBox.

Triggered with a JSON request body:
    {"video_path": "/abs/path/to/video.mp4",
     "intervals_csv": "/tmp/...csv",       # optional override
     "overlay_mp4":   "/tmp/...mp4",       # optional override (empty disables)
     "interval_mode": "hit"}                # hit | rally | both

Returns JSON:
    {"video": "...", "windows": N, "hit_intervals": [[s,e], ...],
     "rally_intervals": [[s,e], ...], "intervals_csv": "...",
     "overlay_mp4": "...", "model": "...", "mode": "hit"}
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import _flowunit as modelbox
import onnxruntime as ort

# Resolve sibling helpers regardless of how modelbox imports us.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from hit_sound_utils import (  # noqa: E402
    BufferedRallyDetector,
    DurationFilter,
    OutputSmoother,
    cut_audio,
    get_audio_start_offset,
    load_audio_mono,
    mel_gray_image,
    merge_positive_windows,
    write_overlay_video,
)


def _softmax_pos(logits: np.ndarray) -> float:
    x = logits[0].astype(np.float64)
    x = x - x.max()
    e = np.exp(x)
    return float(e[1] / e.sum())


class HitSoundDetect(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()
        self.session: ort.InferenceSession = None  # type: ignore
        self.input_name = "input"

    def open(self, config):
        # Resolve the ONNX model path relative to this file by default.
        raw_model = config.get_string("model_path", "best.onnx")
        model_path = raw_model if os.path.isabs(raw_model) else os.path.join(_HERE, raw_model)
        if not os.path.exists(model_path):
            modelbox.error(f"hit_sound_detect: model not found: {model_path}")
            return modelbox.Status.StatusCode.STATUS_FAULT
        self.model_path = model_path

        self.target_sr = config.get_int("target_sr", 16000)
        self.window_sec = config.get_float("window_sec", 0.5)
        self.step_sec = config.get_float("step_sec", 0.02)
        self.n_fft = config.get_int("n_fft", 512)
        self.hop_length = config.get_int("hop_length", 170)
        self.n_mels = config.get_int("n_mels", 128)
        self.threshold = config.get_float("threshold", 0.5)
        self.smooth_win = config.get_int("smooth_win", 5)
        self.max_hit_dur = config.get_float("max_hit_dur", 1.0)
        self.max_gap = config.get_float("max_gap", 3.0)
        self.buffer_time = config.get_float("buffer_time", 4.0)
        self.default_intervals_csv = config.get_string(
            "default_intervals_csv", "/tmp/hit_sound_intervals.csv"
        )
        self.default_overlay_mp4 = config.get_string(
            "default_overlay_mp4", "/tmp/hit_sound_overlay.mp4"
        )
        self.write_overlay = config.get_bool("write_overlay", True)

        providers = ort.get_available_providers()
        chosen = [p for p in ("CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
                  if p in providers] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.model_path, providers=chosen)
        self.input_name = self.session.get_inputs()[0].name
        modelbox.info(f"hit_sound_detect: ONNX loaded {self.model_path} providers={chosen}")
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def process(self, data_context):
        in_data = data_context.input("in_data")
        out_data = data_context.output("out_data")

        for buf in in_data:
            try:
                req = json.loads(str(buf))
            except Exception as exc:
                self._reply_error(out_data, f"invalid JSON request: {exc}")
                continue
            try:
                result = self._run(req)
            except Exception as exc:
                modelbox.error(f"hit_sound_detect: {exc}")
                self._reply_error(out_data, str(exc))
                continue
            self._reply(out_data, result)
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def close(self):
        self.session = None  # type: ignore
        return modelbox.Status()

    def data_pre(self, data_context):
        return modelbox.Status()

    def data_post(self, data_context):
        return modelbox.Status()

    # ----- internals -----

    def _run(self, req: Dict) -> Dict:
        video_path = req.get("video_path")
        if not video_path:
            raise ValueError("video_path is required")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"video not found: {video_path}")
        interval_mode = req.get("interval_mode", "hit")
        intervals_csv = req.get("intervals_csv", self.default_intervals_csv)
        overlay_mp4 = req.get("overlay_mp4", self.default_overlay_mp4 if self.write_overlay else "")

        t0 = time.time()
        audio, sr = load_audio_mono(video_path, target_sr=self.target_sr)
        audio_offset = get_audio_start_offset(video_path)
        duration = max(0.0, len(audio) / sr)
        if duration <= 0:
            raise RuntimeError("invalid audio duration")

        times = np.arange(0.0, max(0.0, duration - self.window_sec) + 1e-9, self.step_sec)
        smoother = OutputSmoother(window_size=self.smooth_win)
        dur_filter = DurationFilter(max_duration=self.max_hit_dur)
        rally = BufferedRallyDetector(
            max_inter_hit_time=self.max_gap, buffer_time=self.buffer_time, step=self.step_sec
        )

        pos_windows: List[Tuple[float, float]] = []
        probs: List[Tuple[float, float]] = []
        for t in times:
            seg = cut_audio(audio, sr, start_t=float(t), length_s=self.window_sec)
            img = mel_gray_image(seg, sr, self.n_fft, self.hop_length, self.n_mels)
            arr = img.astype(np.float32) / 255.0
            arr = (arr - 0.5) / 0.5
            arr = np.expand_dims(arr, axis=(0, 1))
            prob = _softmax_pos(self.session.run(None, {self.input_name: arr})[0])
            center = float(t) + audio_offset + self.window_sec / 2.0
            probs.append((center, prob))
            pred_raw = 1 if prob >= self.threshold else 0
            pred_smooth = smoother.smooth(pred_raw)
            pred_filt = dur_filter.filter(center, pred_smooth)
            rally.process(center, pred_filt)
            if pred_filt == 1:
                pos_windows.append((float(t) + audio_offset, float(t) + audio_offset + self.window_sec))

        hit_intervals = merge_positive_windows(pos_windows, join_gap=self.step_sec)
        ts, states = rally.finalize()
        rally_intervals: List[Tuple[float, float]] = []
        start_t = None
        prev_t = None
        for tt, ss in zip(ts.tolist(), states.tolist()):
            tt = float(tt); ss = int(ss)
            if ss == 1 and start_t is None:
                start_t = tt
            elif ss == 0 and start_t is not None:
                rally_intervals.append((start_t, (prev_t if prev_t is not None else tt) + self.step_sec))
                start_t = None
            prev_t = tt
        if start_t is not None and prev_t is not None:
            rally_intervals.append((start_t, prev_t + self.step_sec))

        if interval_mode == "rally":
            selected = rally_intervals
        elif interval_mode == "both":
            selected = rally_intervals
        else:
            selected = hit_intervals

        if intervals_csv:
            Path(intervals_csv).parent.mkdir(parents=True, exist_ok=True)
            Path(intervals_csv).write_text(
                "\n".join([f"{s:.3f},{e:.3f}" for s, e in selected]),
                encoding="utf-8",
            )

        if overlay_mp4:
            Path(overlay_mp4).parent.mkdir(parents=True, exist_ok=True)
            write_overlay_video(video_path, overlay_mp4, selected)

        elapsed = time.time() - t0
        modelbox.info(
            f"hit_sound_detect: video={video_path} windows={len(times)} "
            f"hits={len(hit_intervals)} rallies={len(rally_intervals)} "
            f"mode={interval_mode} elapsed={elapsed:.2f}s"
        )
        return {
            "video": video_path,
            "model": self.model_path,
            "mode": interval_mode,
            "windows": int(len(times)),
            "duration_s": float(duration),
            "elapsed_s": round(elapsed, 3),
            "hit_intervals": [[round(s, 3), round(e, 3)] for s, e in hit_intervals],
            "rally_intervals": [[round(s, 3), round(e, 3)] for s, e in rally_intervals],
            "intervals_csv": intervals_csv,
            "overlay_mp4": overlay_mp4,
        }

    def _reply(self, out_data, payload: Dict) -> None:
        body = (json.dumps(payload) + chr(0)).encode("utf-8").strip()
        out_buf = modelbox.Buffer(self.get_bind_device(), body)
        out_data.push_back(out_buf)

    def _reply_error(self, out_data, msg: str) -> None:
        self._reply(out_data, {"error": msg})
