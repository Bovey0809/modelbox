#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
"""hit_window_emitter — expand flowunit that decodes audio from a configured
mp4, slides a fixed-length window over it, and emits one (1, 1, n_mels, T)
float32 mel-spectrogram tensor per window. Drives the inference variants that
use the real inference-flowunit pattern (openvino / onnxruntime).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import _flowunit as modelbox

_HERE = os.path.dirname(os.path.abspath(__file__))
# hit_sound_utils ships alongside hit_sound_detect; reuse it.
_UTILS = os.path.normpath(os.path.join(_HERE, "..", "hit_sound_detect"))
if _UTILS not in sys.path:
    sys.path.insert(0, _UTILS)
from hit_sound_utils import (  # noqa: E402
    cut_audio,
    load_audio_mono,
    mel_gray_image,
)


class HitWindowEmitter(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()

    def open(self, config):
        self.video_path = config.get_string("video_path", "")
        if not self.video_path or not os.path.exists(self.video_path):
            modelbox.error(f"hit_window_emitter: video_path not found: {self.video_path}")
            return modelbox.Status.StatusCode.STATUS_FAULT
        self.target_sr = config.get_int("target_sr", 16000)
        self.window_sec = config.get_float("window_sec", 0.5)
        self.step_sec = config.get_float("step_sec", 0.02)
        self.n_fft = config.get_int("n_fft", 512)
        self.hop_length = config.get_int("hop_length", 170)
        self.n_mels = config.get_int("n_mels", 128)
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def process(self, data_context):
        in_data = data_context.input("in_data")
        mel_out = data_context.output("mel")

        # One trigger buffer → one full audio scan. We don't care about the
        # trigger body; the video is configured at open() time.
        for _ in in_data:
            try:
                audio, sr = load_audio_mono(self.video_path, target_sr=self.target_sr)
            except Exception as exc:
                modelbox.error(f"hit_window_emitter: load_audio_mono failed: {exc}")
                return modelbox.Status.StatusCode.STATUS_FAULT
            duration = max(0.0, len(audio) / sr)
            times = np.arange(
                0.0, max(0.0, duration - self.window_sec) + 1e-9, self.step_sec
            )
            for idx, t in enumerate(times):
                seg = cut_audio(audio, sr, start_t=float(t), length_s=self.window_sec)
                img = mel_gray_image(seg, sr, self.n_fft, self.hop_length, self.n_mels)
                arr = img.astype(np.float32) / 255.0
                arr = (arr - 0.5) / 0.5
                # shape (1, 1, n_mels, T) row-major, contiguous
                arr = np.ascontiguousarray(np.expand_dims(arr, axis=(0, 1)))
                buf = modelbox.Buffer(self.get_bind_device(), arr.tobytes())
                buf.set("index", int(idx))
                buf.set("time", float(t) + self.window_sec / 2.0)
                mel_out.push_back(buf)
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def close(self):
        return modelbox.Status()

    def data_pre(self, data_context):
        return modelbox.Status()

    def data_post(self, data_context):
        return modelbox.Status()
