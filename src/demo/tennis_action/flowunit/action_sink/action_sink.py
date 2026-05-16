#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
"""action_sink — final sink for the tennis_action pipeline.

Consumes per-hit ASFormer logits + hit_meta, plus per-frame ball_pos and
tracked_poses streams, plus the collapsed dropped_hits list. At session
end, writes:
  - <output_json>            structured per-hit results + summary
  - <output_overlay>         source video with skeleton + ball + action
                             labels burned in
  - <output_dropped>         per-hit drop reasons
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import _flowunit as modelbox

try:
    import _tennis_store as _store
    _HAS_STORE = True
except ImportError:
    _store = None
    _HAS_STORE = False

# cv2 is optional at import time; the overlay step gracefully degrades.
try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except ImportError:
    cv2 = None
    _HAS_CV2 = False


# COCO-17 skeleton edges (pairs of kpt indices to draw lines between).
_SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
             (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
             (0, 1), (0, 2), (1, 3), (2, 4)]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def write_results(out_json: Path, out_dropped: Path,
                  confirmed: list[dict[str, Any]],
                  dropped: list[dict[str, Any]],
                  fps: float) -> None:
    """Write the JSON results + dropped-hits file."""
    audio_hits = len(confirmed) + len(dropped)
    drop_reasons: dict[str, int] = {}
    for d in dropped:
        r = d.get("reason", "unknown")
        drop_reasons[r] = drop_reasons.get(r, 0) + 1
    body = {
        "summary": {"audio_hits": audio_hits, "confirmed": len(confirmed),
                    "dropped": drop_reasons},
        "hits": [],
    }
    for h in confirmed:
        body["hits"].append({
            **h,
            "t_sec": h["frame_idx"] / fps if fps > 0 else 0.0,
        })
    out_json.write_text(json.dumps(body, indent=2) + "\n")
    out_dropped.write_text(json.dumps(dropped, indent=2) + "\n")


def render_overlay(source_video: str, output_path: str,
                   confirmed: list[dict[str, Any]],
                   poses_per_frame: dict[int, list[dict[str, Any]]],
                   ball_per_frame: dict[int, tuple[float, float, float]],
                   hit_label_persist_frames: int = 30,
                   encoder: str = "libx264") -> bool:
    """Re-open the source video, draw skeleton/ball/action labels onto each
    frame, write the result via cv2. Returns True on success, False on
    failure (caller decides whether to fall back / log)."""
    # NOTE: encoder/overlay_encoder is currently advisory only — cv2.VideoWriter
    # uses fourcc not named encoders. For real h264_nvenc output, this function
    # would need to be rewritten to pipe frames to ffmpeg via subprocess.
    # Documented as a v2 concern in docs/superpowers/specs/...-design.md §7.
    if not _HAS_CV2:
        modelbox.error("action_sink: cv2 not available; skipping overlay")
        return False
    cap = cv2.VideoCapture(source_video)
    if not cap.isOpened():
        modelbox.error(f"action_sink: cannot open source {source_video}")
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_map = {"libx264": "avc1", "h264_nvenc": "avc1",
                  "mp4v": "mp4v", "libx265": "hev1"}
    fourcc_str = fourcc_map.get(encoder, "mp4v")
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        # Preferred codec unavailable; fall back to mp4v (always present in
        # OpenCV builds with FFmpeg). This mirrors the pre-v2 behaviour and
        # keeps CI green on machines without libopenh264 / nvenc.
        if fourcc_str != "mp4v":
            modelbox.info(
                f"action_sink: {fourcc_str} unavailable, retrying with mp4v")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            cap.release()
            modelbox.error(
                f"action_sink: cannot open writer for {output_path}")
            return False
    # Per-frame label map.
    label_map: dict[int, str] = {}
    for hit in confirmed:
        f = int(hit["frame_idx"])
        label = f"hit{hit.get('hit_id', 0)}: {hit.get('action', '?')}"
        for off in range(0, hit_label_persist_frames):
            label_map[f + off] = label
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Ball circle.
        ball = ball_per_frame.get(frame_idx)
        if ball and ball[0] >= 0:
            cv2.circle(frame, (int(ball[0]), int(ball[1])), 8, (0, 0, 255), 2)
        # Skeletons.
        for pose in poses_per_frame.get(frame_idx, []):
            kpts = pose.get("kpts", [])
            for a, b in _SKELETON:
                if a < len(kpts) and b < len(kpts):
                    pa, pb = kpts[a], kpts[b]
                    if pa[2] > 0.3 and pb[2] > 0.3:
                        cv2.line(frame, (int(pa[0]), int(pa[1])),
                                 (int(pb[0]), int(pb[1])), (0, 255, 0), 2)
            tid = pose.get("track_id", -1)
            bbox = pose.get("bbox", [0, 0, 0, 0])
            cv2.putText(frame, f"id{tid}", (int(bbox[0]), int(bbox[1]) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        # Action label.
        label = label_map.get(frame_idx)
        if label:
            cv2.rectangle(frame, (10, 10), (10 + 9 * len(label), 40),
                          (0, 0, 0), -1)
            cv2.putText(frame, label, (14, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)
        frame_idx += 1
    writer.release()
    cap.release()
    return True


class ActionSink(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()
        self._logits: list[np.ndarray] = []
        self._metas: list[dict[str, Any]] = []
        self._dropped: list[dict[str, Any]] = []
        self._ball: dict[int, tuple[float, float, float]] = {}
        self._poses: dict[int, list[dict[str, Any]]] = {}
        # Tracks whether _finalize_outputs ran with real data; close() uses
        # this to decide whether to emit the final fallback "0 hits" artifacts.
        self._finalized: bool = False

    def open(self, config):
        self.output_json = config.get_string("output_json",
                                             "/tmp/tennis_actions.json")
        self.output_overlay = config.get_string("output_overlay",
                                                "/tmp/tennis_actions_overlay.mp4")
        self.output_dropped = config.get_string("output_dropped",
                                                "/tmp/tennis_actions_dropped.json")
        self.class_names_path = config.get_string("class_names_path", "")
        self.overlay_encoder = config.get_string("overlay_encoder",
                                                 "h264_nvenc")
        self.overlay_fallback_encoder = config.get_string(
            "overlay_fallback_encoder", "libx264")
        self.hit_label_persist_frames = config.get_int(
            "hit_label_persist_frames", 30)
        self.source_video = config.get_string("source_video", "")
        if self.class_names_path and os.path.exists(self.class_names_path):
            try:
                with open(self.class_names_path, encoding="utf-8") as f:
                    self.classes = json.load(f).get("classes", [])
            except OSError as exc:
                modelbox.error(
                    f"action_sink: cannot read class_names_path: {exc}")
                self.classes = []
        else:
            self.classes = []
        # Pre-write empty placeholder JSON files at flowunit init.
        # The engine's StreamFlowUnitDataContext::IsDataPre/IsDataPost return
        # false when the input stream is empty (only an end_flag arrives,
        # which happens when hit_fuse emits 0 confirmed hits). In that case
        # neither data_pre nor data_post fire. close() isn't called either
        # under SIGKILL (modelbox-tool's RunImpl spin loop never exits). So
        # the only place we're guaranteed to run is here in open().
        try:
            write_results(Path(self.output_json), Path(self.output_dropped),
                          [], [], 30.0)
        except OSError as exc:
            modelbox.error(
                f"action_sink: cannot pre-write placeholder JSON: {exc}")
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def data_pre(self, data_context):
        self._logits = []
        self._metas = []
        self._dropped = []
        self._ball = {}
        self._poses = {}
        return modelbox.Status()

    def process(self, data_context):
        for buf in data_context.input("logits"):
            arr = np.frombuffer(buf.as_object(), dtype=np.float32)
            self._logits.append(arr)
        for buf in data_context.input("hit_meta"):
            self._metas.append(
                json.loads(bytes(buf.as_object()).decode("utf-8"))
            )
        # dropped_hits, ball, and pose history are all read from the shared
        # _tennis_store side-channel in data_post(), not as direct port inputs,
        # to avoid ModelBox stream-cardinality mismatches against the
        # N-buffer/session logits + hit_meta streams.
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def data_post(self, data_context):
        self._finalize_outputs()
        return modelbox.Status()

    def close(self):
        # Fallback: if data_post never ran (e.g. hit_fuse emitted an
        # empty stream so StreamFlowUnitDataContext::IsDataPost returned
        # false), still emit JSON/overlay so a "0 hits" run is
        # distinguishable from a crashed run.
        if not self._finalized:
            self._finalize_outputs()
        return modelbox.Status()

    def _finalize_outputs(self) -> None:
        """Resolve confirmed/dropped hits, then write JSON + overlay.

        Called from data_post() when real buffers arrived, and from close()
        as the final fallback when data_post was skipped because the input
        stream was empty.
        """
        # Fetch ball, pose histories, and dropped-hits list from the shared store.
        if _HAS_STORE:
            self._ball = _store.get_all_ball()
            self._poses = _store.get_all_poses()
            self._dropped = _store.get_all_dropped()
        confirmed: list[dict[str, Any]] = []
        for logit, meta in zip(self._logits, self._metas):
            probs = _softmax(logit.flatten())
            cls_idx = int(np.argmax(probs))
            cls_name = self.classes[cls_idx] if cls_idx < len(self.classes) \
                else f"cls{cls_idx}"
            confirmed.append({
                **meta,
                "action": cls_name,
                "action_conf": float(probs[cls_idx]),
            })
        fps = self._infer_fps_from_video()
        try:
            write_results(Path(self.output_json), Path(self.output_dropped),
                          confirmed, self._dropped, fps)
        except OSError as exc:
            modelbox.error(f"action_sink: JSON write failed: {exc}")
        if self.source_video and os.path.exists(self.source_video):
            ok = render_overlay(
                source_video=self.source_video,
                output_path=self.output_overlay,
                confirmed=confirmed,
                poses_per_frame=self._poses,
                ball_per_frame=self._ball,
                hit_label_persist_frames=self.hit_label_persist_frames,
                encoder=self.overlay_encoder,
            )
            if not ok:
                modelbox.error("action_sink: overlay render failed")
        self._finalized = True

    def _infer_fps_from_video(self) -> float:
        if not _HAS_CV2 or not self.source_video \
                or not os.path.exists(self.source_video):
            return 30.0
        cap = cv2.VideoCapture(self.source_video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        return float(fps)
