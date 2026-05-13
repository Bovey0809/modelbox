#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
"""sutrack_track — Python flowunit hosting the SUTrack single-object tracker.

The flowunit imports `load_tracker` from the user's modelS/sutrack codebase
(configured via [config].sutrack_dir) and runs `tracker.initialize()` +
`tracker.track()` over a posted video. POST a JSON request like:

    {"video_path": "/abs/clip.mp4",
     "init_bbox": [x, y, w, h],
     "text": "optional language prompt",
     "output_video": "/tmp/sutrack_overlay.mp4",
     "output_json":  "/tmp/sutrack_bboxes.json",
     "bbox_normalized": false}

Returns JSON with the per-frame bbox list (xywh, pixels), output paths, and
elapsed time.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List

import _flowunit as modelbox


class SUTrackTrack(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()
        self._tracker_loader = None  # callable from modelloader.load_tracker

    def open(self, config):
        self.sutrack_dir = config.get_string("sutrack_dir", "")
        if not self.sutrack_dir or not os.path.isdir(self.sutrack_dir):
            modelbox.error(f"sutrack_track: sutrack_dir not found: {self.sutrack_dir}")
            return modelbox.Status.StatusCode.STATUS_FAULT
        # Prepend the SUTrack repo to sys.path so its `modelloader`/`sutrack.lib`
        # imports resolve to that tree, not to anything else on PYTHONPATH.
        if self.sutrack_dir not in sys.path:
            sys.path.insert(0, self.sutrack_dir)
        pkg = os.path.join(self.sutrack_dir, "sutrack")
        if pkg not in sys.path:
            sys.path.insert(0, pkg)
        try:
            from modelloader import load_tracker  # noqa: WPS433
        except Exception as exc:
            modelbox.error(f"sutrack_track: cannot import modelloader: {exc}")
            return modelbox.Status.StatusCode.STATUS_FAULT
        self._tracker_loader = load_tracker

        self.model_name = config.get_string("model_name", "sutrack_b224")
        cp = config.get_string("config_path", "")
        self.config_path = cp if cp else os.path.join(self.sutrack_dir, "config.json")
        self.device = config.get_string("device", "cpu")
        self.default_output_video = config.get_string(
            "default_output_video", "/tmp/sutrack_overlay.mp4"
        )
        self.default_output_json = config.get_string(
            "default_output_json", "/tmp/sutrack_bboxes.json"
        )
        modelbox.info(
            f"sutrack_track: sutrack_dir={self.sutrack_dir} model={self.model_name} "
            f"config={self.config_path} device={self.device}"
        )
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
                modelbox.error(f"sutrack_track: {exc}")
                self._reply_error(out_data, str(exc))
                continue
            self._reply(out_data, result)
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def close(self):
        return modelbox.Status()

    def data_pre(self, data_context):
        return modelbox.Status()

    def data_post(self, data_context):
        return modelbox.Status()

    # ----- internals -----

    def _run(self, req: Dict) -> Dict:
        # Lazy heavy-deps import so flowunit module load doesn't pay for cv2/torch.
        import cv2  # noqa: WPS433
        import numpy as np  # noqa: WPS433

        video_path = req.get("video_path")
        if not video_path or not os.path.exists(video_path):
            raise FileNotFoundError(f"video not found: {video_path}")
        init_bbox = req.get("init_bbox")
        if not init_bbox or len(init_bbox) != 4:
            raise ValueError("init_bbox must be a 4-element [x,y,w,h] list")
        init_bbox = [float(v) for v in init_bbox]
        text = req.get("text")
        output_video = req.get("output_video", self.default_output_video)
        output_json = req.get("output_json", self.default_output_json)
        bbox_normalized = bool(req.get("bbox_normalized", False))

        t0 = time.time()
        tracker, _ = self._tracker_loader(
            config_path=self.config_path,
            model_name=self.model_name,
            override_device=self.device,
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"cv2 failed to open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_video:
            os.makedirs(os.path.dirname(output_video) or "/", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        ret, frame0 = cap.read()
        if not ret or frame0 is None:
            cap.release()
            raise RuntimeError("failed to read first frame")
        if bbox_normalized:
            init_bbox = [
                init_bbox[0] * width, init_bbox[1] * height,
                init_bbox[2] * width, init_bbox[3] * height,
            ]
        init_rgb = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
        init_info = {"init_bbox": init_bbox}
        if text:
            init_info["init_nlp"] = text
        tracker.initialize(init_rgb, init_info)
        bboxes: List[List[float]] = [init_bbox]
        if writer:
            self._draw(frame0, init_bbox)
            writer.write(frame0)

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out = tracker.track(frame_rgb)
            bbox = [float(v) for v in out["target_bbox"]]
            bboxes.append(bbox)
            if writer:
                self._draw(frame, bbox)
                writer.write(frame)
        cap.release()
        if writer:
            writer.release()

        if output_json:
            os.makedirs(os.path.dirname(output_json) or "/", exist_ok=True)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump({"bboxes_xywh": bboxes}, f)

        elapsed = time.time() - t0
        modelbox.info(
            f"sutrack_track: video={video_path} frames={len(bboxes)} "
            f"fps={len(bboxes)/max(elapsed, 1e-3):.2f} elapsed={elapsed:.2f}s"
        )
        return {
            "video": video_path,
            "model": self.model_name,
            "frames": len(bboxes),
            "elapsed_s": round(elapsed, 3),
            "fps_processing": round(len(bboxes) / max(elapsed, 1e-3), 2),
            "init_bbox": init_bbox,
            "output_video": output_video,
            "output_json": output_json,
        }

    def _draw(self, frame, bbox):
        import cv2  # noqa: WPS433
        x, y, w, h = bbox
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + w)), int(round(y + h))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    def _reply(self, out_data, payload: Dict) -> None:
        body = (json.dumps(payload) + chr(0)).encode("utf-8").strip()
        out_data.push_back(modelbox.Buffer(self.get_bind_device(), body))

    def _reply_error(self, out_data, msg: str) -> None:
        self._reply(out_data, {"error": msg})
