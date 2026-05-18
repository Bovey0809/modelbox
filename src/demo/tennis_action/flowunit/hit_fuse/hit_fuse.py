#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
"""hit_fuse — cross-confirm + hitter assignment + T-frame keypoint window
assembly. Collapse on three input streams (hit_centers, ball_pos,
tracked_poses); expand on output (hit_window + hit_meta paired per hit,
plus dropped_hits collapsed once).

Built incrementally:
  Task 6a (this file initially): cross_confirm().
  Task 6b: assign_hitter().
  Task 6c: assemble_window().
  Task 6d: HitFuse.open() validation + Process()/data_post() wiring.
"""

from __future__ import annotations

import builtins
import json
from typing import Any

import numpy as np
import _flowunit as modelbox

try:
    import _tennis_store as _store
    _HAS_STORE = True
except ImportError:
    _store = None
    _HAS_STORE = False


# --- Pose feature engineering (mirrors action_recognition repo's
# utils/dataset.py::engineer_pose_features so the per-hit feature tensor
# matches what the trained ASFormer ONNX expects: (B, 189, T) channel-first.
# Without this asformer_infer_ort sees (T, 17, 3) raw keypoints and either
# crashes on shape mismatch or produces garbage logits. ---

_CONF_THRESHOLD = 0.1

_COCO_BONES = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 6), (11, 12), (0, 5), (0, 6),
]

_COCO_ANGLES = [
    (5, 7, 9), (6, 8, 10), (7, 5, 11), (8, 6, 12),
    (5, 11, 13), (6, 12, 14), (11, 13, 15), (12, 14, 16),
]

_FEATURE_DIM = 189  # 51 base + 34 rel + 34 vel + 34 acc + 28 bone + 8 angle


def engineer_pose_features(kpts: np.ndarray) -> np.ndarray:
    """Engineer ASFormer-input features from (T, 17, 3) COCO keypoints.

    Ports `action_recognition/utils/dataset.py::engineer_pose_features` —
    every derived feature is confidence-gated (zeroed when contributing
    joints have conf < CONF_THRESHOLD).

    Returns: (T, 189) float32.
    """
    T = kpts.shape[0]
    xy = kpts[:, :, :2]
    conf = kpts[:, :, 2]
    joint_valid = conf >= _CONF_THRESHOLD  # (T, 17)

    # 1. Base 51-d: gated (x, y, conf) flat.
    gated = kpts.copy()
    gated[conf < _CONF_THRESHOLD, :2] = 0.0
    base = gated.reshape(T, -1)

    # 2. Hip-relative 34-d.
    hip_center = (xy[:, 11:12, :] + xy[:, 12:13, :]) / 2
    relative = (xy - hip_center).reshape(T, -1)
    hip_valid = joint_valid[:, 11] & joint_valid[:, 12]
    rel_gate = (joint_valid & hip_valid[:, np.newaxis]).astype(np.float32)
    rel_gate = np.repeat(rel_gate[:, :, np.newaxis], 2, axis=2).reshape(T, -1)
    relative *= rel_gate

    # 3. Velocity 34-d.
    velocity = np.zeros_like(xy)
    velocity[1:] = xy[1:] - xy[:-1]
    velocity = velocity.reshape(T, -1)
    vel_gate = np.zeros((T, 17), dtype=np.float32)
    vel_gate[1:] = (joint_valid[1:] & joint_valid[:-1]).astype(np.float32)
    vel_gate = np.repeat(vel_gate[:, :, np.newaxis], 2, axis=2).reshape(T, -1)
    velocity *= vel_gate

    # 4. Acceleration 34-d.
    accel = np.zeros_like(xy)
    if T > 2:
        accel[2:] = xy[2:] - 2 * xy[1:-1] + xy[:-2]
    accel = accel.reshape(T, -1)
    acc_gate = np.zeros((T, 17), dtype=np.float32)
    if T > 2:
        acc_gate[2:] = (joint_valid[2:] & joint_valid[1:-1]
                        & joint_valid[:-2]).astype(np.float32)
    acc_gate = np.repeat(acc_gate[:, :, np.newaxis], 2, axis=2).reshape(T, -1)
    accel *= acc_gate

    # 5. Bone-vector 28-d.
    bones, bone_gates = [], []
    for j1, j2 in _COCO_BONES:
        bones.append(xy[:, j2] - xy[:, j1])
        bone_gates.append((joint_valid[:, j1] & joint_valid[:, j2])
                          .astype(np.float32))
    bone_feats = np.stack(bones, axis=1).reshape(T, -1)
    bone_gate = np.stack(bone_gates, axis=1)
    bone_gate = np.repeat(bone_gate[:, :, np.newaxis], 2, axis=2).reshape(T, -1)
    bone_feats *= bone_gate

    # 6. Joint-angle 8-d (normalised to [0, 1]).
    angles, angle_gates = [], []
    for j1, j2, j3 in _COCO_ANGLES:
        v1 = xy[:, j1] - xy[:, j2]
        v2 = xy[:, j3] - xy[:, j2]
        dot = (v1 * v2).sum(axis=1)
        norms = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-8
        cos_angle = np.clip(dot / norms, -1.0, 1.0)
        angles.append(np.arccos(cos_angle) / np.pi)
        angle_gates.append((joint_valid[:, j1] & joint_valid[:, j2]
                            & joint_valid[:, j3]).astype(np.float32))
    angle_feats = np.stack(angles, axis=1)
    angle_feats *= np.stack(angle_gates, axis=1)

    out = np.concatenate([base, relative, velocity, accel,
                          bone_feats, angle_feats], axis=1).astype(np.float32)
    assert out.shape == (T, _FEATURE_DIM), out.shape
    return out


# --- Algorithmic helpers (testable without ModelBox) ---

_CROSS_CONFIRM_WIN = 5      # ± frame window (was ±3)
_LOW_MOTION_PX = 5.0        # source-px magnitude below which the ball
                            # counts as "stopped" (handles serve toss
                            # → impact transition where v_pre ≈ 0).


def cross_confirm(ball_map: dict[int, tuple[float, float, float]],
                  hit_frame: int,
                  require_visual_confirm: bool) -> tuple[bool, str]:
    """Decide whether the ball trajectory near `hit_frame` is consistent
    with a real racquet contact.

    ball_map: {frame_idx: (cx, cy, peak)}; sentinels are (-1, -1, 0).
    Returns (ok, reason). reason="" on success; otherwise one of:
      "no_ball_near_hit", "trajectory_not_consistent", "visual_skipped".

    Behavior:
      * Window widened to ±5 frames on each side of the hit (was ±3) —
        ball recall on phone clips runs 5–50 %, so the smaller window
        was dropping most hits at "insufficient ball samples".
      * If both sides have at least 1 valid ball detection (but fewer
        than 2 per side after widening), soft-accept as
        `visual_skipped` rather than hard-drop. This was already the
        behaviour when `require_visual_confirm=False`; we extend it to
        the partial-data case under `require_visual_confirm=True`.
      * Trajectory check accepts a serve-style signature where v_pre is
        nearly stationary (`|v_pre| < _LOW_MOTION_PX`) and v_post is
        moving — the original check rejected serves because the ball
        toss has small pre-motion and the post-impact ball doesn't
        reverse direction.
    """
    def _valid(f):
        p = ball_map.get(f, (-1.0, -1.0, 0.0))
        return p if p[0] >= 0 else None

    win = _CROSS_CONFIRM_WIN
    pre_pts = [_valid(hit_frame + d) for d in range(-win, 1)]
    post_pts = [_valid(hit_frame + d) for d in range(0, win + 1)]
    pre_pts = [p for p in pre_pts if p is not None]
    post_pts = [p for p in post_pts if p is not None]

    if not pre_pts and not post_pts:
        # Zero balls anywhere in ±win frames — model can't see this clip
        # well. With require_visual_confirm we drop; without, fall back
        # to audio-only.
        return (not require_visual_confirm,
                "visual_skipped" if not require_visual_confirm
                else "no_ball_near_hit")
    if len(pre_pts) < 2 or len(post_pts) < 2:
        # We have some signal but not enough to compute velocity
        # reliably. Soft-accept as visual_skipped: downstream
        # `fused_conf` will use raw `audio_conf` instead of the boosted
        # value, so callers can distinguish.
        return (True, "visual_skipped")

    # Velocity estimates from first/last pair in each side.
    v_pre = (pre_pts[-1][0] - pre_pts[0][0], pre_pts[-1][1] - pre_pts[0][1])
    v_post = (post_pts[-1][0] - post_pts[0][0], post_pts[-1][1] - post_pts[0][1])
    dot = v_pre[0] * v_post[0] + v_pre[1] * v_post[1]
    mag_pre = (v_pre[0] ** 2 + v_pre[1] ** 2) ** 0.5
    mag_post = (v_post[0] ** 2 + v_post[1] ** 2) ** 0.5
    # Reversal (most hits) or slow-down (drop shots / hits into net).
    if dot < 0 or mag_post < 0.3 * max(1e-6, mag_pre):
        return (True, "")
    # Serve / stop-then-accelerate: ball was near-stationary before.
    if mag_pre < _LOW_MOTION_PX:
        return (True, "")
    return (False, "trajectory_not_consistent")


def _bbox_iou_simple(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


_ASSIGN_HITTER_WIN = 5  # ± frame search (was ±2)


def assign_hitter(pose_map: dict[int, list[dict[str, Any]]],
                  ball: tuple[float, float, float],
                  hit_frame: int) -> dict[str, Any] | None:
    """Pick the track whose closer wrist (L or R) is nearest to `ball` at
    `hit_frame`. Search ±5 frames outward from the hit (was ±2).

    Returns {track_id, dist, tie_resolved, frame_used} or None when no
    track can be located within ±_ASSIGN_HITTER_WIN frames. Widened
    because pose recall on phone clips can be patchy and the player is
    often missing from the exact hit frame even when present in
    neighbouring frames.
    """
    bx, by, _ = ball
    poses = None
    frame_used = hit_frame
    # Outward search: 0, ±1, ±2, ..., ±_ASSIGN_HITTER_WIN.
    offsets = [0]
    for d in range(1, _ASSIGN_HITTER_WIN + 1):
        offsets.extend([-d, d])
    for d in offsets:
        cand = pose_map.get(hit_frame + d)
        if cand:
            poses = cand
            frame_used = hit_frame + d
            break
    if not poses:
        return None

    def _wrist_dist(p):
        lw = p["kpts"][9]
        rw = p["kpts"][10]
        d_l = ((lw[0] - bx) ** 2 + (lw[1] - by) ** 2) ** 0.5
        d_r = ((rw[0] - bx) ** 2 + (rw[1] - by) ** 2) ** 0.5
        return min(d_l, d_r)

    dists = [(p["track_id"], _wrist_dist(p), p) for p in poses]
    dists.sort(key=lambda t: t[1])
    best_tid, best_dist, best_p = dists[0]

    tie_resolved = False
    if len(dists) > 1 and abs(dists[1][1] - best_dist) < 5.0:
        ball_box = [bx - 15, by - 15, bx + 15, by + 15]
        iou_best = _bbox_iou_simple(best_p["bbox"], ball_box)
        iou_alt = _bbox_iou_simple(dists[1][2]["bbox"], ball_box)
        if iou_alt > iou_best:
            best_tid, best_dist, best_p = dists[1]
            tie_resolved = True
        elif iou_alt == iou_best and dists[1][0] < best_tid:
            best_tid, best_dist, best_p = dists[1]
            tie_resolved = True

    return {"track_id": int(best_tid), "dist": float(best_dist),
            "tie_resolved": tie_resolved, "frame_used": int(frame_used)}


# Stubs to be filled in tasks 6c–6d.


def _find_track_kpts(poses: list[dict[str, Any]], track_id: int) \
        -> list[list[float]] | None:
    """Return the kpts list of `track_id` in `poses`, or None if absent."""
    for p in poses:
        if p["track_id"] == track_id:
            return p["kpts"]
    return None


def assemble_window(pose_map: dict[int, list[dict[str, Any]]],
                    track_id: int, hit_frame: int, T: int,
                    min_coverage: float, image_width: int,
                    image_height: int) -> tuple[np.ndarray, dict] | None:
    """Gather T keypoint frames of `track_id` centered on `hit_frame`,
    linear-interpolating short gaps and padding boundaries.

    Returns (np.ndarray [T, 17, 3] in normalized coords, meta) or None
    when coverage is below `min_coverage`.
    """
    half = T // 2
    f_start = hit_frame - half
    f_end = f_start + T  # exclusive

    raw: list[list[list[float]] | None] = []
    for f in range(f_start, f_end):
        poses = pose_map.get(f, [])
        raw.append(_find_track_kpts(poses, track_id))

    valid_count = sum(1 for k in raw if k is not None)
    # Leading Nones will be boundary-padded, so they count toward coverage;
    # trailing Nones represent genuine data loss and do not.
    first_valid_idx = next((i for i, k in enumerate(raw) if k is not None), -1)
    if first_valid_idx == -1:
        return None
    effective_count = valid_count + first_valid_idx
    if effective_count < int(min_coverage * T):
        return None

    boundary_padded = False

    # Forward-fill leading None entries from the first valid kpts.
    # first_valid_idx is already computed above (used in coverage check).
    if first_valid_idx > 0:
        boundary_padded = True
        src = raw[first_valid_idx]
        for i in range(first_valid_idx):
            raw[i] = [list(p) for p in src]

    # Back-fill trailing None entries from the last valid kpts.
    # Index of the last valid entry in raw (count from the right).
    last_rev = next((i for i, k in enumerate(reversed(raw)) if k is not None), -1)
    last_idx = T - 1 - last_rev
    if last_idx < T - 1:
        boundary_padded = True
        src = raw[last_idx]
        for i in range(last_idx + 1, T):
            raw[i] = [list(p) for p in src]

    # Interpolate interior gaps. After boundary padding, raw[0] and raw[-1]
    # are guaranteed non-None.
    i = 0
    while i < T:
        if raw[i] is not None:
            i += 1
            continue
        # Find the next valid frame.
        j = i + 1
        while j < T and raw[j] is None:
            j += 1
        if j >= T:
            break  # No further valid frame (shouldn't happen after back-fill).
        prev_idx = i - 1
        prev_k = raw[prev_idx]
        next_k = raw[j]
        span = j - prev_idx
        for fill_i in range(i, j):
            alpha = (fill_i - prev_idx) / span
            interp = []
            for ki in range(17):
                x = prev_k[ki][0] * (1 - alpha) + next_k[ki][0] * alpha
                y = prev_k[ki][1] * (1 - alpha) + next_k[ki][1] * alpha
                c = min(prev_k[ki][2], next_k[ki][2])
                interp.append([float(x), float(y), float(c)])
            raw[fill_i] = interp
        i = j

    # All entries should now be non-None.
    arr = np.zeros((T, 17, 3), dtype=np.float32)
    for ti, kpts in enumerate(raw):
        if kpts is None:
            # Should not occur; guard against runtime surprise.
            return None
        for ki, (x, y, c) in enumerate(kpts):
            arr[ti, ki, 0] = x / image_width
            arr[ti, ki, 1] = y / image_height
            arr[ti, ki, 2] = c

    return arr, {"boundary_padded": boundary_padded}


class HitFuse(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()
        self._ball: dict[int, tuple[float, float, float]] = {}
        self._poses: dict[int, list[dict[str, Any]]] = {}
        self._centers_raw: list[dict[str, Any]] | None = None

    def open(self, config):
        self.T = config.get_int("T", 32)
        # require_visual_confirm + min_window_coverage are stored as floats
        # for forward-compat (get_float on a bool key would still work via
        # _Cfg in tests but the real Config exposes get_bool).
        if hasattr(config, "get_bool"):
            self.require_visual_confirm = bool(
                config.get_bool("require_visual_confirm", True))
        else:
            self.require_visual_confirm = True
        self.min_window_coverage = config.get_float("min_window_coverage", 0.75)
        self.image_width = config.get_int("image_width", 1280)
        self.image_height = config.get_int("image_height", 720)
        meta_path = config.get_string("asformer_meta_path", "")
        if not meta_path:
            modelbox.error("hit_fuse: asformer_meta_path required")
            return modelbox.Status.StatusCode.STATUS_FAULT
        try:
            with builtins.open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except OSError as exc:
            modelbox.error(f"hit_fuse: cannot read asformer_meta_path: {exc}")
            return modelbox.Status.StatusCode.STATUS_FAULT
        if int(meta.get("T", -1)) != self.T:
            modelbox.error(
                f"hit_fuse: meta T={meta.get('T')} != configured T={self.T}")
            return modelbox.Status.StatusCode.STATUS_FAULT
        if int(meta.get("num_kpts", -1)) != 17:
            modelbox.error("hit_fuse: meta num_kpts must be 17")
            return modelbox.Status.StatusCode.STATUS_FAULT
        self._ball = {}
        self._poses = {}
        self._centers_raw = None
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def data_pre(self, data_context):
        # Reset per-stream state so multi-stream sessions don't carry over.
        self._ball = {}
        self._poses = {}
        self._centers_raw = None
        return modelbox.Status()

    def process(self, data_context):
        # All three inputs are collapsed streams (1 buffer/session each):
        #   - hit_centers  : produced by hit_centers_emitter (collapse=true)
        #   - ball_history : produced by ball_history_collector (collapse=true)
        #   - poses_history: produced by poses_history_collector (collapse=true)
        # Routing ball/pose data through the history collectors (instead of the
        # previous _tennis_store side-channel) makes ModelBox block this node
        # until the video branch has fully finished, fixing the audio-vs-video
        # race that previously caused all hits to drop with "no_ball_near_hit".
        for buf in data_context.input("hit_centers"):
            self._centers_raw = json.loads(bytes(buf.as_object()).decode("utf-8"))
        for buf in data_context.input("ball_history"):
            raw = json.loads(bytes(buf.as_object()).decode("utf-8"))
            # JSON keys come back as strings; coerce to int frame indices.
            self._ball = {int(k): tuple(v) for k, v in raw.items()}
        for buf in data_context.input("poses_history"):
            raw = json.loads(bytes(buf.as_object()).decode("utf-8"))
            self._poses = {int(k): v for k, v in raw.items()}
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def data_post(self, data_context):
        if self._centers_raw is None:
            self._centers_raw = []
        # ball/pose data is now delivered through process() via the
        # ball_history / poses_history collector ports (above); no need to
        # pull from _tennis_store anymore.
        confirmed, dropped = self.fuse_session(self._centers_raw, self._ball,
                                               self._poses)
        modelbox.warn(
            f"hit_fuse.process: confirmed={len(confirmed)} "
            f"dropped={len(dropped)} ball_frames={len(self._ball)} "
            f"pose_frames={len(self._poses)}")
        # Route dropped hits via _tennis_store rather than a direct port edge.
        # ModelBox's match-stream check deadlocks when one of action_sink's
        # inputs (dropped_hits = 1 buffer/session) has a different cardinality
        # than its siblings (hit_meta + logits = N buffers/session).
        if _HAS_STORE:
            _store.write_dropped(dropped)
        win_out = data_context.output("hit_window")
        meta_out = data_context.output("hit_meta")
        mask_out = data_context.output("mask")
        for hit_id, (arr, meta) in enumerate(confirmed):
            # arr is the (T, 17, 3) keypoint window assemble_window returned.
            # Engineer to (T, 189) — matches what ASFormer was trained on —
            # then transpose to (189, T) channel-first to match the ONNX
            # input layout: features ['batch', 189, 'time']. The flowunit
            # `asformer_infer_ort` reads buffer bytes into this shape
            # directly, so the on-wire layout must be (C, T) contiguous.
            feats = engineer_pose_features(arr)            # (T, 189)
            feats_ct = np.ascontiguousarray(feats.T)       # (189, T)
            wb = modelbox.Buffer(self.get_bind_device(), feats_ct.tobytes())
            wb.set("hit_id", int(hit_id))
            win_out.push_back(wb)
            # Mask is (T,) all-ones because engineer_pose_features fills
            # every frame (gating happens per-feature, not per-frame).
            mask_arr = np.ones(self.T, dtype=np.float32)
            mkb = modelbox.Buffer(self.get_bind_device(), mask_arr.tobytes())
            mkb.set("hit_id", int(hit_id))
            mask_out.push_back(mkb)
            meta["hit_id"] = int(hit_id)
            mb = modelbox.Buffer(self.get_bind_device(),
                                 json.dumps(meta).encode("utf-8"))
            mb.set("hit_id", int(hit_id))
            meta_out.push_back(mb)
        return modelbox.Status()

    def close(self):
        return modelbox.Status()

    def fuse_session(self, centers_raw, ball_map, pose_map):
        """Pure-function entry point used by tests and by data_post()."""
        confirmed: list[tuple[np.ndarray, dict[str, Any]]] = []
        dropped: list[dict[str, Any]] = []
        for entry in centers_raw:
            f = int(entry["frame_idx"])
            audio_conf = float(entry["audio_conf"])
            ok, reason = cross_confirm(ball_map, f, self.require_visual_confirm)
            if not ok:
                dropped.append({"audio_hit_frame_idx": f,
                                "audio_conf": audio_conf, "reason": reason})
                continue
            ball = ball_map.get(f, (-1.0, -1.0, 0.0))
            assign = assign_hitter(pose_map, ball, f)
            if assign is None:
                dropped.append({"audio_hit_frame_idx": f,
                                "audio_conf": audio_conf,
                                "reason": "no_player_at_hit"})
                continue
            window = assemble_window(pose_map, assign["track_id"], f, self.T,
                                     self.min_window_coverage,
                                     self.image_width, self.image_height)
            if window is None:
                dropped.append({"audio_hit_frame_idx": f,
                                "audio_conf": audio_conf,
                                "reason": "window_coverage_low"})
                continue
            arr, wmeta = window
            visual_agrees = reason != "visual_skipped"
            meta = {
                "frame_idx": f, "track_id": int(assign["track_id"]),
                "audio_conf": audio_conf,
                "fused_conf": (audio_conf + 1.0) / 2.0 if visual_agrees
                              else audio_conf,
                "ball_xy": [float(ball[0]), float(ball[1])],
                "tie_resolved": bool(assign["tie_resolved"]),
                "boundary_padded": bool(wmeta["boundary_padded"]),
                "visual_agrees": visual_agrees,
            }
            confirmed.append((arr, meta))
        return confirmed, dropped
