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

import json
from typing import Any

import numpy as np
import _flowunit as modelbox


# --- Algorithmic helpers (testable without ModelBox) ---

def cross_confirm(ball_map: dict[int, tuple[float, float, float]],
                  hit_frame: int,
                  require_visual_confirm: bool) -> tuple[bool, str]:
    """Decide whether the ball trajectory near `hit_frame` is consistent
    with a real racquet contact.

    ball_map: {frame_idx: (cx, cy, peak)}; sentinels are (-1, -1, 0).
    Returns (ok, reason). reason="" on success; otherwise one of:
      "no_ball_near_hit", "trajectory_not_consistent", "visual_skipped".
    """
    def _valid(f):
        p = ball_map.get(f, (-1.0, -1.0, 0.0))
        return p if p[0] >= 0 else None

    pre_pts = [_valid(hit_frame + d) for d in (-3, -2, -1, 0)]
    post_pts = [_valid(hit_frame + d) for d in (0, 1, 2, 3)]
    pre_pts = [p for p in pre_pts if p is not None]
    post_pts = [p for p in post_pts if p is not None]

    if not pre_pts and not post_pts:
        return (not require_visual_confirm,
                "visual_skipped" if not require_visual_confirm
                else "no_ball_near_hit")
    if len(pre_pts) < 2 or len(post_pts) < 2:
        return (not require_visual_confirm,
                "visual_skipped" if not require_visual_confirm
                else "no_ball_near_hit")

    # Velocity estimates from first/last pair in each side.
    v_pre = (pre_pts[-1][0] - pre_pts[0][0], pre_pts[-1][1] - pre_pts[0][1])
    v_post = (post_pts[-1][0] - post_pts[0][0], post_pts[-1][1] - post_pts[0][1])
    dot = v_pre[0] * v_post[0] + v_pre[1] * v_post[1]
    mag_pre = max(1e-6, (v_pre[0] ** 2 + v_pre[1] ** 2) ** 0.5)
    mag_post = (v_post[0] ** 2 + v_post[1] ** 2) ** 0.5
    if dot < 0 or mag_post < 0.3 * mag_pre:
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


def assign_hitter(pose_map: dict[int, list[dict[str, Any]]],
                  ball: tuple[float, float, float],
                  hit_frame: int) -> dict[str, Any] | None:
    """Pick the track whose closer wrist (L or R) is nearest to `ball` at
    `hit_frame`. Look ±2 frames if the target frame is empty.

    Returns {track_id, dist, tie_resolved, frame_used} or None when no track
    can be located in [f-2, f+2].
    """
    bx, by, _ = ball
    poses = None
    frame_used = hit_frame
    for d in (0, -1, 1, -2, 2):
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

    def open(self, config):
        raise NotImplementedError("Filled in Task 6d")

    def process(self, data_context):
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def data_pre(self, data_context):
        return modelbox.Status()

    def data_post(self, data_context):
        return modelbox.Status()

    def close(self):
        return modelbox.Status()
