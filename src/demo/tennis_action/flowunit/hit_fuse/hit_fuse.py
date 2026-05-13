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


# Stubs to be filled in tasks 6b–6d.
def assign_hitter(*a, **k):
    raise NotImplementedError("Filled in Task 6b")


def assemble_window(*a, **k):
    raise NotImplementedError("Filled in Task 6c")


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
