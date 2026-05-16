#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
"""stream_discard — silent sink that accepts and discards all input buffers.

Satisfies ModelBox's requirement that every declared output port is connected,
for branches whose data is consumed via the _tennis_store side-channel rather
than direct graph edges (e.g. ball_pos, tracked_poses).
"""

from __future__ import annotations

import _flowunit as modelbox


class StreamDiscard(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()

    def open(self, config):
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def process(self, data_context):
        # Drain and discard all incoming buffers.
        for _buf in data_context.input("in_data"):
            pass
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def close(self):
        return modelbox.Status()
