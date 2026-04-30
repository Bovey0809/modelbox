#
# Copyright 2021 The Modelbox Project Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import numpy as np


def nms(boxes, scores, iou_thr):
    """Single-class NMS (Numpy)."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_thr)[0]
        order = order[inds + 1]
    return keep


def postprocess_anchor_free(predictions, num_classes, conf_thr, iou_thr,
                            scale_x, scale_y):
    """Decode + NMS for Ultralytics anchor-free output.

    Args:
        predictions: [N, 4 + num_classes] np.ndarray. First 4 columns are
            cxcywh in network-input pixel space; remaining columns are
            per-class scores (already passed through sigmoid by the
            Ultralytics export head).
        num_classes: number of class channels.
        conf_thr: minimum max-class score to keep.
        iou_thr: IoU threshold for NMS.
        scale_x, scale_y: factors to map back to original image. For the
            ModelBox resize flowunit (non-aspect-preserving stretch),
            scale_x = orig_w / net_w, scale_y = orig_h / net_h.

    Returns:
        np.ndarray of shape [K, 6]: x1,y1,x2,y2,score,label, in original
        image coordinates. None / empty if nothing passes.
    """
    if predictions.size == 0:
        return None

    boxes_cxcywh = predictions[:, :4]
    scores = predictions[:, 4 : 4 + num_classes]

    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]
    valid = cls_scores > conf_thr
    if not valid.any():
        return None

    boxes = boxes_cxcywh[valid]
    cls_scores = cls_scores[valid]
    cls_inds = cls_inds[valid]

    boxes_xyxy = np.empty_like(boxes)
    boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2.0) * scale_x
    boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2.0) * scale_y
    boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2.0) * scale_x
    boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2.0) * scale_y

    keep = nms(boxes_xyxy, cls_scores, iou_thr)
    if not keep:
        return None
    return np.concatenate(
        [boxes_xyxy[keep], cls_scores[keep, None], cls_inds[keep, None]], 1
    )


def draw_bbox(image, results):
    h, w, _ = image.shape
    for bbox in results:
        x1, y1, x2, y2, score, label = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(x2, w)
        y2 = min(y2, h)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label_str = f"{int(label)}:{score:.2f}"
        cv2.putText(
            image,
            label_str,
            (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
    return image
