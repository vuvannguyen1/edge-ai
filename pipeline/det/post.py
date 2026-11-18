"""Decode + NMS bằng numpy (bản Python của phần hậu xử lý sẽ viết lại bằng C trên ESP32)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import SX, SY, SRC_W, SRC_H, DET_W, DET_H
from det.model import NUM_CLASSES, base_for

# Hai bộ ngưỡng: Kaggle tối ưu mAP (ngưỡng rất thấp, giữ nhiều box), chung kết tối ưu đếm đúng.
KAGGLE = dict(conf=0.001, iou=0.5, max_det=100, pre_nms=1000, multi_label=True)   # iou 0.6 -> 0.5 (14/11, theo EDA)
FINAL = dict(conf=0.35, iou=0.5, max_det=20)


def _sig(x):
    return 1 / (1 + np.exp(-x))


def nms(boxes, scores, iou_thr):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        r = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[r, 0]); yy1 = np.maximum(boxes[i, 1], boxes[r, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[r, 2]); yy2 = np.minimum(boxes[i, 3], boxes[r, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        a_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        a_r = (boxes[r, 2] - boxes[r, 0]) * (boxes[r, 3] - boxes[r, 1])
        order = r[inter / (a_i + a_r - inter + 1e-9) <= iou_thr]
    return np.asarray(keep, int)


def decode(outputs, conf=0.001, iou=0.6, max_det=100, pre_nms=1000, multi_label=True):
    """outputs: các mức [out_s8 (H/8, W/8, 12), out_s16 (H/16, W/16, 12)] (thêm out_s4 nếu có head stride 4), float
    (đã dequantize), 1 ảnh (160x160: 20x20 và 10x10; 320x240: 30x40 và 15x20, s4 là 60x80).
    Stride của mỗi mức suy từ kích thước lưới, nên thứ tự đầu ra và EDGE_STRIDES không ảnh hưởng.
    -> (n,6) [x1,y1,x2,y2 (pixel ảnh GỐC 320x240), score, class_id]"""
    all_b, all_s = [], []
    for o in outputs:
        gh, gw = o.shape[0], o.shape[1]                  # hàng = trục y, cột = trục x
        s = DET_H // gh
        assert s * gh == DET_H and s * gw == DET_W, f"lưới {gh}x{gw} không khớp input {DET_W}x{DET_H}"
        b = base_for(s)
        sg = _sig(o)
        j = np.arange(gw)[None, :]
        i = np.arange(gh)[:, None]
        cx = (j + 2 * sg[..., 0] - 0.5) * s
        cy = (i + 2 * sg[..., 1] - 0.5) * s
        w = (2 * sg[..., 2]) ** 2 * b
        h = (2 * sg[..., 3]) ** 2 * b
        all_b.append(np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1).reshape(-1, 4))
        all_s.append((sg[..., 4:5] * sg[..., 5:]).reshape(-1, NUM_CLASSES))
    boxes, scores = np.concatenate(all_b), np.concatenate(all_s)

    if multi_label:                                    # đa nhãn: một ô có thể ra nhiều lớp
        bi, ci = np.nonzero(scores > conf)
    else:                                              # mỗi ô chỉ lấy lớp có score cao nhất
        ci_all = scores.argmax(1)
        bi = np.nonzero(scores[np.arange(len(scores)), ci_all] > conf)[0]
        ci = ci_all[bi]
    if bi.size == 0:
        return np.zeros((0, 6), np.float32)
    sc = scores[bi, ci]
    if sc.size > pre_nms:
        top = np.argpartition(-sc, pre_nms)[:pre_nms]
        bi, ci, sc = bi[top], ci[top], sc[top]
    bx = boxes[bi]
    keep = nms(bx + ci[:, None] * 4096.0, sc, iou)[:max_det]   # NMS tách theo lớp bằng offset
    bx, sc, ci = bx[keep], sc[keep], ci[keep]

    bx[:, [0, 2]] = np.clip(bx[:, [0, 2]] / SX, 0, SRC_W)
    bx[:, [1, 3]] = np.clip(bx[:, [1, 3]] / SY, 0, SRC_H)
    return np.concatenate([bx, sc[:, None], ci[:, None]], 1).astype(np.float32)
