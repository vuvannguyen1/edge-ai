"""mAP@0.5 kiểu Ultralytics (nội suy 101 điểm). Dùng để đánh giá trên tập validation tách từ train."""
import numpy as np


def box_iou(a, b):
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None] - inter + 1e-9)


def compute_ap(recall, precision):
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy cũ chỉ có trapz
    return float(trapz(np.interp(x, mrec, mpre), x))


def map50(preds, gts, nc=7, iou_thr=0.5):
    """preds: list (n,6) [x1,y1,x2,y2,score,cls]; gts: list (m,5) [cls,x1,y1,x2,y2]. Cùng hệ toạ độ.
    -> (mAP, list AP từng lớp)"""
    recs = {c: [] for c in range(nc)}          # (score, is_tp)
    n_gt = np.zeros(nc, int)
    for p, g in zip(preds, gts):
        for c in range(nc):
            gc = g[g[:, 0] == c, 1:5]
            pc = p[p[:, 5] == c]
            n_gt[c] += len(gc)
            if not len(pc):
                continue
            pc = pc[np.argsort(-pc[:, 4])]
            tp = np.zeros(len(pc), bool)
            if len(gc):
                ious = box_iou(pc[:, :4], gc)
                used = np.zeros(len(gc), bool)
                for k in range(len(pc)):
                    cand = np.where((ious[k] >= iou_thr) & ~used)[0]
                    if cand.size:
                        m = cand[np.argmax(ious[k, cand])]
                        used[m] = True
                        tp[k] = True
            recs[c] += list(zip(pc[:, 4], tp))
    aps = []
    for c in range(nc):
        if n_gt[c] == 0:
            continue
        if not recs[c]:
            aps.append(0.0); continue
        r = np.asarray(sorted(recs[c], key=lambda t: -t[0]))
        tpc = np.cumsum(r[:, 1]); fpc = np.cumsum(1 - r[:, 1])
        aps.append(compute_ap(tpc / n_gt[c], tpc / (tpc + fpc)))
    return (float(np.mean(aps)) if aps else 0.0), aps
