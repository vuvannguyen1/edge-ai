"""Phép thử khứ hồi target -> decode (Việc 4, 14/11). Bắt lỗi đảo trục x/y khi input không vuông. Chạy vài giây, CPU.

  python pipeline/det/test_roundtrip.py                       # 160, 256x192, 320x240 (+ 2 cấu hình có head stride 4)
  python pipeline/det/test_roundtrip.py --sizes 320x240 320x240@4,8,16     # "@" = EDGE_STRIDES

Với nhãn của --n ảnh train đầu tiên (không augmentation):
  1. encode_targets -> target từng mức;
  2. dựng logit "hoàn hảo": tại mỗi ô dương, nghịch đảo công thức decode để ra đúng box của target,
     obj và lớp đúng = +10, các lớp khác = -10; ô âm: mọi thứ = -10;
  3. chạy post.decode (tắt NMS để không lẫn với lỗi khác) -> box trên ảnh gốc 320x240.
Mỗi box nhãn phải có một box dự đoán cùng lớp với IoU > 0,95, trừ box mất hết ô vì bị box khác ghi đè.
Không ghi file nào. Mỗi kích thước chạy trong một tiến trình con riêng (EDGE_IMGSZ được đọc lúc import).
"""
import argparse, os, subprocess, sys
from pathlib import Path

import numpy as np

LOGIT = 10.0


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def perfect_logits(targets):
    from det.model import STRIDES, BASE
    outs = []
    for t, s, b in zip(targets, STRIDES, BASE):
        gh, gw = t.shape[:2]
        o = np.full(t.shape, -LOGIT, np.float32)
        r, q = np.nonzero(t[..., 4] > 0)                       # r = hàng (trục y), q = cột (trục x)
        x1, y1, x2, y2 = t[r, q, 0], t[r, q, 1], t[r, q, 2], t[r, q, 3]
        # nghịch đảo: cx = (q + 2*sig(tx) - 0.5)*s ; w = (2*sig(tw))^2 * b
        o[r, q, 0] = _logit(((x1 + x2) / 2 / s - q + 0.5) / 2)
        o[r, q, 1] = _logit(((y1 + y2) / 2 / s - r + 0.5) / 2)
        o[r, q, 2] = _logit(np.sqrt((x2 - x1) / b) / 2)
        o[r, q, 3] = _logit(np.sqrt((y2 - y1) / b) / 2)
        o[r, q, 4] = LOGIT
        o[r, q, 5 + np.argmax(t[r, q, 5:], -1)] = LOGIT
        outs.append(o)
    return outs


def iou(a, b):
    ix = np.clip(np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0]), 0, None)
    iy = np.clip(np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1]), 0, None)
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter + 1e-9)


def run_one(data, n, thr):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common import SRC_W, SRC_H, IMGSZ_STR, STRIDES_STR
    from det.data import read_label, to_input_boxes, encode_targets
    from det.post import decode

    ids = sorted(p.stem for p in (Path(data) / "train" / "images").glob("*.png"))[:n]
    n_box = n_over = n_ok = 0
    worst, fails = 1.0, []
    for i in ids:
        lab = read_label(Path(data) / "train" / "labels" / f"{i}.txt")
        if not len(lab):
            continue
        lab[:, [1, 3]] = np.clip(lab[:, [1, 3]], 0, SRC_W)
        lab[:, [2, 4]] = np.clip(lab[:, [2, 4]], 0, SRC_H)
        inp = to_input_boxes(lab)
        tg = encode_targets(inp)
        pred = decode(perfect_logits(tg), conf=0.5, iou=1.01, max_det=10_000, pre_nms=10_000, multi_label=True)
        for k in range(len(lab)):
            n_box += 1
            # box còn giữ ít nhất một ô ở một mức nào đó? (box nhỏ ghi đè box lớn trong encode_targets)
            kept = any(np.any((t[..., 4] > 0) & np.all(t[..., :4] == inp[k, 1:], -1)
                              & (t[..., 5 + int(inp[k, 0])] > 0)) for t in tg)
            if not kept:
                n_over += 1
                continue
            same = pred[pred[:, 5] == lab[k, 0]]
            best = float(iou(lab[k, 1:], same[:, :4]).max()) if len(same) else 0.0
            if best > thr:
                n_ok += 1
                worst = min(worst, best)
            else:
                fails.append((i, int(lab[k, 0]), np.round(lab[k, 1:], 1).tolist(), round(best, 3)))

    print(f"[{IMGSZ_STR} @{STRIDES_STR}] lưới {[t.shape[:2] for t in tg]}  |  {len(ids)} ảnh, {n_box} box nhãn, "
          f"{n_over} box bị ghi đè hết ô (bỏ qua), {n_ok} khớp (IoU nhỏ nhất {worst:.4f}), {len(fails)} hỏng")
    for f in fails[:10]:
        print("   HỎNG: ảnh %s lớp %d box %s IoU tốt nhất %.3f" % f)
    if n_over > 0.05 * n_box:
        print(f"   HỎNG: {n_over}/{n_box} box mất hết ô (> 5%) -> nghi encode_targets bỏ sót ô")
    ok = not fails and n_ok > 0 and n_over <= 0.05 * n_box
    print(f"[{IMGSZ_STR} @{STRIDES_STR}] =>", "ĐẠT" if ok else "KHÔNG ĐẠT")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    ap.add_argument("--n", type=int, default=50, help="số ảnh train đầu tiên dùng để thử")
    ap.add_argument("--iou", type=float, default=0.95)
    ap.add_argument("--sizes", nargs="+", default=["160", "256x192", "320x240", "160@4,8,16", "320x240@4,8,16"])
    ap.add_argument("--one", action="store_true", help="(nội bộ) chạy với EDGE_IMGSZ hiện tại")
    a = ap.parse_args()

    if a.one:
        sys.exit(0 if run_one(a.data, a.n, a.iou) else 1)
    bad = []
    for sz in a.sizes:
        size, _, strides = sz.partition("@")
        env = {**os.environ, "EDGE_IMGSZ": size, "EDGE_STRIDES": strides or "8,16", "PYTHONIOENCODING": "utf-8"}
        env.setdefault("KERAS_BACKEND", "tensorflow")
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        r = subprocess.run([sys.executable, __file__, "--one", "--data", a.data, "--n", str(a.n), "--iou", str(a.iou)],
                           env=env)
        if r.returncode:
            bad.append(sz)
    print("\n=> TẤT CẢ ĐẠT" if not bad else f"\n=> KHÔNG ĐẠT ở: {', '.join(bad)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
