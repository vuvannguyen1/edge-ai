"""Tinh chỉnh hậu xử lý (NMS, ngưỡng score, đa nhãn) trên tập validation bằng file INT8 – KHÔNG cần train lại.

  python det/tune_post.py --data .. --run ../runs/det_194k

Chạy mô hình 1 lần trên ảnh validation (lấy từ split.json), lưu đầu ra thô, rồi thử nhiều cấu hình decode.
Cấu hình tốt nhất -> sửa dict KAGGLE trong det/post.py rồi tạo lại submission.
"""
import argparse, itertools, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CLASSES, DET_W, DET_H, IMGSZ_STR
from det.data import read_image, read_label, preprocess
from det.post import decode, KAGGLE
from det.metrics import map50
from tflite_utils import TFLiteRunner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="..")
    ap.add_argument("--run", default="../runs/det_194k")
    a = ap.parse_args()
    D, R = Path(a.data), Path(a.run)

    ids = json.load(open(R / "split.json"))["val_ids"]
    assert ids, "split.json không có ảnh validation (run này train với --val-frac 0?)"
    run = TFLiteRunner(R / "det_int8.tflite")
    h, w = (int(v) for v in run.inp["shape"][1:3])
    assert (h, w) == (DET_H, DET_W), f"File .tflite nhận ảnh {w}x{h} nhưng EDGE_IMGSZ đang là {IMGSZ_STR}: set EDGE_IMGSZ={w}x{h}"
    t0 = time.time()
    raw = [run(preprocess(read_image(D / "train" / "images" / f"{i}.png")).astype(np.float32) / 255.0) for i in ids]
    gts = [read_label(D / "train" / "labels" / f"{i}.txt") for i in ids]
    print(f"{len(ids)} ảnh validation, input {IMGSZ_STR}, suy luận INT8 {time.time()-t0:.0f}s")

    grid = {"iou": [0.45, 0.5, 0.55, 0.6, 0.65, 0.7], "conf": [0.001, 0.0001],
            "multi_label": [True, False], "pre_nms": [1000, 3000]}
    rows = []
    for iou, conf, ml, pre in itertools.product(*grid.values()):
        m, aps = map50([decode(o, conf=conf, iou=iou, max_det=100, pre_nms=pre, multi_label=ml) for o in raw], gts)
        rows.append((m, iou, conf, ml, pre, aps))
    rows.sort(key=lambda r: -r[0])

    cur = (KAGGLE["iou"], KAGGLE["conf"], KAGGLE["multi_label"], KAGGLE["pre_nms"])
    base = next(r for r in rows if (r[1], r[2], r[3], r[4]) == cur)
    print(f"\nHiện tại (KAGGLE trong det/post.py: iou={cur[0]}, conf={cur[1]}, multi_label={cur[2]}, "
          f"pre_nms={cur[3]}): mAP50 = {base[0]:.4f}")
    print("\nTop 8 cấu hình:")
    print(f"{'mAP50':>7} {'iou':>5} {'conf':>7} {'multi':>6} {'pre_nms':>8}")
    for m, iou, conf, ml, pre, _ in rows[:8]:
        print(f"{m:7.4f} {iou:5.2f} {conf:7.4f} {str(ml):>6} {pre:8d}")
    best = rows[0]
    print(f"\nTốt nhất hơn hiện tại {100*(best[0]-base[0]):+.2f} điểm mAP "
          f"(= {70*(best[0]-base[0]):+.2f} điểm Kaggle).")
    print("AP từng lớp (tốt nhất):", {c: round(x, 4) for c, x in zip(CLASSES, best[5])})
    print(f"\nSửa det/post.py:  KAGGLE = dict(conf={best[2]}, iou={best[1]}, max_det=100, pre_nms={best[4]}, multi_label={best[3]})")


if __name__ == "__main__":
    main()
