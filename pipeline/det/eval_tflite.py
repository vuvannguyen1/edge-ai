"""Đo lại mAP50 val của file det_int8.tflite có sẵn trong một run. CHỈ ĐỌC, không ghi file nào.

  python pipeline/det/eval_tflite.py --data . --run runs/det_torch_base
  python pipeline/det/eval_tflite.py --data . --run runs/det_torch_base --iou 0.6     # so với ngưỡng NMS cũ

Dùng làm phép kiểm hồi quy: sửa code tiền/hậu xử lý xong chạy lại, con số phải giữ nguyên (hoặc thay đổi đúng như dự kiến).
Ảnh val lấy theo split.json của chính run đó; hậu xử lý theo dict KAGGLE trong det/post.py (có thể đè bằng tham số).
EDGE_IMGSZ phải khớp kích thước input của file .tflite.
"""
import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CLASSES, DET_W, DET_H, IMGSZ_STR
from det.data import read_image, read_label, preprocess
from det.post import decode, KAGGLE
from det.metrics import map50
from tflite_utils import TFLiteRunner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    ap.add_argument("--run", required=True, help="thư mục run có det_int8.tflite và split.json")
    ap.add_argument("--tflite", default="det_int8.tflite", help="tên file trong thư mục run")
    ap.add_argument("--iou", type=float, default=None, help="đè ngưỡng IoU của NMS (mặc định theo KAGGLE)")
    ap.add_argument("--conf", type=float, default=None, help="đè ngưỡng score (mặc định theo KAGGLE)")
    a = ap.parse_args()

    run, data = Path(a.run), Path(a.data)
    val_ids = json.load(open(run / "split.json"))["val_ids"]
    assert val_ids, f"{run}/split.json không có ảnh val (run train với --val-frac 0?)"
    post = {**KAGGLE, **{k: v for k, v in (("iou", a.iou), ("conf", a.conf)) if v is not None}}

    det = TFLiteRunner(run / a.tflite)
    shape = tuple(int(v) for v in det.inp["shape"][1:3])
    assert shape == (DET_H, DET_W), \
        f"File .tflite nhận ảnh {shape[1]}x{shape[0]} nhưng EDGE_IMGSZ đang là {IMGSZ_STR}: set EDGE_IMGSZ={shape[1]}x{shape[0]}"

    preds, gts = [], []
    for i in val_ids:
        img = read_image(data / "train" / "images" / f"{i}.png")
        preds.append(decode(det(preprocess(img).astype(np.float32) / 255.0), **post))
        gts.append(read_label(data / "train" / "labels" / f"{i}.txt"))
    m, aps = map50(preds, gts)

    print(f"run {run}  |  {len(val_ids)} ảnh val  |  input {shape[1]}x{shape[0]}  |  hậu xử lý {post}")
    for n, ap_ in zip(CLASSES, aps):
        print(f"  {n:<8} {ap_:.4f}")
    print(f"mAP50 INT8 = {m:.4f}")
    s = run / "summary.json"
    if s.exists() and "val_map50_int8" in (d := json.load(open(s))):
        print(f"(summary.json lúc export: {d['val_map50_int8']:.4f})")


if __name__ == "__main__":
    main()
