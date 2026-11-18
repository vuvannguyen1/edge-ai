"""Tạo submission.csv từ HAI FILE TFLite INT8 (đúng mô hình sẽ nạp lên ESP32-S3).

  python make_submission.py --data . --det runs/det/det_int8.tflite \
      --intent runs/intent/intent_int8.tflite --vocab runs/intent/vocab.json --out submission.csv

Giữ đúng thứ tự dòng của sample_submission.csv, rồi tự kiểm tra định dạng trước khi nộp.
"""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import INTENTS, normalize, SRC_W, SRC_H, DET_W, DET_H, IMGSZ_STR
from det.data import read_image, preprocess
from det.post import decode, KAGGLE
from intent.train_intent import encode_tokens
from tflite_utils import TFLiteRunner


def det_json(dets):
    return json.dumps([{"class_id": int(d[5]), "score": round(float(d[4]), 4),
                        "bbox": [round(float(v), 1) for v in d[:4]]} for d in dets], separators=(",", ":"))


def validate(path, sample):
    sub = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(sub.columns) == ["task_type", "sample_id", "prediction"], "sai tên cột"
    assert len(sub) == len(sample), "sai số dòng"
    assert (sub["task_type"].values == sample["task_type"].values).all() and \
           (sub["sample_id"].values == sample["sample_id"].values).all(), "ID/thứ tự không khớp sample_submission"
    assert not sub.duplicated(["task_type", "sample_id"]).any(), "trùng ID"
    n_box = 0
    for p in sub.loc[sub.task_type == "DETECTION", "prediction"]:
        arr = json.loads(p)
        assert isinstance(arr, list) and len(arr) <= 100
        for b in arr:
            assert set(b) == {"class_id", "score", "bbox"} and 0 <= b["class_id"] <= 6 and 0 <= b["score"] <= 1
            x1, y1, x2, y2 = b["bbox"]
            assert 0 <= x1 <= x2 <= SRC_W and 0 <= y1 <= y2 <= SRC_H
        n_box += len(arr)
    assert sub.loc[sub.task_type == "INTENT", "prediction"].isin(INTENTS).all(), "nhãn intent không hợp lệ"
    n_det = (sub.task_type == "DETECTION").sum()
    print(f"OK: {len(sub)} dòng, {n_det} ảnh ({n_box/max(n_det,1):.1f} box/ảnh), {len(sub)-n_det} câu hỏi")
    dist = sub.loc[sub.task_type == "INTENT", "prediction"].value_counts(normalize=True)
    print("Phân bố intent dự đoán trên test:", {k: f"{v:.1%}" for k, v in dist.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    ap.add_argument("--det", default="runs/det/det_int8.tflite")
    ap.add_argument("--intent", default="runs/intent/intent_int8.tflite")
    ap.add_argument("--vocab", default="runs/intent/vocab.json")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--nms-iou", type=float, default=None, help="đè ngưỡng IoU của NMS (mặc định theo KAGGLE trong det/post.py)")
    a = ap.parse_args()
    D = Path(a.data)
    post = {**KAGGLE, **({"iou": a.nms_iou} if a.nms_iou is not None else {})}
    print("Hậu xử lý detection:", post)

    sample = pd.read_csv(D / "sample_submission.csv", dtype=str, keep_default_na=False)
    questions = pd.read_csv(D / "intent_test.csv", dtype=str).set_index("question_id")["question"]
    vocab = json.load(open(a.vocab))
    det, intent = TFLiteRunner(a.det), TFLiteRunner(a.intent)
    h, w = (int(v) for v in det.inp["shape"][1:3])
    if (h, w) != (DET_H, DET_W):
        sys.exit(f"Model detection nhận input {w}x{h} (rộng x cao) nhưng EDGE_IMGSZ={IMGSZ_STR}. "
                 f"Chạy: set EDGE_IMGSZ={w}x{h}")

    rows = []
    for t, sid in zip(sample["task_type"], sample["sample_id"]):
        if t == "DETECTION":
            img = preprocess(read_image(D / "test" / "images" / f"{sid}.png")).astype(np.float32) / 255.0
            rows.append((t, sid, det_json(decode(det(img), **post))))
        else:
            ids = np.asarray(encode_tokens(normalize(questions[sid]).split(), vocab), np.int32)
            rows.append((t, sid, INTENTS[int(np.argmax(intent(ids)[0]))]))

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task_type", "sample_id", "prediction"])
        w.writerows(rows)
    validate(a.out, sample)


if __name__ == "__main__":
    main()
