"""Export TFLite INT8 từ một run đã train + đo mAP50 của chính file INT8. Luôn chạy trên backend TensorFlow (CPU).

train_det.py tự gọi bước này sau khi train. Khi train bằng backend PyTorch (GPU trên Windows), nó được chạy ở
tiến trình riêng với KERAS_BACKEND=tensorflow. Chạy tay khi cần export lại:

  set KERAS_BACKEND=tensorflow && python det/export_det.py --data . --run runs/det
"""
import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
import numpy as np
import keras

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import IMGSZ_STR, run_meta
from det.data import load_train, preprocess, read_label
from det.post import decode, KAGGLE
from det.metrics import map50


def export_run(data, run, seed=42, limit=0, extra=None, ids=None, imgs=None):
    assert keras.backend.backend() == "tensorflow", "export cần KERAS_BACKEND=tensorflow"
    from tflite_utils import export_det_int8, list_ops, TFLiteRunner
    run = Path(run)
    if imgs is None:
        ids, imgs, _ = load_train(data)
        if limit:
            ids, imgs = ids[:limit], imgs[:limit]
    val_ids = json.load(open(run / "split.json"))["val_ids"]
    n_tr = len(ids) - len(val_ids)

    model = keras.models.load_model(run / ("best.keras" if val_ids else "last.keras"), compile=False)
    shape = "x".join(str(v) for v in model.input_shape[2:0:-1])
    assert shape == IMGSZ_STR, f"Mô hình trong {run} nhận ảnh {shape} nhưng EDGE_IMGSZ đang là {IMGSZ_STR}: set EDGE_IMGSZ={shape}"
    rng = np.random.default_rng(0)                         # cùng cách chọn ảnh đại diện như các run trước
    rep = np.asarray([preprocess(imgs[k]) for k in rng.choice(n_tr, min(300, n_tr), replace=False)])
    tfl = export_det_int8(model, rep, run / "det_int8.tflite")
    summary = {"imgsz": IMGSZ_STR, **run_meta(), "params": model.count_params(), "tflite_kb": len(tfl) / 1024,
               "ops": list_ops(tfl)}
    summary.update(extra or {})                            # train_det.py truyền git_commit/argv của lệnh train, đè lên

    if val_ids:
        by_id = dict(zip(ids, imgs))
        runner = TFLiteRunner(tfl)
        preds = [decode(runner(preprocess(by_id[i]).astype(np.float32) / 255.0), **KAGGLE) for i in val_ids]
        gts = [read_label(Path(data) / "train" / "labels" / f"{i}.txt") for i in val_ids]
        m_q, aps_q = map50(preds, gts)
        summary.update(val_map50_int8=m_q, ap_int8=[round(x, 4) for x in aps_q])
    print(json.dumps(summary, indent=1))
    json.dump(summary, open(run / "summary.json", "w"), indent=1)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--extra", default="", help="JSON gộp thêm vào summary.json (mặc định: train_meta.json của run nếu có)")
    a = ap.parse_args()
    meta = Path(a.run) / "train_meta.json"                 # do train_det.py --no-export ghi (run train ở Kaggle)
    extra = json.loads(a.extra) if a.extra else (json.load(open(meta)) if meta.exists() else {})
    export_run(a.data, a.run, a.seed, a.limit, extra)
