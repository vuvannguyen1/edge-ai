"""Huấn luyện TEACHER cho distillation (chỉ dùng lúc train student, không nạp lên ESP32).

  python det/train_teacher.py --data . --out runs/teacher --epochs 40

Teacher = MobileNetV2 pretrained ImageNet + head cùng định dạng student, nhìn ảnh 2x (xem det/teacher.py).
Dùng ĐÚNG phép chia val của train_det.py (cùng --seed, --val-frac) để teacher không thấy ảnh val của student;
nếu khác đi, mAP val của student khi distill sẽ bị ảo. Không dùng thư mục test/.
Kết quả: <out>/teacher.keras (bản EMA tốt nhất trên val), <out>/summary.json.
Sau đó:  python det/train_det.py --data . --out runs/det_kd --teacher runs/teacher/teacher.keras
"""
import argparse, json, shutil, sys, time
from pathlib import Path

import numpy as np
import keras

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from det.model import STRIDES, BASE
from det.teacher import build_teacher
from det.data import load_train, DetSeq, preprocess_hires
from det.loss import make_level_loss
from det.train_det import EMAEval, gpu_peak_report
from common import IMGSZ_STR, CLASSES, run_meta


class ShiftedSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """Lịch learning rate gốc nhưng bắt đầu từ bước `offset` (dùng khi train tiếp từ giữa chừng)."""
    def __init__(self, base, offset):
        self.base, self.offset = base, int(offset)

    def __call__(self, step):
        return self.base(step + self.offset)

    def get_config(self):
        return {"base": keras.saving.serialize_keras_object(self.base), "offset": self.offset}

    @classmethod
    def from_config(cls, config):
        return cls(keras.saving.deserialize_keras_object(config["base"]), config["offset"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    ap.add_argument("--out", default="runs/teacher")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--alpha", type=float, default=1.0, help="độ rộng MobileNetV2 (0.35/0.5/0.75/1.0/1.3/1.4)")
    ap.add_argument("--val-frac", type=float, default=0.1)       # PHẢI giống train_det.py
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--empty-p", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)              # PHẢI giống train_det.py
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="chỉ dùng N ảnh đầu (thử nhanh)")
    ap.add_argument("--resume", default="", help="file .keras (thường là last.keras) để train tiếp; dùng --out KHÁC "
                                                 "thư mục cũ để không ghi đè best.keras đã có")
    ap.add_argument("--initial-epoch", type=int, default=0, help="số epoch đã train của file --resume")
    ap.add_argument("--init-weights", default="",
                    help="file .keras của teacher đã train: khởi tạo trọng số từ đó (thay ImageNet), lịch lr chạy lại từ "
                         "đầu. Teacher tích chập hoàn toàn nên nạp được cả từ teacher của EDGE_IMGSZ khác")
    a = ap.parse_args()
    assert not (a.resume and a.init_weights), "--resume và --init-weights không dùng cùng lúc"
    assert not a.resume or 0 < a.initial_epoch < a.epochs, "--resume cần --initial-epoch trong khoảng (0, --epochs)"
    assert not a.resume or Path(a.resume).resolve().parent != Path(a.out).resolve(), \
        "--out phải khác thư mục chứa file --resume, nếu không best.keras cũ sẽ bị ghi đè"

    keras.utils.set_random_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ids, imgs, labs = load_train(a.data)
    if a.limit:
        ids, imgs, labs = ids[:a.limit], imgs[:a.limit], labs[:a.limit]
    print(f"Đọc {len(ids)} ảnh trong {time.time()-t0:.0f}s")

    perm = np.random.default_rng(a.seed).permutation(len(ids))    # giống hệt train_det.main
    n_val = int(len(ids) * a.val_frac)
    va, tr = perm[:n_val], perm[n_val:]
    json.dump({"val_ids": [ids[k] for k in va]}, open(out / "split.json", "w"))
    tr_imgs, tr_labs = [imgs[k] for k in tr], [labs[k] for k in tr]
    va_imgs, va_labs = [imgs[k] for k in va], [labs[k] for k in va]

    src_file = a.resume or a.init_weights
    model = build_teacher(alpha=a.alpha, weights=None if (a.no_pretrained or src_file) else "imagenet")
    if src_file:
        src = keras.models.load_model(src_file, compile=False)
        model.set_weights(src.get_weights())
        del src
        print(f"Train tiếp từ {a.resume}, đã xong {a.initial_epoch}/{a.epochs} epoch" if a.resume
              else f"Khởi tạo trọng số từ {a.init_weights}")
    t_h, t_w = model.input_shape[1:3]
    print(f"Teacher: {model.count_params():,} tham số, input {t_w}x{t_h}, student input {IMGSZ_STR} (rộng x cao)")
    seq = DetSeq(tr_imgs, tr_labs, a.batch, train=True, empty_p=a.empty_p, seed=a.seed, mode="teacher",
                 workers=a.workers, use_multiprocessing=False)
    steps = len(seq) * a.epochs
    warm = min(len(seq) * 2, steps // 10)
    sched = keras.optimizers.schedules.CosineDecay(a.lr * 0.01, steps - warm, alpha=0.01,
                                                   warmup_target=a.lr, warmup_steps=warm)
    done = a.initial_epoch * len(seq)                 # số bước đã train trước đó (cần cùng --batch, --epochs như lần đầu)
    if a.resume:
        sched = ShiftedSchedule(sched, done)          # learning rate đi tiếp đúng chỗ đã dừng
        seq.epoch = a.initial_epoch                   # augmentation không lặp lại các epoch đầu
    model.compile(optimizer=keras.optimizers.AdamW(sched, weight_decay=a.wd, clipnorm=10.0),
                  loss=[make_level_loss(s, b) for s, b in zip(STRIDES, BASE)])
    print(f"Backend Keras: {keras.backend.backend()}")
    cb = EMAEval(va_imgs, va_labs, out, every=a.eval_every, prep=preprocess_hires, eval_batch=min(a.batch, 16))
    if a.resume:
        cb.updates = done                             # EMA giữ hệ số trung bình như đang ở giữa quá trình train
    model.fit(seq, epochs=a.epochs, initial_epoch=a.initial_epoch, callbacks=[cb], verbose=2)
    gpu = gpu_peak_report()

    shutil.copyfile(out / ("best.keras" if n_val else "last.keras"), out / "teacher.keras")
    best = max((r for r in cb.log if "map50" in r), key=lambda r: r["map50"], default=None)
    summary = {"imgsz": IMGSZ_STR, **run_meta(), "imgsz_student": IMGSZ_STR, "imgsz_teacher": f"{t_w}x{t_h}",
               "params": model.count_params(), "backbone": f"MobileNetV2 alpha={a.alpha}",
               "pretrained": None if a.no_pretrained else "imagenet (keras.applications)",
               "init_weights": a.init_weights or None, "resume": a.resume or None,
               "seed": a.seed, "val_frac": a.val_frac, "batch": a.batch, "lr": a.lr, "epochs": a.epochs,
               **(gpu or {}),
               "val_map50": best["map50"] if best else None, "best_epoch": best["epoch"] if best else None,
               "ap": dict(zip(CLASSES, best["ap"])) if best else None}
    print(json.dumps(summary, indent=1))
    json.dump(summary, open(out / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
