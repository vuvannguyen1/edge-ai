"""Phép thử nhanh đường train distillation (Việc 2, 14/11). Chạy khoảng 1-2 phút trên GPU.

Với kd_weight = 0, loss distill chỉ còn loss nhãn thật, nên train student qua build_distill_trainer phải cho ĐÚNG
trọng số như train student thường. Nếu lệch -> đường train distill còn lỗi (như lỗi gradient cộng dồn đã gặp).

  set KERAS_BACKEND=torch
  .venv-torch\\Scripts\\python pipeline/det/test_kd_equiv.py --teacher runs/teacher_resume/teacher.keras

  --buggy : dựng mô hình fit bằng keras.Model thường (không có DistillTrainer.zero_grad) để chứng minh phép thử
            bắt được lỗi cũ. Kỳ vọng: KHÔNG ĐẠT.
Không augmentation, không EMA, chỉ đọc --limit ảnh đầu của train/. Không ghi file nào.
"""
import argparse, os, sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
import numpy as np
import keras

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from det.model import build_model, STRIDES, BASE
from det.data import load_train, DetSeq
from det.loss import make_level_loss, make_distill_loss
from det.train_det import build_distill_trainer


def max_diff(ws_a, ws_b):
    return max(float(np.abs(a - b).max()) for a, b in zip(ws_a, ws_b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    ap.add_argument("--teacher", default="runs/teacher_resume/teacher.keras")
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--buggy", action="store_true", help="tái hiện lỗi cũ: fit bằng keras.Model thường")
    a = ap.parse_args()

    print(f"Backend Keras: {keras.backend.backend()}")
    if keras.backend.backend() == "torch":
        import torch
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
        print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "không có (chạy CPU)")
    keras.utils.set_random_seed(a.seed)

    ids, imgs, labs = load_train(a.data)
    imgs, labs = imgs[:a.limit], labs[:a.limit]
    seq_a = DetSeq(imgs, labs, a.batch, train=False, mode="student")
    seq_b = DetSeq(imgs, labs, a.batch, train=False, mode="distill")

    # Thứ tự dựng giống train_det.main (student trước, rồi nạp teacher) để tên biến trùng y như khi train thật.
    stu_b = build_model()
    teacher = keras.models.load_model(a.teacher, compile=False)
    stu_a = build_model()
    stu_a.set_weights(stu_b.get_weights())
    w0 = [w.copy() for w in stu_b.get_weights()]

    opt = lambda: keras.optimizers.AdamW(a.lr, weight_decay=5e-4, clipnorm=10.0)
    stu_a.compile(optimizer=opt(), loss=[make_level_loss(s, b) for s, b in zip(STRIDES, BASE)])
    trainer = build_distill_trainer(stu_b, teacher, cls=keras.Model if a.buggy else None)
    trainer.compile(optimizer=opt(), loss=[make_distill_loss(s, b, 0.0) for s, b in zip(STRIDES, BASE)])
    print(f"Mô hình fit khi distill: {type(trainer).__name__}  ({'tái hiện lỗi cũ' if a.buggy else 'bản đã sửa'})")
    print(f"{a.limit} ảnh, batch {a.batch}, {a.steps} bước, kd_weight 0\n")

    print(f"{'bước':>4}  {'loss thường':>12}  {'loss distill':>12}  {'lệch trọng số':>14}")
    for k in range(a.steps):
        i = k % len(seq_a)
        xa, ya = seq_a[i]
        (xs, xh), yb = seq_b[i]
        assert np.array_equal(xa, xs), "hai bộ dữ liệu không cùng ảnh"
        la = stu_a.train_on_batch(xa, ya, return_dict=True)["loss"]
        lb = trainer.train_on_batch([xs, xh], yb, return_dict=True)["loss"]
        if k < 5 or (k + 1) % 5 == 0:
            print(f"{k + 1:>4}  {float(la):>12.5f}  {float(lb):>12.5f}  {max_diff(stu_a.get_weights(), stu_b.get_weights()):>14.2e}")

    d = max_diff(stu_a.get_weights(), stu_b.get_weights())
    moved = max_diff(stu_a.get_weights(), w0)
    print(f"\nTrọng số đã thay đổi so với lúc đầu: {moved:.3e} (phải > 0, nếu không là chưa train gì)")
    print(f"Lệch tuyệt đối lớn nhất giữa student thường và student distill: {d:.3e}  (ngưỡng {a.tol:g})")
    ok = d < a.tol and moved > 0
    print("=> ĐẠT: đường train distill đúng." if ok else "=> KHÔNG ĐẠT: đường train distill vẫn khác train thường.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
