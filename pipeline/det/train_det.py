"""Huấn luyện detection.

  python det/train_det.py --data . --out runs/det --epochs 200

Quy trình: tách 10% train làm validation (train/test cùng phân phối nên đủ tin cậy) -> train với augmentation,
EMA trọng số -> đánh giá mAP50 trên val mỗi --eval-every epoch, lưu bản tốt nhất ->
export TFLite INT8 -> đo lại mAP50 bằng chính file INT8 (con số sát với Kaggle và chung kết nhất).
Chạy với --val-frac 0 để train trên toàn bộ dữ liệu cho bản nộp cuối (dùng số epoch đã chọn).

Distillation (tùy chọn): train teacher trước bằng det/train_teacher.py rồi thêm --teacher runs/teacher/teacher.keras.
Teacher chỉ dùng lúc train; file export vẫn là student 1 mô hình, <= 200.000 tham số.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

import numpy as np
import keras

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from det.model import build_model, parse_arch, STRIDES, BASE
KD_STRIDES = (8, 16)      # các mức teacher có; mức khác (stride 4) chỉ học nhãn thật khi distill
from det.loss import scale_cls_weight
from det.data import load_train, DetSeq, preprocess, to_input_boxes
from det.loss import make_level_loss, make_distill_loss
from det.post import decode, KAGGLE
from det.metrics import map50
from common import IMGSZ_STR, STRIDES_STR, DET_W, DET_H, run_meta
# tflite_utils (cần TensorFlow) chỉ import ở bước export, để tiến trình train trên backend torch không nạp TensorFlow.


def gpu_peak_report():
    """In bộ nhớ GPU cao nhất PyTorch đã giữ trong lần train này -> dict cho summary (None nếu không có GPU torch).
    Trên Windows, khi vượt VRAM thật, driver NVIDIA lấn sang RAM hệ thống (sysmem fallback) thay vì báo hết bộ nhớ,
    train chậm khoảng 10 lần -> đỉnh sát hoặc vượt VRAM là dấu hiệu phải giảm batch."""
    if keras.backend.backend() != "torch":
        return None
    import torch
    if not torch.cuda.is_available():
        return None
    peak = torch.cuda.max_memory_reserved() / 2**20
    total = torch.cuda.get_device_properties(0).total_memory / 2**20
    ok = peak <= 0.9 * total
    print(f"GPU: đỉnh bộ nhớ PyTorch giữ {peak:.0f} MB / VRAM {total:.0f} MB -> "
          + ("ỔN" if ok else "QUÁ SÁT hoặc VƯỢT VRAM (có thể đã lấn sang RAM hệ thống): giảm --batch"), flush=True)
    return {"gpu_peak_mb": round(peak), "gpu_total_mb": round(total)}


class EMAEval(keras.callbacks.Callback):
    """EMA trọng số (như YOLOv5) + đánh giá mAP50 trên val bằng trọng số EMA, lưu bản tốt nhất."""
    def __init__(self, val_imgs, val_labs, out, every=5, decay=0.9995, tau=1000, net=None, prep=preprocess,
                 eval_batch=None, patience=0, stop_after=0):
        """net: mô hình cần theo dõi/lưu nếu khác mô hình đang fit (khi distill, mô hình fit là student + teacher).
        prep: hàm tiền xử lý ảnh val (teacher dùng preprocess_hires).
        eval_batch: batch khi suy luận trên val (không ảnh hưởng kết quả). None = 64 ảnh 160x160, giảm theo diện tích
            input (320x240 -> 21): batch 64 ở 320x240 làm tràn VRAM 4 GB sang RAM, train chậm 10 lần.
        patience: dừng sớm khi mAP50 val không lập kỷ lục sau N lần đo liên tiếp (0 = tắt).
        stop_after: dừng sau N epoch nhưng vẫn giữ lịch learning rate của tổng số epoch (0 = tắt)."""
        super().__init__()
        self._net, self.eval_batch = net, eval_batch or max(4, 64 * 160 * 160 // (DET_W * DET_H))
        self.patience, self.stop_after, self.bad = patience, stop_after, 0
        self.vx = np.asarray([prep(im) for im in val_imgs], np.float32) / 255.0 if len(val_imgs) else None
        self.vg = val_labs
        self.out, self.every, self.decay, self.tau = out, every, decay, tau
        self.updates, self.ema, self.best = 0, None, -1.0
        self.log = []

    @property
    def net(self):
        return self._net if self._net is not None else self.model

    def on_train_batch_end(self, batch, logs=None):
        w = self.net.get_weights()
        if self.ema is None:
            self.ema = [x.copy() for x in w]
            return
        self.updates += 1
        d = self.decay * (1 - np.exp(-self.updates / self.tau))
        for e, x in zip(self.ema, w):
            e *= d
            e += (1 - d) * x

    @staticmethod
    def _free_gpu():
        if keras.backend.backend() == "torch":
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def evaluate(self):
        self._free_gpu()                                   # trả bộ nhớ của bước train trước khi suy luận trên val
        outs = self.net.predict(self.vx, batch_size=self.eval_batch, verbose=0)
        self._free_gpu()                                   # và trả bộ nhớ của bước suy luận trước khi train tiếp
        outs = outs if isinstance(outs, (list, tuple)) else [outs]
        # đủ MỌI mức đầu ra (trước đây chỉ lấy 2 mức đầu -> mô hình có head s4 bị bỏ sót s16 khi chấm val)
        preds = [decode([o[k] for o in outs], **KAGGLE) for k in range(len(self.vx))]
        return map50(preds, self.vg)

    def on_epoch_end(self, epoch, logs=None):
        stop = bool(self.stop_after) and epoch + 1 >= self.stop_after
        last = epoch + 1 == self.params["epochs"] or stop
        if (epoch + 1) % self.every and not last:
            return
        raw = self.net.get_weights()
        self.net.set_weights(self.ema)
        row = {"epoch": epoch + 1, "loss": float(logs.get("loss", np.nan))}
        if self.vx is not None:
            m, aps = self.evaluate()
            row.update(map50=m, ap=[round(a, 4) for a in aps])
            if m > self.best:
                self.best, self.bad = m, 0
                self.net.save(self.out / "best.keras")
            else:
                self.bad += 1
            print(f"  epoch {epoch+1}: loss {row['loss']:.4f}  val mAP50 {m:.4f}  (best {self.best:.4f})", flush=True)
        self.net.save(self.out / "last.keras")    # luôn lưu bản EMA mới nhất
        self.net.set_weights(raw)
        self.log.append(row)
        json.dump(self.log, open(self.out / "log.json", "w"), indent=1)
        if stop:
            print(f"  Dừng theo --stop-after ở epoch {epoch+1}", flush=True)
            self.model.stop_training = True
        elif self.patience and self.bad >= self.patience:
            print(f"  Dừng sớm ở epoch {epoch+1}: mAP50 val không cải thiện sau {self.bad} lần đo "
                  f"(best {self.best:.4f})", flush=True)
            self.model.stop_training = True


class DistillTrainer(keras.Model):
    """Mô hình để fit khi distill = student + teacher đóng băng.

    Trên backend torch, Keras xoá gradient mỗi bước bằng Module.zero_grad(), tức chỉ xoá các biến có trong
    parameters() của mô hình đang fit. Danh sách đó được lập theo TÊN biến, nên biến của student trùng tên với biến
    của teacher (hai head đều tên out_s8 / out_s16) bị teacher đè mất -> gradient của head student không bao giờ
    được xoá, cộng dồn qua mọi bước và làm hỏng việc train. Ở đây xoá trực tiếp trên mọi biến train được."""
    def zero_grad(self, set_to_none=True):
        for v in self.trainable_variables:
            v.value.grad = None


class ChunkedTeacher(keras.layers.Layer):
    """Teacher đóng băng chạy theo từng nhóm `chunk` ảnh, trong torch.no_grad (chỉ backend torch).

    Mô hình Functional của Keras giữ mọi tensor trung gian tới hết lượt gọi, kể cả khi không cần gradient:
    MobileNetV2 ở 640x480 tốn khoảng 550 MB/ảnh, nên gọi cả batch 16 một lần cần khoảng 9 GB (vượt xa 4 GB VRAM,
    driver lấn sang RAM hệ thống, chậm khoảng 20 lần). Chia nhóm thì bộ nhớ trung gian được trả sau mỗi nhóm.
    Kết quả giống hệt gọi cả batch (BatchNorm ở chế độ suy luận, mỗi ảnh tính độc lập)."""
    def __init__(self, teacher, chunk, **kw):
        super().__init__(name="teacher_chunked", **kw)
        self.teacher, self.chunk = teacher, int(chunk)
        self.built = True

    def call(self, x):
        if keras.backend.backend() != "torch" or self.chunk <= 0:
            return self.teacher(x, training=False)
        import torch
        with torch.no_grad():
            parts = [self.teacher(x[i:i + self.chunk], training=False) for i in range(0, x.shape[0], self.chunk)]
        return [torch.cat([p[k] for p in parts]) for k in range(len(parts[0]))]

    def compute_output_spec(self, x):
        return self.teacher.compute_output_spec(x)


def build_distill_trainer(model, teacher, cls=None, teacher_chunk=0):
    """-> mô hình 2 đầu vào [ảnh student, ảnh lớn], mỗi đầu ra = concat[student, teacher] cho make_distill_loss.
    cls: lớp mô hình (mặc định DistillTrainer); test_kd_equiv.py --buggy truyền keras.Model để tái hiện lỗi cũ.
    teacher_chunk > 0: teacher chạy theo nhóm ảnh nhỏ để tiết kiệm VRAM (xem ChunkedTeacher); 0 = gọi cả batch như cũ."""
    cls = cls or DistillTrainer
    teacher.trainable = False
    # Teacher chỉ có 2 mức (8, 16). Student có head stride 4 (Việc 10): s8/s16 distill, s4 chỉ học nhãn thật.
    s_shape = dict(zip(STRIDES, [tuple(o.shape[1:]) for o in model.outputs]))
    assert [tuple(o.shape[1:]) for o in teacher.outputs] == [s_shape[s] for s in KD_STRIDES], \
        "Lưới đầu ra teacher không khớp student (teacher train với EDGE_IMGSZ khác?)"
    x_s = keras.Input(model.input_shape[1:], name="image")
    x_t = keras.Input(teacher.input_shape[1:], name="image_hires")
    t_call = ChunkedTeacher(teacher, teacher_chunk) if teacher_chunk > 0 else (lambda x: teacher(x, training=False))
    s_out, t_out = model(x_s), t_call(x_t)
    t_of = dict(zip(KD_STRIDES, t_out))
    trainer = cls([x_s, x_t], [keras.layers.Concatenate(name=f"kd_s{st}")([so, t_of[st]]) if st in t_of
                               else keras.layers.Identity(name=f"hard_s{st}")(so)
                               for st, so in zip(STRIDES, s_out)])
    assert len(trainer.trainable_variables) == len(model.trainable_variables), "teacher chưa được đóng băng hết"
    return trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    ap.add_argument("--out", default="runs/det")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--empty-p", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--teacher", default="", help="file .keras của teacher (det/train_teacher.py) -> bật distillation")
    ap.add_argument("--kd-weight", type=float, default=1.0, help="trọng số phần loss bắt chước teacher")
    ap.add_argument("--teacher-chunk", type=int, default=0,
                    help="teacher chạy theo nhóm N ảnh để tiết kiệm VRAM (nên dùng 2-4 khi teacher 640x480); 0 = cả batch")
    ap.add_argument("--limit", type=int, default=0, help="chỉ dùng N ảnh đầu (thử nhanh)")
    ap.add_argument("--arch", default="", help='Việc 6: số kênh "p3,p4,p5", ví dụ "64,64,96"; "" = mặc định 48,80,96')
    ap.add_argument("--w-cls-mul", type=float, default=1.0, help="Việc 6: nhân trọng số loss phân lớp W_CLS")
    ap.add_argument("--no-export", action="store_true",
                    help="không export INT8 sau khi train (ghi train_meta.json; export sau bằng det/export_det.py)")
    ap.add_argument("--patience", type=int, default=0,
                    help="dừng sớm khi mAP50 val không cải thiện sau N lần đo liên tiếp (0 = tắt)")
    ap.add_argument("--stop-after", type=int, default=0,
                    help="dừng sau N epoch, vẫn giữ lịch learning rate của --epochs (0 = tắt)")
    ap.add_argument("--aug-light", action="store_true",
                    help="augment nhẹ để fine-tune cuối (kiểu close-mosaic): zoom 0.85-1.15, màu nhẹ, "
                         "không nhiễu/mờ/giảm phân giải")
    ap.add_argument("--init-weights", default="",
                    help="file .keras của student đã train (cùng kiến trúc): khởi tạo trọng số từ đó thay vì ngẫu nhiên. "
                         "Kiến trúc tích chập hoàn toàn nên nạp được cả từ run khác EDGE_IMGSZ")
    a = ap.parse_args()

    keras.utils.set_random_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ids, imgs, labs = load_train(a.data)
    if a.limit:
        ids, imgs, labs = ids[:a.limit], imgs[:a.limit], labs[:a.limit]
    print(f"Đọc {len(ids)} ảnh trong {time.time()-t0:.0f}s")

    perm = np.random.default_rng(a.seed).permutation(len(ids))
    n_val = int(len(ids) * a.val_frac)
    va, tr = perm[:n_val], perm[n_val:]
    json.dump({"val_ids": [ids[k] for k in va]}, open(out / "split.json", "w"))
    print(f"Backend Keras: {keras.backend.backend()}")
    tr_imgs, tr_labs = [imgs[k] for k in tr], [labs[k] for k in tr]
    va_imgs, va_labs = [imgs[k] for k in va], [labs[k] for k in va]

    model = build_model(**parse_arch(a.arch))
    assert model.count_params() <= 200_000, f"{model.count_params():,} tham số > 200.000 (giới hạn của thể lệ)"
    if a.w_cls_mul != 1.0:
        print(f"W_CLS x{a.w_cls_mul} -> {scale_cls_weight(a.w_cls_mul):.5f}")
    if a.init_weights:
        model.set_weights(keras.models.load_model(a.init_weights, compile=False).get_weights())
        print(f"Khởi tạo trọng số từ {a.init_weights}")
    print(f"Tham số: {model.count_params():,}   input: {IMGSZ_STR} (rộng x cao)   stride: {STRIDES_STR}")
    assert model.count_params() <= 200_000, "Vượt giới hạn 200.000 tham số của thể lệ!"
    train_seq = DetSeq(tr_imgs, tr_labs, a.batch, train=True, empty_p=a.empty_p, seed=a.seed, light=a.aug_light,
                       mode="distill" if a.teacher else "student", workers=a.workers, use_multiprocessing=False)
    steps = len(train_seq) * a.epochs
    warm = min(len(train_seq) * 3, steps // 10)
    sched = keras.optimizers.schedules.CosineDecay(a.lr * 0.01, steps - warm, alpha=0.01,
                                                   warmup_target=a.lr, warmup_steps=warm)
    opt = keras.optimizers.AdamW(sched, weight_decay=a.wd, clipnorm=10.0)
    if a.teacher:
        # Mô hình để fit = student + teacher đóng băng; đầu ra mỗi mức = concat[student, teacher] cho make_distill_loss.
        teacher = keras.models.load_model(a.teacher, compile=False)
        trainer = build_distill_trainer(model, teacher, teacher_chunk=a.teacher_chunk)
        trainer.compile(optimizer=opt, loss=[make_distill_loss(s, b, a.kd_weight) if s in KD_STRIDES
                                             else make_level_loss(s, b) for s, b in zip(STRIDES, BASE)])
        print(f"Distillation: teacher {a.teacher} ({teacher.count_params():,} tham số), kd_weight {a.kd_weight}")
    else:
        trainer = model
        trainer.compile(optimizer=opt, loss=[make_level_loss(s, b) for s, b in zip(STRIDES, BASE)])

    cb = EMAEval(va_imgs, va_labs, out, every=a.eval_every, net=model, patience=a.patience if n_val else 0,
                 stop_after=a.stop_after)
    map_init = None
    if a.init_weights and n_val:
        map_init = float(cb.evaluate()[0])
        print(f"mAP50 val của trọng số khởi tạo (trước khi train): {map_init:.4f}", flush=True)
    trainer.fit(train_seq, epochs=a.epochs, callbacks=[cb], verbose=2)
    gpu = gpu_peak_report()

    # ---- export INT8 từ bản tốt nhất (hoặc bản cuối nếu không có val) + đo mAP của chính file INT8
    extra = {**run_meta(), "imgsz": IMGSZ_STR, "strides": STRIDES_STR, "init_weights": a.init_weights or None, "val_map50_init": map_init, "aug_light": a.aug_light,
             "val_map50_float": cb.best if n_val else None, "teacher": a.teacher or None,
             "kd_weight": a.kd_weight if a.teacher else None, "teacher_chunk": a.teacher_chunk if a.teacher else None,
             "arch": a.arch or "48,80,96", "w_cls_mul": a.w_cls_mul, "train_backend": keras.backend.backend(),
             "batch": a.batch, "lr": a.lr, "epochs_planned": a.epochs,
             "epochs_run": cb.log[-1]["epoch"] if cb.log else 0, **(gpu or {})}
    if a.no_export:
        # Train ở máy khác (Kaggle): chỉ lưu thông tin run; export INT8 chạy sau bằng det/export_det.py --run <out>,
        # script đó tự đọc train_meta.json để ghi đủ thông tin vào summary.json.
        json.dump(extra, open(out / "train_meta.json", "w"), indent=1)
        print(f"Bỏ qua export (--no-export). Đã ghi {out / 'train_meta.json'}; "
              f"mAP50 float tốt nhất: {extra['val_map50_float']}")
    elif keras.backend.backend() == "tensorflow":
        from det.export_det import export_run
        export_run(a.data, out, a.seed, a.limit, extra, ids=ids, imgs=imgs)
    else:
        # TFLite converter cần TensorFlow -> chạy bước export ở tiến trình riêng với KERAS_BACKEND=tensorflow (CPU).
        cmd = [sys.executable, str(Path(__file__).with_name("export_det.py")), "--data", a.data, "--run", str(out),
               "--seed", str(a.seed), "--limit", str(a.limit), "--extra", json.dumps(extra)]
        subprocess.run(cmd, check=True, env={**os.environ, "KERAS_BACKEND": "tensorflow", "CUDA_VISIBLE_DEVICES": "-1"})


if __name__ == "__main__":
    main()
