# Pipeline huấn luyện – Edge AI Challenge 2026 lần 2

Đặt thư mục này vào gốc `EdgeAI/` (ngang hàng với `train/`, `test/`, `intent_train.csv`).

```
common.py              normalize(), từ điển tên vật, hằng số (dùng chung, port sang C); EDGE_IMGSZ -> DET_W, DET_H
tflite_utils.py        export INT8 + chạy file .tflite
intent/train_intent.py TextCNN: CV theo khuôn câu -> train full -> intent_int8.tflite
det/model.py           mô hình detection 194.008 tham số, input EDGE_IMGSZ (160x160 hoặc 4:3), head stride 8 + 16
det/data.py            đọc dữ liệu, augmentation, gán target
det/loss.py            loss anchor-free kiểu YOLOv5 (GIoU + obj + cls)
det/post.py            decode + NMS (2 bộ ngưỡng: KAGGLE và FINAL)
det/metrics.py         mAP@0.5 kiểu Ultralytics
det/train_det.py       train + EMA + đánh giá val + export det_int8.tflite + mAP của bản INT8
det/export_det.py      export INT8 + đo mAP INT8 (train_det.py tự gọi; luôn chạy trên TensorFlow CPU)
det/eval_tflite.py     đo lại mAP50 val của det_int8.tflite có sẵn (chỉ đọc, dùng kiểm hồi quy)
det/test_roundtrip.py  phép thử target -> decode ở 160, 256x192, 320x240 (bắt lỗi đảo trục x/y)
det/test_kd_equiv.py   phép thử đường train distill (kd_weight 0 phải trùng train thường)
det/teacher.py         mô hình teacher (MobileNetV2 ImageNet + head giống student, ảnh 2x) – chỉ dùng lúc train
det/train_teacher.py   fine-tune teacher trên train/ (cùng phép chia val với train_det.py)
make_submission.py     chạy 2 file INT8 -> submission.csv + tự kiểm tra định dạng
```

Cài: `pip install tensorflow pandas scikit-learn opencv-python`

## Chạy bằng notebook

Thư mục `notebooks/` ở gốc dự án có `train_det.ipynb`, `train_intent.ipynb` và `train_all.ipynb` (intent → detection →
submission). Chỉ cần sửa cell *Cấu hình* rồi chạy các cell; notebook tự đặt `KERAS_BACKEND` và gọi đúng môi trường
(`.venv-torch` cho detection trên GPU, `.venv-tf` cho intent và submission).

## Thứ tự chạy

```bash
# 1. Intent (vài phút trên CPU)
python intent/train_intent.py --data . --out runs/intent

# 2. Detection (nên chạy trên GPU Kaggle; thử nhanh trước bằng --epochs 30)
python det/train_det.py --data . --out runs/det --epochs 200 --batch 64

# 3. Submission
python make_submission.py --data . --det runs/det/det_int8.tflite \
    --intent runs/intent/intent_int8.tflite --vocab runs/intent/vocab.json --out submission.csv
```

Submission luôn được sinh từ **file INT8**, nên điểm trên Kaggle phản ánh đúng mô hình sẽ nạp lên ESP32-S3.

## Train detection bằng GPU trên Windows (backend PyTorch)

TensorFlow không dùng được GPU trên Windows gốc. Code detection (`det/model.py`, `det/loss.py`, `det/teacher.py`)
viết bằng `keras` / `keras.ops` nên chạy được trên backend PyTorch. Môi trường riêng `.venv-torch`:

```powershell
py -3.12 -m venv .venv-torch
.venv-torch\Scripts\python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
.venv-torch\Scripts\python -m pip install keras==3.15.1 tensorflow==2.21.0 numpy==2.5.4 opencv-python pandas scikit-learn h5py

$env:KERAS_BACKEND = "torch"        # phải đặt TRƯỚC khi chạy; không đặt thì Keras dùng TensorFlow (CPU)
.venv-torch\Scripts\python pipeline/det/train_det.py --data . --out runs/det_torch --epochs 120
```

- Dòng `Backend Keras: torch` ở đầu log xác nhận đang train bằng PyTorch.
- TensorFlow trong `.venv-torch` chỉ dùng cho bước export: sau khi train, `train_det.py` tự chạy `det/export_det.py`
  ở tiến trình riêng với `KERAS_BACKEND=tensorflow` để tạo `det_int8.tflite` và đo mAP INT8.
- Intent (`intent/train_intent.py`), `make_submission.py`, `det/tune_post.py` vẫn chạy bằng TensorFlow như cũ.
- Kết quả train trên hai backend không trùng nhau từng số (khác bộ sinh số ngẫu nhiên khi khởi tạo trọng số), nên
  khi so sánh thí nghiệm hãy so các run cùng backend.

## Distillation với teacher pretrained (tùy chọn)

```bash
# 1. Fine-tune teacher (3,7 triệu tham số, ảnh 320x320). Cần GPU: trên CPU mất khoảng 14 phút/epoch.
python det/train_teacher.py --data . --out runs/teacher --epochs 40

# 2. Train student có teacher. Mọi tham số khác giữ như baseline để so sánh được.
python det/train_det.py --data . --out runs/det_kd --epochs 120 --teacher runs/teacher/teacher.keras
```

- Teacher và student phải dùng cùng `--seed`, `--val-frac` và cùng `EDGE_IMGSZ`; nếu không, teacher đã thấy ảnh val của student và mAP val bị ảo.
- Teacher đủ tốt để làm thầy khi `runs/teacher/summary.json` có `val_map50` cao hơn rõ rệt so với student (0,909). Nếu không, đừng distill.
- `--kd-weight` (mặc định 1.0) là thứ duy nhất nên chỉnh; thử 0.5 và 2.0.
- File nộp vẫn là student INT8, 1 mô hình, <= 200.000 tham số. Trọng số pretrained đã ghi trong `DECLARATIONS.md`.

## Kết quả cần đọc sau mỗi lần chạy
- `runs/intent/summary.json`: `cv_acc` (accuracy CV theo khuôn câu), `fold0_float` so với `fold0_int8`
- `runs/det/summary.json`: `val_map50_float` so với `val_map50_int8`, AP từng lớp, `ops` (chỉ được có
  ADD, CONV_2D, DEPTHWISE_CONV_2D, RESIZE_NEAREST_NEIGHBOR)
- `runs/det/log.json`: mAP50 theo epoch

## Thứ tự thí nghiệm gợi ý (đổi 1 thứ mỗi lần)
1. Baseline 200 epoch → ghi lại val mAP50 INT8, AP từng lớp (dự kiến mouse và toy_car thấp nhất).
2. Nếu INT8 thấp hơn float > 1,5 điểm: tăng ảnh đại diện lên 500, hoặc cân nhắc QAT.
3. Nếu AP mouse/toy_car thấp: tăng kênh stage `p3` trong `det/model.py` (còn dư khoảng 6k tham số, phải giảm chỗ khác), chạy lại `python det/model.py` để kiểm tra giới hạn.
4. Thử `--empty-p 0` và `0.1` (ảnh nền tổng hợp ảnh hưởng mAP ra sao).
5. Thử input 4:3 bằng biến môi trường `EDGE_IMGSZ` (cmd: `set EDGE_IMGSZ=320x240`, hoặc `IMGSZ` trong cell cấu hình notebook). Mặc định `160` = 160×160 kéo giãn như cũ. Mọi bước (train, export, eval, submission) phải cùng giá trị; các script tự báo lỗi nếu file mô hình không khớp.
6. Bản nộp cuối: `--val-frac 0` với số epoch tốt nhất.

## Lưu ý thể lệ
- Không dùng test để train, pseudo-label hay xây vocab. Vocab chỉ từ `intent_train.csv`.
- Mọi dữ liệu ngoài/pretrained phải ghi vào `DECLARATIONS.md`. Student và Intent train từ đầu; pretrained chỉ dùng cho teacher (đã khai báo).
- Chung kết: dùng bộ ngưỡng `FINAL` trong `det/post.py` (tối ưu đếm), tinh chỉnh trên ảnh OV7670 thật.
