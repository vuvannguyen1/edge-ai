# Khai báo dữ liệu ngoài và trọng số pretrained

Theo thể lệ Edge AI Challenge 2026 lần 2: mọi dữ liệu ngoài và trọng số pretrained phải được ghi lại ở đây.

## Dữ liệu ngoài

Không dùng. Chỉ dùng `train/` và `intent_train.csv` do BTC cung cấp. Không dùng `test/` hay `intent_test.csv` để train, gán nhãn hay pseudo-label.

## Trọng số pretrained

| # | Mô hình | Nguồn | Dùng để làm gì | Có nằm trong mô hình nộp không |
|---|---|---|---|---|
| 1 | MobileNetV2 (alpha 1.0), pretrained ImageNet-1k | `keras.applications.MobileNetV2(weights="imagenet")`, tải từ `storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/` | Backbone của **teacher** detection (`pipeline/det/teacher.py`, `pipeline/det/train_teacher.py`). Teacher được fine-tune trên `train/` (đúng phần train của phép chia val) rồi dùng làm thầy cho distillation khi train student (`pipeline/det/train_det.py --teacher ...`). | **Không.** Teacher chỉ dùng lúc huấn luyện, và chỉ cho bản distill `runs/kd_320x240` (student 194.008 tham số). Các bản nộp từ Việc 7 (`v7_s4_p3`, 198.196 tham số; bản cuối `g_s4_p3` cùng cấu hình) **không dùng teacher**: student khởi tạo ngẫu nhiên, train từ đầu chỉ trên `train/`, 1 lần suy luận/ảnh. Bản `v9_ft_g` (Việc 9) là `g_s4_p3` train tiếp 15 epoch với augment nhẹ, cũng không dùng teacher hay pretrained. |

Ghi chú:
- Cần tự kiểm tra lại giấy phép của trọng số MobileNetV2 trên trang Keras Applications trước khi nộp hồ sơ.
- Mô hình Intent TextCNN train từ đầu, không dùng pretrained.
- Khi đổi teacher sang mô hình khác (ví dụ từ Hugging Face), thêm một dòng mới vào bảng trên.
