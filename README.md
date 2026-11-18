# Checkpoint bài nộp Kaggle 98,42483 (Edge AI Challenge 2026 lần 2, vòng loại)

Code: commit `7ea5098` (thư mục `pipeline/` trong gói này). md5 submission.csv: `f79bcc612d8bcd1122e2ab19d9a6ddc7`.

## Detection: `detection/det_int8.tflite`
- 198,196 tham số, 318.6 KB, INT8 (input và output int8), input 320x240 (rộng x cao), RGB, [0,255] -> /255 rồi lượng tử theo tham số input.
- Toán tử: ADD, CONV_2D, DEPTHWISE_CONV_2D, RESIZE_NEAREST_NEIGHBOR. 3 đầu ra stride 4/8/16 (lưới 60x80, 30x40, 15x20, 12 kênh: tx, ty, tw, th, obj, 7 lớp).
- Hậu xử lý (pipeline/det/post.py, decode): conf 0.001, NMS iou 0.6 theo lớp, đa nhãn, tối đa 100 box.
- Lịch sử train: `parent_g_s4_p3` train từ đầu 120 epoch trên 6.000 ảnh train (EDGE_STRIDES=4,8,16, --arch 64,64,72),
  rồi fine-tune 15 epoch lr 2e-4 với --aug-light. Không teacher, không pretrained. `det_float.keras` là bản float trước khi lượng tử.

## Intent: `intent/intent_int8.tflite`
- TextCNN 28,005 tham số, 39.1 KB, toán tử: CONCATENATION, CONV_2D, EXPAND_DIMS, FULLY_CONNECTED, GATHER, REDUCE_MAX, RESHAPE. Từ điển: `intent/vocab.json`.

## Tái tạo submission.csv (cho ra file giống hệt, md5 như trên)
```
set EDGE_IMGSZ=320x240
set EDGE_STRIDES=4,8,16
python pipeline/make_submission.py --data <thư mục dữ liệu cuộc thi> --det detection/det_int8.tflite ^
    --intent intent/intent_int8.tflite --vocab intent/vocab.json --nms-iou 0.6 --out submission.csv
```
