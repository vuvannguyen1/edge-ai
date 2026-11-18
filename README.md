# Edge AI Challenge 2026 lần 2 – Bài nộp vòng loại (public LB 98,42483)

Repo chứa hai mô hình đã dùng để tạo bài nộp, code suy luận và file `submission.csv` đã nộp lên Kaggle.
Chạy lại theo hướng dẫn bên dưới sẽ cho ra file **giống hệt từng byte** với bản đã nộp
(md5 `f79bcc612d8bcd1122e2ab19d9a6ddc7`).

## Mô hình

| | Detection | Intent |
|---|---|---|
| File | `detection/det_int8.tflite` | `intent/intent_int8.tflite` (+ `intent/vocab.json`) |
| Kiến trúc | CNN một giai đoạn, 3 đầu ra stride 4/8/16 | TextCNN |
| Số tham số | 198.196 | 28.005 |
| Định dạng | TFLite INT8 (input/output int8), 318,6 KB | TFLite INT8, 39,1 KB |
| Input | Ảnh RGB 320×240 (rộng × cao) | Chuỗi token theo `vocab.json` |
| Toán tử | ADD, CONV_2D, DEPTHWISE_CONV_2D, RESIZE_NEAREST_NEIGHBOR | CONCATENATION, CONV_2D, EXPAND_DIMS, FULLY_CONNECTED, GATHER, REDUCE_MAX, RESHAPE |

Hậu xử lý detection (`pipeline/det/post.py`): ngưỡng conf 0,001, NMS IoU 0,6 theo từng lớp, đa nhãn, tối đa 100 box/ảnh.

## Tuân thủ thể lệ vòng loại

- Mỗi mô hình dưới 200.000 tham số (detection 198.196, intent 28.005).
- Một mô hình detection, một lần suy luận mỗi ảnh: không TTA, không ensemble.
- Không dùng teacher, không dùng trọng số pretrained, không dùng dữ liệu ngoài cho hai mô hình trên; cả hai train từ đầu chỉ trên dữ liệu train của BTC.
- Không dùng tập test để train, gán nhãn hay pseudo-label.
- Chỉ dùng toán tử TFLite Micro hỗ trợ.
- Chi tiết khai báo: [`DECLARATIONS.md`](DECLARATIONS.md).

## Tạo lại submission.csv

### 1. Cài đặt

Python 3.12. Kết quả byte-exact đã kiểm tra với `tensorflow 2.21.0`, `opencv-python 5.0.0.93`, `numpy 2.5.4`, chạy trên CPU.

```bash
pip install -r pipeline/requirements.txt
```

### 2. Chuẩn bị dữ liệu

Dữ liệu cuộc thi không có trong repo. Thư mục truyền vào `--data` cần có:

```
<DATA>/
├── test/images/*.png
├── intent_test.csv
└── sample_submission.csv
```

### 3. Chạy

Bắt buộc đặt hai biến môi trường `EDGE_IMGSZ=320x240` và `EDGE_STRIDES=4,8,16` như dưới đây. Nếu thiếu, code dùng cấu hình mặc định của phiên bản cũ và kết quả sẽ sai.

**Linux / macOS**
```bash
EDGE_IMGSZ=320x240 EDGE_STRIDES=4,8,16 python pipeline/make_submission.py --data <DATA> --det detection/det_int8.tflite --intent intent/intent_int8.tflite --vocab intent/vocab.json --nms-iou 0.6 --out submission_rerun.csv
```

**Windows PowerShell**
```powershell
$env:PYTHONIOENCODING="utf-8"; $env:EDGE_IMGSZ="320x240"; $env:EDGE_STRIDES="4,8,16"
python pipeline/make_submission.py --data <DATA> --det detection/det_int8.tflite --intent intent/intent_int8.tflite --vocab intent/vocab.json --nms-iou 0.6 --out submission_rerun.csv
```

**Windows cmd**
```bat
set PYTHONIOENCODING=utf-8
set EDGE_IMGSZ=320x240
set EDGE_STRIDES=4,8,16
python pipeline/make_submission.py --data <DATA> --det detection/det_int8.tflite --intent intent/intent_int8.tflite --vocab intent/vocab.json --nms-iou 0.6 --out submission_rerun.csv
```

Trên Windows cần `PYTHONIOENCODING=utf-8` để script in được log tiếng Việt.
Script tự kiểm tra định dạng file ra (đủ 8.000 dòng, khớp `sample_submission.csv`).

### 4. Đối chiếu

```bash
md5sum submission_rerun.csv          # Linux / macOS
certutil -hashfile submission_rerun.csv MD5   # Windows
```

Kết quả phải là `f79bcc612d8bcd1122e2ab19d9a6ddc7`, trùng với `submission.csv` trong repo.

## Cấu trúc repo

| Đường dẫn | Nội dung |
|---|---|
| `submission.csv` | File đã nộp (public LB 98,42483) |
| `detection/det_int8.tflite` | Mô hình detection dùng để nộp |
| `detection/det_float.keras` | Cùng mô hình ở dạng float, trước khi lượng tử INT8 |
| `detection/parent_g_s4_p3/` | Trọng số trung gian trong quá trình train (không dùng khi suy luận) |
| `detection/*.json` | Cấu hình và log train |
| `intent/intent_int8.tflite`, `intent/vocab.json` | Mô hình intent dùng để nộp và từ điển |
| `intent/intent_float.keras`, `intent/summary.json` | Bản float và thông tin mô hình intent |
| `pipeline/` | Code: tiền xử lý, mô hình, train, export, hậu xử lý, tạo submission (`make_submission.py`) |
| `notebooks/` | Notebook train đã dùng |
| `classes.txt` | Tên 7 lớp detection |
| `DECLARATIONS.md` | Khai báo dữ liệu ngoài và trọng số pretrained |
