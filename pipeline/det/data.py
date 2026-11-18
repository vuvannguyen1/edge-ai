"""Đọc dữ liệu, augmentation và mã hóa target cho detection.

Toạ độ:
  - label gốc: YOLO chuẩn hóa (cls cx cy w h) trên ảnh 320x240
  - nội bộ: box xyxy theo pixel
  - input mô hình: DET_W x DET_H (160x160 kéo giãn x*0.5, y*0.667; 320x240 giữ nguyên tỉ lệ)
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import keras

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import SRC_W, SRC_H, DET_W, DET_H, DET_K, SX, SY
from det.model import STRIDES, NUM_CLASSES, OUT_CH


# ---------------------------------------------------------------- đọc dữ liệu
def read_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_label(path):
    """-> (n,5) float32 [cls, x1, y1, x2, y2] pixel trên ảnh gốc."""
    if not Path(path).exists():
        return np.zeros((0, 5), np.float32)
    a = np.loadtxt(path, ndmin=2, dtype=np.float32)
    if a.size == 0:
        return np.zeros((0, 5), np.float32)
    c, cx, cy, w, h = a.T
    return np.stack([c, (cx - w / 2) * SRC_W, (cy - h / 2) * SRC_H,
                     (cx + w / 2) * SRC_W, (cy + h / 2) * SRC_H], 1).astype(np.float32)


def load_train(data_dir):
    d = Path(data_dir) / "train"
    ids = sorted(p.stem for p in (d / "images").glob("*.png"))
    imgs = [read_image(d / "images" / f"{i}.png") for i in ids]
    labs = [read_label(d / "labels" / f"{i}.txt") for i in ids]
    return ids, imgs, labs


def preprocess(img):
    """Ảnh gốc RGB uint8 320x240 -> DET_W x DET_H uint8 (INTER_AREA). Trên ESP32 làm y hệt."""
    return cv2.resize(img, (DET_W, DET_H), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- augmentation
def make_empty(img, boxes, rng):
    """Tạo cảnh không có vật bằng inpaint các box (bù cho việc train không có ảnh nền)."""
    mask = np.zeros(img.shape[:2], np.uint8)
    for _, x1, y1, x2, y2 in boxes:
        cv2.rectangle(mask, (int(x1) - 3, int(y1) - 3), (int(x2) + 3, int(y2) + 3), 255, -1)
    return cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA), np.zeros((0, 5), np.float32)


TEACHER_SCALE = 2      # teacher nhìn ảnh lớn gấp 2 mỗi chiều (320x320 khi student 160x160, 640x480 khi 320x240)
                       # -> cùng kích thước lưới đầu ra


def preprocess_hires(img, k=TEACHER_SCALE):
    """Ảnh gốc -> input của teacher (DET_W*k x DET_H*k). Chỉ dùng khi train/đánh giá teacher, không lên ESP32."""
    return cv2.resize(img, (DET_W * k, DET_H * k), interpolation=cv2.INTER_LINEAR)


def _color(im, dh, ds, dv, ct, wb):
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + dh) % 180
    hsv[..., 1] *= ds
    hsv[..., 2] *= dv
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    out = (out - 128) * ct + 128
    out *= wb                                             # lệch cân bằng trắng
    return out


def augment(img, boxes, rng, empty_p=0.05, hires=0, light=False):
    """hires=k>0: trả thêm ảnh thứ ba là CÙNG phép biến đổi hình học + màu nhưng dựng ở (DET_W*k, DET_H*k) và không thêm
    nhiễu/mờ/giảm phân giải (ảnh cho teacher). Với hires=0, kết quả giống hệt bản trước (cùng thứ tự rút số ngẫu nhiên)."""
    if len(boxes) and rng.random() < empty_p:
        img, boxes = make_empty(img, boxes, rng)

    # --- hình học: zoom 0.75-1.3, lệch tâm, lật ngang. Zoom > 1 cắt vật ở mép giống camera thật.
    # light=True (fine-tune cuối, --aug-light): biến đổi nhẹ, không nhiễu/mờ/giảm phân giải. light=False y hệt bản cũ.
    z = rng.uniform(0.85, 1.15) if light else rng.uniform(0.75, 1.3)
    ar = rng.uniform(0.95, 1.05) if light else rng.uniform(0.9, 1.1)
    sx, sy = SX * z * ar, SY * z / ar
    cx = SRC_W / 2 + rng.uniform(-0.2, 0.2) * SRC_W
    cy = SRC_H / 2 + rng.uniform(-0.2, 0.2) * SRC_H
    tx, ty = DET_W / 2 - sx * cx, DET_H / 2 - sy * cy
    flip = rng.random() < 0.5
    if flip:
        sx, tx = -sx, DET_W - tx
    M = np.float32([[sx, 0, tx], [0, sy, ty]])
    fill = tuple(int(v) for v in rng.integers(0, 255, 3)) if rng.random() < 0.5 else (114, 114, 114)
    out = cv2.warpAffine(img, M, (DET_W, DET_H), flags=cv2.INTER_AREA if z < 1 else cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    big = None
    if hires:
        big = cv2.warpAffine(img, M * hires, (DET_W * hires, DET_H * hires),
                             flags=cv2.INTER_AREA if z * hires < 1 else cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=fill)

    if len(boxes):
        x = boxes[:, [1, 3]] * sx + tx
        y = boxes[:, [2, 4]] * sy + ty
        x1, x2 = x.min(1), x.max(1)
        y1, y2 = y.min(1), y.max(1)
        area0 = (x2 - x1) * (y2 - y1)
        cx1, cx2 = np.clip(x1, 0, DET_W), np.clip(x2, 0, DET_W)
        cy1, cy2 = np.clip(y1, 0, DET_H), np.clip(y2, 0, DET_H)
        area = (cx2 - cx1) * (cy2 - cy1)
        keep = (area >= 0.4 * area0) & (cx2 - cx1 >= 2) & (cy2 - cy1 >= 2)
        boxes = np.stack([boxes[:, 0], cx1, cy1, cx2, cy2], 1)[keep]

    # --- màu sắc và chất lượng ảnh (mô phỏng OV7670: nhiễu, lệch màu, mờ, độ phân giải thấp)
    if light:
        col = (rng.uniform(-3, 3), rng.uniform(0.75, 1.25), rng.uniform(0.8, 1.2), rng.uniform(0.9, 1.1),
               rng.uniform(0.95, 1.05, size=3))
        out = np.clip(_color(out, *col), 0, 255).astype(np.uint8)
    else:
        col = (rng.uniform(-5, 5), rng.uniform(0.5, 1.5), rng.uniform(0.6, 1.4), rng.uniform(0.75, 1.25),
               rng.uniform(0.9, 1.1, size=3))
        out = _color(out, *col)
        if rng.random() < 0.5:
            out += rng.normal(0, rng.uniform(2, 10), out.shape)
        out = np.clip(out, 0, 255).astype(np.uint8)
        if rng.random() < 0.2:
            out = cv2.GaussianBlur(out, (3, 3), 0)
        if rng.random() < 0.15:
            f = rng.uniform(0.5, 0.8)
            small = cv2.resize(out, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            out = cv2.resize(small, (DET_W, DET_H), interpolation=cv2.INTER_LINEAR)
    if hires:
        big = np.clip(_color(big, *col), 0, 255).astype(np.uint8)
        return out, boxes.astype(np.float32), big
    return out, boxes.astype(np.float32)


def to_input_boxes(boxes):
    """Box ảnh gốc -> box toạ độ input DET_W x DET_H (không augmentation)."""
    b = boxes.copy()
    b[:, [1, 3]] *= SX
    b[:, [2, 4]] *= SY
    return b


# ---------------------------------------------------------------- mã hóa target
# cạnh dài box (px input) -> mức chịu trách nhiệm (64/32 px ở 160). Mức 4 (Việc 7) là CỘNG THÊM: vật <= 24*K px
# (48 px ở 320x240) được gán cho cả mức 4 lẫn mức 8; mức 8 và 16 giữ nguyên như khi không có head stride 4.
SIZE_RANGES = {4: (0, 24 * DET_K), 8: (0, 64 * DET_K), 16: (32 * DET_K, 1e9)}


def encode_targets(boxes):
    """boxes (n,5) toạ độ input -> tuple target cho từng mức, mỗi cái (DET_H/s, DET_W/s, 12):
    [x1,y1,x2,y2 (px input), pos, 7 one-hot class]. Gán kiểu YOLOv5: ô chứa tâm + 2 ô kề gần tâm nhất."""
    outs = []
    order = np.argsort(-(boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])) if len(boxes) else []
    for s in STRIDES:
        gh, gw = DET_H // s, DET_W // s               # hàng = trục y, cột = trục x
        t = np.zeros((gh, gw, OUT_CH), np.float32)
        lo, hi = SIZE_RANGES[s]
        for k in order:                           # box lớn trước, box nhỏ ghi đè -> ưu tiên vật nhỏ
            c, x1, y1, x2, y2 = boxes[k]
            if not (lo <= max(x2 - x1, y2 - y1) <= hi):
                continue
            gx, gy = (x1 + x2) / 2 / s, (y1 + y2) / 2 / s
            gi, gj = min(int(gx), gw - 1), min(int(gy), gh - 1)
            cells = [(gj, gi)]
            cells.append((gj, gi - 1) if gx - gi < 0.5 else (gj, gi + 1))
            cells.append((gj - 1, gi) if gy - gj < 0.5 else (gj + 1, gi))
            for r, q in cells:
                if 0 <= r < gh and 0 <= q < gw:
                    t[r, q, :4] = (x1, y1, x2, y2)
                    t[r, q, 4] = 1
                    t[r, q, 5:] = 0
                    t[r, q, 5 + int(c)] = 1
        outs.append(t)
    return tuple(outs)


class DetSeq(keras.utils.PyDataset):
    """mode: "student" -> x = ảnh DET_W x DET_H;  "teacher" -> x = ảnh lớn gấp TEACHER_SCALE mỗi chiều;
    "distill" -> x = (ảnh student, ảnh lớn) cho cùng một phép augmentation. Target luôn ở toạ độ input của student."""
    def __init__(self, imgs, labs, batch=64, train=True, empty_p=0.05, seed=0, mode="student", light=False, **kw):
        super().__init__(**kw)
        self.light = light                               # --aug-light: augment nhẹ cho fine-tune cuối
        assert mode in ("student", "teacher", "distill")
        self.mode = mode
        self.imgs, self.labs, self.batch, self.train, self.empty_p = imgs, labs, batch, train, empty_p
        self.idx = np.arange(len(imgs))
        self.seed, self.epoch = seed, 0
        if train:
            np.random.default_rng(seed).shuffle(self.idx)

    def __len__(self):
        return int(np.ceil(len(self.idx) / self.batch))

    def on_epoch_end(self):
        self.epoch += 1
        if self.train:
            np.random.default_rng(self.seed + self.epoch).shuffle(self.idx)

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, self.epoch, i))
        k = 0 if self.mode == "student" else TEACHER_SCALE
        xs, xh, ts = [], [], [[] for _ in STRIDES]       # ts[k]: target của mức STRIDES[k]
        for j in self.idx[i * self.batch:(i + 1) * self.batch]:
            if self.train:
                im, bx, *big = augment(self.imgs[j], self.labs[j], rng, self.empty_p, hires=k, light=self.light)
            else:
                im, bx = preprocess(self.imgs[j]), to_input_boxes(self.labs[j])
                big = [preprocess_hires(self.imgs[j], k)] if k else []
            for t_list, t in zip(ts, encode_targets(bx)):
                t_list.append(t)
            xs.append(im); xh += big
        y = tuple(np.asarray(t) for t in ts)
        x = np.asarray(xs, np.float32) / 255.0
        if self.mode == "student":
            return x, y
        xh = np.asarray(xh, np.float32) / 255.0
        return (xh, y) if self.mode == "teacher" else ((x, xh), y)
