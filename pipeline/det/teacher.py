"""Mô hình TEACHER cho distillation. KHÔNG nạp lên ESP32, không tính vào giới hạn 200.000 tham số.

Backbone MobileNetV2 pretrained ImageNet (keras.applications) + neck FPN/PAN + head cùng định dạng với student:
mỗi ô 12 kênh [tx, ty, tw, th, obj, 7 class logits], cùng cách decode (det/post.py, det/loss.py).

Teacher nhận ảnh lớn gấp TEACHER_SCALE=2 mỗi chiều (320x320 khi student 160x160, 640x480 khi student 320x240).
Stride 16 và 32 trên ảnh lớn cho ra đúng lưới stride 8 và 16 của student (20x20 và 10x10 ở 160; 30x40 và 15x20 ở
320x240). Nhờ đó:
  - teacher train bằng CHÍNH loss và target của student (toạ độ input student), không cần code riêng;
  - đầu ra teacher khớp từng ô với đầu ra student -> distill dày đặc theo từng ô.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import keras
from keras import layers

from common import DET_W, DET_H
from det.model import OUT_CH, NUM_CLASSES, STRIDES
from det.data import TEACHER_SCALE


def _cbr(x, ch, k=3, s=1):
    x = layers.Conv2D(ch, k, strides=s, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)


def _head(x, ch, name):
    x = _cbr(_cbr(x, ch), ch)
    return layers.Conv2D(OUT_CH, 1, name=name)(x)


def _up_to(x, ref):
    """Phóng nearest x lên đúng lưới của ref. Vì EDGE_IMGSZ chia hết cho 16 và teacher nhận ảnh gấp 2, lưới /16 luôn
    đúng bằng 2 lần lưới /32 (160: 10x10 -> 20x20; 320x240: 15x20 -> 30x40), nên dùng UpSampling2D như bản cũ:
    giữ y hệt mô hình 160 (kể cả thứ tự khởi tạo ngẫu nhiên). Resizing chỉ là dự phòng nếu lưới không đúng x2."""
    if (ref.shape[1], ref.shape[2]) == (2 * x.shape[1], 2 * x.shape[2]):
        return layers.UpSampling2D()(x)
    return layers.Resizing(ref.shape[1], ref.shape[2], interpolation="nearest")(x)


def build_teacher(alpha=1.0, neck=128, weights="imagenet"):
    assert TEACHER_SCALE == 2 and STRIDES == (8, 16), "lưới teacher chỉ khớp student khi scale=2, stride (8,16)"
    inp = layers.Input((DET_H * TEACHER_SCALE, DET_W * TEACHER_SCALE, 3), name="image_hires")   # float [0,1] như student
    x = layers.Rescaling(2.0, offset=-1.0)(inp)                       # MobileNetV2 cần [-1,1]
    bb = keras.applications.MobileNetV2(input_tensor=x, include_top=False, weights=weights, alpha=alpha)
    c3 = bb.get_layer("block_6_expand_relu").output                   # /8  trên ảnh lớn
    c4 = bb.get_layer("block_13_expand_relu").output                  # /16 -> lưới s8 của student
    c5 = bb.get_layer("out_relu").output                              # /32 -> lưới s16 của student

    p5 = _cbr(c5, neck, 1)
    p4 = layers.Add()([_cbr(c4, neck, 1), _up_to(p5, c4)])
    p3 = layers.Add()([_cbr(c3, neck, 1), _up_to(p4, c3)])              # chi tiết phân giải cao cho vật nhỏ
    n4 = _cbr(layers.Add()([p4, _cbr(p3, neck, 3, 2)]), neck)
    n5 = _cbr(layers.Add()([p5, _cbr(n4, neck, 3, 2)]), neck)

    m = keras.Model(inp, [_head(n4, neck, "out_s8"), _head(n5, neck, "out_s16")], name="edge_det_teacher")
    for name, s in (("out_s8", STRIDES[0]), ("out_s16", STRIDES[1])):   # bias theo prior, giống student
        b = np.zeros(OUT_CH, np.float32)
        b[4] = np.log(3.0 / ((DET_H / s) * (DET_W / s)))
        b[5:] = np.log(0.6 / (NUM_CLASSES - 0.99))
        m.get_layer(name).bias.assign(b)
    return m


if __name__ == "__main__":
    t = build_teacher(weights=None)
    print(f"teacher: input {t.input_shape[1:]}, {t.count_params():,} tham số, out {[tuple(o.shape[1:]) for o in t.outputs]}")
