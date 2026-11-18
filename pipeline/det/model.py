"""Mô hình detection (xem det_model.py gốc): input DET_W x DET_H (160x160 kéo giãn, hoặc 4:3 như 320x240),
2 head anchor-free stride 8 & 16.
Mỗi ô 12 kênh: [tx, ty, tw, th, obj, 7 class logits]. Decode xem det/post.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DET_W, DET_H, DET_K, DET_STRIDES
from keras import layers, Model     # không import tensorflow: chạy được cả khi KERAS_BACKEND=torch

NUM_CLASSES = 7
STRIDES = DET_STRIDES                            # (8, 16) mặc định; (4, 8, 16) khi EDGE_STRIDES=4,8,16


def base_for(s):
    """Kích thước chuẩn của box ở mức stride s: w = (2*sigmoid(tw))^2 * base -> tối đa 4*base px (s16 phủ cả ảnh)."""
    return 2.5 * s * DET_K


BASE = tuple(base_for(s) for s in STRIDES)
OUT_CH = 4 + 1 + NUM_CLASSES  # 12


def conv_bn(x, filters, k=3, s=1, act=True, name=None):
    x = layers.Conv2D(filters, k, strides=s, padding="same", use_bias=False, name=name)(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU(6.0)(x) if act else x


def dw_bn(x, s=1):
    x = layers.DepthwiseConv2D(3, strides=s, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU(6.0)(x)


def inverted_residual(x, out_ch, s=1, expand=3):
    in_ch = x.shape[-1]
    h = conv_bn(x, in_ch * expand, k=1) if expand != 1 else x
    h = dw_bn(h, s)
    h = conv_bn(h, out_ch, k=1, act=False)  # linear bottleneck
    if s == 1 and in_ch == out_ch:
        h = layers.Add()([x, h])
    return h


def stage(x, out_ch, n, s, expand=3):
    x = inverted_residual(x, out_ch, s=s, expand=expand)
    for _ in range(n - 1):
        x = inverted_residual(x, out_ch, s=1, expand=expand)
    return x


def head(x, ch, name):
    x = dw_bn(x)
    x = conv_bn(x, ch, k=1)
    x = dw_bn(x)
    return layers.Conv2D(OUT_CH, 1, padding="same", name=name)(x)


def parse_arch(s):
    """"p3,p4,p5" (ví dụ "64,64,96") -> dict cho build_model; "" -> {} (kiến trúc mặc định 48,80,96)."""
    if not s:
        return {}
    v = [int(x) for x in s.split(",")]
    assert len(v) in (3, 4), f'--arch cần "p3,p4,p5" hoặc "p3,p4,p5,s4" (đang là {s!r})'
    return dict(zip(("p3", "p4", "p5", "s4"), v))


def build_model(width=1.0, p3=48, p4=80, p5=96, s4=32):
    """p3/p4/p5: số kênh của các tầng /8, /16, /32 (mặc định là kiến trúc 194.008 tham số đã dùng).
    Việc 6 thử p3 64 (chi tiết cho vật nhỏ), bù bằng p4 64 để vẫn <= 200.000 tham số."""
    c = lambda n: max(8, int(n * width + 4) // 8 * 8)
    inp = layers.Input((DET_H, DET_W, 3), name="image")

    # Kích thước lưới ghi cho 160x160; với 320x240: /8 = 40x30, /16 = 20x15, /32 = 10x8 (Resizing lên 20x15)
    x = conv_bn(inp, c(16), k=3, s=2)                  # /2   80x80
    x = inverted_residual(x, c(16), s=1, expand=1)
    x = x4 = stage(x, c(32), n=2, s=2)                 # /4   40x40  (24 -> 32)
    p3 = stage(x, c(p3), n=3, s=2)                     # /8   20x20  (40 -> 48)
    p4 = stage(p3, c(p4), n=2, s=2)                    # /16  10x10
    p5 = stage(p4, c(p5), n=1, s=2)                    # /32  5x5  (receptive field cho vật lớn)

    neck = c(64)                                           # 56 -> 64
    l5 = conv_bn(p5, neck, k=1)
    l4 = conv_bn(p4, neck, k=1)
    l4 = layers.Add()([l4, layers.Resizing(l4.shape[1], l4.shape[2], interpolation="nearest")(l5)])
    l3 = conv_bn(p3, neck, k=1)
    l3 = layers.Add()([l3, layers.Resizing(l3.shape[1], l3.shape[2], interpolation="nearest")(l4)])

    out_s8 = head(l3, neck, "out_s8")    # 20x20x12
    out_s16 = head(l4, neck, "out_s16")  # 10x10x12
    outs = [out_s8, out_s16]
    if 4 in STRIDES:
        # Head stride 4 (Việc 7) cho vật nhỏ: lưới 80x60 ở 320x240. Tạo SAU các lớp cũ để kiến trúc mặc định giữ y hệt.
        # Mỏng (32 kênh) để vừa 200.000 tham số và nhẹ cho ESP32: đặc trưng /4 + l3 hạ kênh rồi phóng nearest x2.
        ch4 = c(s4)                                    # số kênh head stride 4 (mặc định 32; "--arch p3,p4,p5,s4")
        l2 = conv_bn(x4, ch4, k=1)
        l3d = conv_bn(l3, ch4, k=1)
        l2 = layers.Add()([l2, layers.Resizing(l2.shape[1], l2.shape[2], interpolation="nearest")(l3d)])
        outs = [head(l2, ch4, "out_s4"), *outs]
    m = Model(inp, outs, name="edge_det")
    # Khởi tạo bias theo prior (kiểu YOLOv5): obj và class bắt đầu ở xác suất thấp -> hội tụ nhanh, ổn định
    import numpy as np
    for s in STRIDES:
        layer = m.get_layer(f"out_s{s}")
        b = np.zeros(OUT_CH, np.float32)
        b[4] = np.log(3.0 / ((DET_H / s) * (DET_W / s)))   # ~3 vật mỗi ảnh mỗi mức
        b[5:] = np.log(0.6 / (NUM_CLASSES - 0.99))
        layer.bias.assign(b)
    return m




if __name__ == "__main__":
    m = build_model()
    print(f"student: input {m.input_shape[1:]}, {m.count_params():,} tham số, "
          f"out {[tuple(o.shape[1:]) for o in m.outputs]}, BASE {BASE}")
