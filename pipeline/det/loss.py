"""Loss anchor-free kiểu YOLOv5 cho từng mức đầu ra.

Decode (giống hệt det/post.py và code C trên ESP32):
  cx = (j + 2*sig(tx) - 0.5) * stride      cy = (i + 2*sig(ty) - 0.5) * stride
  w  = (2*sig(tw))^2 * base                h  = (2*sig(th))^2 * base
  score[c] = sig(obj) * sig(cls_c)
Loss = box (1 - GIoU, ô dương) + obj (BCE, target = IoU ở ô dương, 0 ở ô âm) + cls (BCE, ô dương).
Trọng số theo YOLOv5, đã chỉnh theo số lớp, số mức và cỡ ảnh.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DET_W, DET_H, DET_STRIDES
import keras
from keras import ops

# Viết bằng keras.ops để chạy được trên cả backend TensorFlow lẫn PyTorch (KERAS_BACKEND=torch -> GPU trên Windows).

NC, NL = 7, len(DET_STRIDES)                         # NL = 2 mặc định; 3 khi có head stride 4
W_BOX = 0.05
W_CLS = 0.5 * NC / 80 * 3 / NL
W_OBJ = 1.0 * (DET_W * DET_H) / 640 ** 2 * 3 / NL     # theo diện tích input (160x160 -> như cũ)
# obj loss là trung bình trên mọi ô; nhân theo số ô của lưới để mỗi ô có trọng số như nhau giữa các mức
# (8: 4 lần số ô của 16 -> 4.0 như cũ; stride 4 có 16 lần -> 16.0).
BALANCE = {4: 16.0, 8: 4.0, 16: 1.0}
LABEL_SMOOTH = 0.0


def bce_logits(target, logits):
    """BCE từ logit, cùng công thức ổn định số học với hàm sigmoid_cross_entropy_with_logits của TensorFlow."""
    return ops.maximum(logits, 0.0) - logits * target + ops.log1p(ops.exp(-ops.abs(logits)))


def decode_tf(pred, stride, base):
    gh, gw = pred.shape[1], pred.shape[2]               # lưới cao x rộng (khác nhau khi input 4:3)
    gy, gx = ops.meshgrid(ops.cast(ops.arange(gh), pred.dtype), ops.cast(ops.arange(gw), pred.dtype), indexing="ij")
    sig = ops.sigmoid(pred[..., :4])
    cx = (gx + 2 * sig[..., 0] - 0.5) * stride
    cy = (gy + 2 * sig[..., 1] - 0.5) * stride
    w = ops.square(2 * sig[..., 2]) * base
    h = ops.square(2 * sig[..., 3]) * base
    return ops.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def iou_giou(a, b, eps=1e-7):
    ix1 = ops.maximum(a[..., 0], b[..., 0]); iy1 = ops.maximum(a[..., 1], b[..., 1])
    ix2 = ops.minimum(a[..., 2], b[..., 2]); iy2 = ops.minimum(a[..., 3], b[..., 3])
    inter = ops.maximum(ix2 - ix1, 0) * ops.maximum(iy2 - iy1, 0)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter + eps
    iou = inter / union
    cw = ops.maximum(a[..., 2], b[..., 2]) - ops.minimum(a[..., 0], b[..., 0])
    ch = ops.maximum(a[..., 3], b[..., 3]) - ops.minimum(a[..., 1], b[..., 1])
    c_area = cw * ch + eps
    return iou, iou - (c_area - union) / c_area


def scale_cls_weight(mul):
    """Nhân W_CLS (Việc 6: thử x3 vì lỗi nhầm lớp phone / mouse / toy_car). Các hàm loss đọc W_CLS lúc chạy,
    nên phải gọi trước khi fit. Trả về W_CLS mới."""
    global W_CLS
    W_CLS = W_CLS * mul
    return W_CLS


def make_level_loss(stride, base):
    def level_loss(y_true, y_pred):
        y_true = ops.cast(y_true, y_pred.dtype)
        pos = y_true[..., 4]
        npos = ops.maximum(ops.sum(pos), 1.0)

        pbox = decode_tf(y_pred, stride, base)
        iou, giou = iou_giou(pbox, y_true[..., :4])
        l_box = ops.sum(pos * (1 - giou)) / npos

        obj_t = pos * ops.stop_gradient(ops.clip(iou, 0, 1))
        l_obj = ops.mean(bce_logits(obj_t, y_pred[..., 4]))

        cls_t = y_true[..., 5:] * (1 - LABEL_SMOOTH) + LABEL_SMOOTH / 2
        l_cls = ops.sum(pos[..., None] * bce_logits(cls_t, y_pred[..., 5:])) / (npos * NC)

        bs = ops.cast(ops.shape(y_pred)[0], y_pred.dtype)
        return bs * (W_BOX * l_box + W_CLS * l_cls + BALANCE[stride] * W_OBJ * l_obj)
    level_loss.__name__ = f"loss_s{stride}"
    return level_loss


def make_distill_loss(stride, base, kd_weight=1.0):
    """Loss khi có teacher. y_pred = concat[student (12 kênh), teacher (12 kênh)] trên cùng lưới.
    = loss nhãn thật (như cũ) + kd_weight * loss bắt chước teacher theo từng ô:
        obj : BCE với xác suất obj của teacher (mọi ô)
        cls : BCE với xác suất lớp của teacher, trọng số = obj của teacher (ô nào teacher thấy vật thì mới tính)
        box : 1 - GIoU giữa box student và box teacher, cùng trọng số
    Dùng lại các trọng số W_BOX / W_CLS / W_OBJ / BALANCE để hai phần cùng thang đo."""
    hard = make_level_loss(stride, base)
    n_out = 5 + NC

    def distill_loss(y_true, y_pred):
        s, t = y_pred[..., :n_out], ops.stop_gradient(y_pred[..., n_out:])
        l_hard = hard(y_true, s)

        t_obj = ops.sigmoid(t[..., 4])
        l_obj = ops.mean(bce_logits(t_obj, s[..., 4]))
        wsum = ops.maximum(ops.sum(t_obj), 1.0)
        l_cls = ops.sum(t_obj[..., None] * bce_logits(
            ops.sigmoid(t[..., 5:]), s[..., 5:])) / (wsum * NC)
        _, giou = iou_giou(decode_tf(s, stride, base), decode_tf(t, stride, base))
        l_box = ops.sum(t_obj * (1 - giou)) / wsum

        bs = ops.cast(ops.shape(y_pred)[0], y_pred.dtype)
        return l_hard + kd_weight * bs * (W_BOX * l_box + W_CLS * l_cls + BALANCE[stride] * W_OBJ * l_obj)
    distill_loss.__name__ = f"kd_s{stride}"
    return distill_loss
