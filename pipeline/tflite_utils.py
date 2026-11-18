"""Export TFLite INT8 và chạy suy luận bằng file .tflite (dùng chung cho đánh giá và tạo submission)."""
from pathlib import Path

import numpy as np
import tensorflow as tf

try:                                     # TF >= 2.20 khuyên dùng LiteRT
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    Interpreter = tf.lite.Interpreter


def export_det_int8(model, rep_images_u8, path):
    """rep_images_u8: (N,160,160,3) uint8 ảnh đã preprocess (không augment), nên 200-500 ảnh."""
    def rep():
        for im in rep_images_u8:
            yield [im[None].astype(np.float32) / 255.0]
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tfl = conv.convert()
    Path(path).write_bytes(tfl)
    return tfl


def list_ops(tfl_bytes):
    it = tf.lite.Interpreter(model_content=tfl_bytes)
    return sorted({d["op_name"] for d in it._get_ops_details()})


class TFLiteRunner:
    """Chạy model INT8: tự lượng tử hóa input và giải lượng tử output. Output trả về theo thứ tự grid lớn -> nhỏ."""
    def __init__(self, path_or_bytes):
        kw = {"model_content": path_or_bytes} if isinstance(path_or_bytes, bytes) else {"model_path": str(path_or_bytes)}
        self.it = Interpreter(**kw)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.outs = sorted(self.it.get_output_details(), key=lambda d: -int(np.prod(d["shape"])))

    def _q(self, x):
        d = self.inp
        if d["dtype"] in (np.int8, np.uint8):
            s, z = d["quantization"]
            info = np.iinfo(d["dtype"])
            return np.clip(np.round(x / s + z), info.min, info.max).astype(d["dtype"])
        return x.astype(d["dtype"])

    def __call__(self, x):
        """x: 1 mẫu, KHÔNG có chiều batch (ảnh float [0,1] hoặc mảng token id)."""
        self.it.set_tensor(self.inp["index"], self._q(np.asarray(x)[None]))
        self.it.invoke()
        res = []
        for d in self.outs:
            o = self.it.get_tensor(d["index"])[0].astype(np.float32)
            s, z = d["quantization"]
            res.append((o - z) * s if s else o)
        return res
