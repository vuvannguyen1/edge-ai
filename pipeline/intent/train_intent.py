"""Huấn luyện Intent TextCNN.

Quy trình:
  1. normalize câu, xây vocab CHỈ từ train (không dùng text test).
  2. Cross-validation 5 fold, chia theo KHUÔN CÂU (thay tên vật = OBJ) -> accuracy validation sát với test,
     vì 72,7% câu test dùng khuôn chưa gặp.
  3. Augmentation mỗi epoch (hàm augment, trên dữ liệu chứ không phải lớp trong mô hình):
     - câu COUNT/EXIST/LIST/LOCATION có tên vật: 30% thay tên vật bằng vật NGOÀI 7 lớp -> nhãn OUT_OF_SCOPE;
     - nếu không: 50% đổi tên vật sang tên khác trong 7 lớp, giữ nhãn;
     - cuối cùng thay ngẫu nhiên 10% từ bằng <unk>, KHÔNG đụng vào tên vật (để mô hình không học "vật lạ = trong phạm vi").
  4. Đo thêm độ chính xác trên câu tổ hợp (khuôn trong phạm vi của fold val + vật ngoài phạm vi, nhãn đúng OOS).
  5. Chọn số epoch theo val_loss CV trung bình, train lại trên toàn bộ train.
  6. Export TFLite INT8, đo accuracy INT8 trên fold validation để biết mức sụt do lượng tử hóa.

Chạy:  python intent/train_intent.py --data . --out runs/intent_v2
"""
import argparse, json, os, random, sys
from pathlib import Path

import numpy as np
import pandas as pd
import keras
from keras import layers
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (INTENTS, INTENT_MAX_LEN, PAD, UNK, OBJECT_SYNONYMS, OUT_OBJECTS, normalize, find_objects,
                    template_of, run_meta)

ALL_NAMES = [n.split() for names in OBJECT_SYNONYMS.values() for n in names]
OUT_TOK = [n.split() for n in OUT_OBJECTS]
OOS = INTENTS.index("OUT_OF_SCOPE")
COMP_INTENTS = ("COUNT", "EXIST", "LOCATION")       # khuôn dùng để đo câu tổ hợp


# ---------------------------------------------------------------- dữ liệu
def build_vocab(sentences):
    words = sorted({w for s in sentences for w in s.split()})
    for names in OBJECT_SYNONYMS.values():      # để augmentation đổi tên vật không sinh UNK
        for n in names:
            words += [w for w in n.split() if w not in words]
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for w in sorted(set(words)):
        vocab[w] = len(vocab)
    # tên vật ngoài phạm vi phải có sẵn từ câu train (không được thêm từ mới vào vocab)
    missing = sorted({w for n in OUT_TOK for w in n if w not in vocab})
    assert not missing, f"OUT_OBJECTS có từ không nằm trong vocab train: {missing}"
    return vocab


def encode_tokens(tokens, vocab):
    ids = [vocab.get(w, UNK) for w in tokens][:INTENT_MAX_LEN]
    return ids + [PAD] * (INTENT_MAX_LEN - len(ids))


def _fill(tokens, spans, pick):
    """Thay mỗi cụm tên vật bằng pick(). -> (tokens mới, tập vị trí của tên vật mới)."""
    out, last, prot = [], 0, []
    for s, e, _ in spans:
        out += tokens[last:s]
        name = pick()
        prot += range(len(out), len(out) + len(name))
        out += name
        last = e
    return out + tokens[last:], set(prot)


def augment(tokens, label, p_oos=0.3, p_swap=0.5, p_unk=0.1):
    """-> (tokens, nhãn). Nhãn có thể đổi thành OUT_OF_SCOPE khi tên vật bị thay bằng vật ngoài 7 lớp."""
    spans = find_objects(tokens)
    prot = {i for s, e, _ in spans for i in range(s, e)}
    if spans and label != OOS and random.random() < p_oos:
        tokens, prot = _fill(tokens, spans, lambda: random.choice(OUT_TOK))   # khuôn trong phạm vi + vật ngoài phạm vi
        label = OOS
    elif spans and random.random() < p_swap:
        tokens, prot = _fill(tokens, spans, lambda: random.choice(ALL_NAMES)) # đổi tên vật trong 7 lớp, giữ nhãn
    tokens = [w if (i in prot or random.random() >= p_unk) else "<unk>" for i, w in enumerate(tokens)]
    return tokens, label


def composite_set(sents, labels, rng, per_sent=3):
    """Câu tổ hợp để đo: câu COUNT/EXIST/LOCATION có ĐÚNG một cụm tên vật, thay cụm đó bằng vật ngoài phạm vi.
    -> {intent gốc: list câu đã thay}; nhãn đúng của mọi câu là OUT_OF_SCOPE."""
    out = {k: [] for k in COMP_INTENTS}
    for s, lab in zip(sents, labels):
        name = INTENTS[lab]
        toks = s.split()
        spans = find_objects(toks)
        if name not in out or len(spans) != 1:
            continue
        (st, en, _), = spans
        for _ in range(per_sent):
            out[name].append(" ".join(toks[:st] + rng.choice(OUT_TOK) + toks[en:]))
    return out


class IntentSeq(keras.utils.PyDataset):
    def __init__(self, sents, labels, vocab, batch=64, augment=False, **kw):
        super().__init__(**kw)
        self.toks = [s.split() for s in sents]
        self.y = np.asarray(labels, np.int32)
        self.vocab, self.batch, self.augment = vocab, batch, augment
        self.idx = np.arange(len(self.toks))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.toks) / self.batch))

    def on_epoch_end(self):
        if self.augment:
            np.random.shuffle(self.idx)

    def __getitem__(self, i):
        b = self.idx[i * self.batch:(i + 1) * self.batch]
        pairs = [augment(self.toks[j], int(self.y[j])) if self.augment else (self.toks[j], int(self.y[j])) for j in b]
        x = [encode_tokens(t, self.vocab) for t, _ in pairs]
        y = [lab for _, lab in pairs]                      # nhãn sau augmentation (có thể đã thành OOS)
        return np.asarray(x, np.int32), np.eye(len(INTENTS), dtype=np.float32)[y]


# ---------------------------------------------------------------- mô hình
@keras.saving.register_keras_serializable()
class GatherEmbedding(layers.Layer):
    """Embedding dùng tf.gather trực tiếp -> chỉ sinh 1 toán tử GATHER (Embedding của Keras 3 sinh thêm LESS/SELECT)."""
    def __init__(self, vocab_size, dim, **kw):
        super().__init__(**kw)
        self.vocab_size, self.dim = vocab_size, dim

    def build(self, input_shape):
        self.embeddings = self.add_weight(shape=(self.vocab_size, self.dim), initializer="uniform", name="embeddings")

    def call(self, x):
        return tf.gather(self.embeddings, x)

    def get_config(self):
        return {**super().get_config(), "vocab_size": self.vocab_size, "dim": self.dim}


@keras.saving.register_keras_serializable()
class RandomUNK(layers.Layer):
    """Khi train: thay mỗi token (không phải PAD) bằng UNK với xác suất p. Khi suy luận: không làm gì.
    KHÔNG còn dùng khi train (đã chuyển sang augment()); giữ lại để load được runs/intent/intent.keras cũ."""
    def __init__(self, p=0.1, **kw):
        super().__init__(**kw)
        self.p = p

    def call(self, x, training=None):
        if not training:
            return x
        drop = tf.logical_and(tf.random.uniform(tf.shape(x)) < self.p, x != PAD)
        return tf.where(drop, tf.cast(UNK, x.dtype), x)

    def get_config(self):
        return {**super().get_config(), "p": self.p}


def build_textcnn(vocab_size, embed_dim=32, filters=64, kernels=(2, 3, 4), dropout=0.3, unk_p=0.1):
    inp = layers.Input((INTENT_MAX_LEN,), dtype="int32", name="tokens")
    x = RandomUNK(unk_p)(inp) if unk_p > 0 else inp   # unk_p=0 -> bản suy luận sạch để export
    x = GatherEmbedding(vocab_size, embed_dim)(x)
    pooled = []
    for k in kernels:
        c = layers.Conv1D(filters, k, activation="relu")(x)
        pooled.append(layers.GlobalMaxPooling1D()(c))
    x = layers.Concatenate()(pooled)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(len(INTENTS), name="logits")(x)
    m = keras.Model(inp, out, name="intent_textcnn")
    m._textcnn_cfg = dict(vocab_size=vocab_size, embed_dim=embed_dim, filters=filters,
                          kernels=kernels, dropout=dropout, unk_p=unk_p)
    return m


def compile_model(m, lr):
    m.compile(optimizer=keras.optimizers.AdamW(lr, weight_decay=1e-4),
              loss=keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.05),
              metrics=["accuracy"])


# ---------------------------------------------------------------- export
def export_int8(model, rep_ids, path):
    """Full-integer INT8. Input token id giữ int32 (GATHER cần), output int8.
    Dựng lại mô hình KHÔNG có lớp RandomUNK rồi chép trọng số, để graph TFLite không dính toán tử của augmentation."""
    cfg = model._textcnn_cfg
    clean = build_textcnn(**{**cfg, "unk_p": 0.0})
    clean.set_weights(model.get_weights())
    model = clean
    def rep():
        for i in range(min(len(rep_ids), 500)):
            yield [rep_ids[i:i + 1]]
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_output_type = tf.int8
    tfl = conv.convert()
    Path(path).write_bytes(tfl)
    return tfl


def tflite_predict(tfl_bytes, ids):
    it = tf.lite.Interpreter(model_content=tfl_bytes)
    it.allocate_tensors()
    i_d, o_d = it.get_input_details()[0], it.get_output_details()[0]
    preds = []
    for row in ids:
        it.set_tensor(i_d["index"], row[None].astype(i_d["dtype"]))
        it.invoke()
        preds.append(int(np.argmax(it.get_tensor(o_d["index"])[0])))
    return np.asarray(preds)


def tflite_ops(tfl_bytes):
    it = tf.lite.Interpreter(model_content=tfl_bytes)
    return sorted({d["op_name"] for d in it._get_ops_details()})


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    ap.add_argument("--out", default="runs/intent_v2")   # không ghi đè runs/intent cũ
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--embed", type=int, default=32)
    ap.add_argument("--filters", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); keras.utils.set_random_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(Path(a.data) / "intent_train.csv")
    df["norm"] = df["question"].map(normalize)
    df["tpl"] = df["norm"].map(template_of)
    y = df["intent"].map(INTENTS.index).values
    vocab = build_vocab(df["norm"])
    json.dump(vocab, open(out / "vocab.json", "w"), ensure_ascii=False, indent=0)
    print(f"vocab={len(vocab)}  câu={len(df)}  khuôn câu={df['tpl'].nunique()}")

    # --- CV theo khuôn câu
    from sklearn.model_selection import GroupKFold
    curves, losses, fold0 = [], [], None
    comp_ok = {k: [] for k in COMP_INTENTS}            # đúng/sai trên câu tổ hợp, gộp qua các fold
    for f, (tr, va) in enumerate(GroupKFold(a.folds).split(df, y, groups=df["tpl"])):
        m = build_textcnn(len(vocab), a.embed, a.filters, unk_p=0.0)   # UNK đã làm trong augment()
        compile_model(m, a.lr)
        h = m.fit(IntentSeq(df["norm"].values[tr], y[tr], vocab, augment=True),
                  validation_data=IntentSeq(df["norm"].values[va], y[va], vocab),
                  epochs=a.epochs, verbose=0)
        curves.append(h.history["val_accuracy"]); losses.append(h.history["val_loss"])
        comp = composite_set(df["norm"].values[va], y[va], random.Random(a.seed + f))
        msg = []
        for k, sents in comp.items():
            ids = np.asarray([encode_tokens(t.split(), vocab) for t in sents], np.int32).reshape(-1, INTENT_MAX_LEN)
            ok = (np.argmax(m.predict(ids, verbose=0), 1) == OOS) if len(ids) else np.zeros(0, bool)
            comp_ok[k] += ok.tolist()
            msg.append(f"{k} {ok.mean() if len(ok) else float('nan'):.3f} (n={len(ok)})")
        print(f"fold {f}: train acc {h.history['accuracy'][-1]:.4f}  best val acc {max(h.history['val_accuracy']):.4f} @ epoch {int(np.argmax(h.history['val_accuracy']))+1}"
              f"  | câu tổ hợp -> OOS: {', '.join(msg)}")
        if f == 0:
            fold0 = (m, va)

    mean_curve = np.mean(curves, axis=0)
    best_ep = int(np.argmin(np.mean(losses, axis=0))) + 1   # accuracy bão hòa 1.0 quá sớm -> chọn theo val_loss
    comp_acc = {f"cv_comp_{k.lower()}": float(np.mean(v)) if v else None for k, v in comp_ok.items()}
    print(f"CV accuracy trung bình tốt nhất {mean_curve.max():.4f} tại epoch {best_ep}")
    print("Câu tổ hợp (khuôn trong phạm vi + vật ngoài phạm vi, đúng = OOS):", {k: round(v, 4) for k, v in comp_acc.items() if v is not None})

    # --- đo mức sụt do INT8 trên fold 0
    m0, va = fold0
    va_ids = np.asarray([encode_tokens(t.split(), vocab) for t in df["norm"].values[va]], np.int32)
    tfl0 = export_int8(m0, va_ids, out / "_fold0_int8.tflite")
    acc_f = float(np.mean(np.argmax(m0.predict(va_ids, verbose=0), 1) == y[va]))
    acc_q = float(np.mean(tflite_predict(tfl0, va_ids) == y[va]))
    print(f"fold0 (model cuối epoch): float {acc_f:.4f}  int8 {acc_q:.4f}")

    # --- train lại trên toàn bộ train
    m = build_textcnn(len(vocab), a.embed, a.filters, unk_p=0.0)
    compile_model(m, a.lr)
    m.fit(IntentSeq(df["norm"].values, y, vocab, augment=True), epochs=best_ep, verbose=0)
    m.save(out / "intent.keras")
    all_ids = np.asarray([encode_tokens(t.split(), vocab) for t in df["norm"]], np.int32)
    tfl = export_int8(m, all_ids, out / "intent_int8.tflite")
    print(f"tham số={m.count_params():,}  file INT8={len(tfl)/1024:.1f} KB  ops={tflite_ops(tfl)}")
    json.dump({**run_meta(), "imgsz": None,                    # imgsz không áp dụng cho intent
               "cv_acc": float(mean_curve.max()), **comp_acc, "best_epoch": best_ep, "fold0_float": acc_f,
               "fold0_int8": acc_q, "params": m.count_params(), "tflite_kb": len(tfl) / 1024, "ops": tflite_ops(tfl)},
              open(out / "summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
