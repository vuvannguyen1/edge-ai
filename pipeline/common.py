"""Hằng số và hàm dùng chung cho cả hai bài toán.

normalize() PHẢI được port y hệt sang C trên ESP32 (lowercase, bỏ dấu, chỉ giữ [a-z0-9], gộp khoảng trắng).
"""
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

CLASSES = ["bottle", "cup", "phone", "book", "mouse", "box", "toy_car"]
INTENTS = ["COUNT", "EXIST", "LIST", "LOCATION", "OUT_OF_SCOPE"]

SRC_W, SRC_H = 320, 240          # kích thước ảnh gốc


def _parse_imgsz(v):
    """"160" -> (160, 160);  "320x240" -> (320, 240)  (rộng x cao)."""
    w, h = v.lower().split("x") if "x" in v.lower() else (v, v)
    return int(w), int(h)


# Input detection rộng x cao, resize từ ảnh gốc (160 = 160x160 kéo giãn như cũ; 320x240 / 256x192 = giữ tỉ lệ 4:3).
# Đổi bằng biến môi trường, ví dụ cmd: set EDGE_IMGSZ=320x240
DET_W, DET_H = _parse_imgsz(os.environ.get("EDGE_IMGSZ", "160"))
assert DET_W <= SRC_W and DET_H <= SRC_H and DET_W % 16 == 0 and DET_H % 16 == 0, \
    f"EDGE_IMGSZ không hợp lệ: {DET_W}x{DET_H} (rộng <= {SRC_W}, cao <= {SRC_H}, chia hết cho 16)"
DET_K = max(DET_W, DET_H) / 160          # hệ số tỉ lệ so với bản 160 (dùng cho BASE, SIZE_RANGES)

# Các mức đầu ra detection. Mặc định 8,16 (mọi run tới Việc 6). "4,8,16" thêm head stride 4 cho vật nhỏ (Việc 7).
# Chỉ cần đặt lúc TRAIN; decode suy stride từ kích thước lưới nên export / tạo submission không cần biến này.
DET_STRIDES = tuple(int(s) for s in os.environ.get("EDGE_STRIDES", "8,16").split(","))
assert DET_STRIDES in ((8, 16), (4, 8, 16)), f"EDGE_STRIDES chỉ nhận 8,16 hoặc 4,8,16 (đang là {DET_STRIDES})"
STRIDES_STR = ",".join(map(str, DET_STRIDES))
SX, SY = DET_W / SRC_W, DET_H / SRC_H    # 160: 0.5, 0.667;  320x240: 1, 1
IMGSZ_STR = f"{DET_W}x{DET_H}"           # ghi vào summary.json

INTENT_MAX_LEN = 24
PAD, UNK = 0, 1

# Từ điển tên gọi 7 lớp (lấy từ EDA bước 3). Dùng cho: augmentation intent, logic chung kết.
# Thứ tự trong list: cụm dài trước để khớp tham lam.
OBJECT_SYNONYMS = {
    "bottle": ["chai nuoc", "chai nhua", "binh nuoc", "chai"],
    "cup": ["coc uong nuoc", "coc", "tach", "ly"],
    "phone": ["dien thoai di dong", "dien thoai", "smartphone"],
    "book": ["sach"],
    "mouse": ["chuot may tinh", "chuot"],
    "box": ["hop giay", "thung giay", "hop"],
    "toy_car": ["xe oto do choi", "oto do choi", "o to do choi", "xe do choi"],
}

# Tên vật NGOÀI 7 lớp, rút từ câu OUT_OF_SCOPE của intent_train.csv (không phải dữ liệu ngoài).
# Dùng cho augmentation intent: khuôn câu trong phạm vi + vật ngoài phạm vi -> nhãn OUT_OF_SCOPE.
OUT_OBJECTS = [
    "binh hoa", "tai nghe", "den ban", "cai thia", "cai vi", "cai dia", "cai mu", "cuon giay ve sinh",
    "giay ve sinh", "laptop", "cai thuoc ke", "thuoc ke", "qua chuoi", "cai remote", "remote", "kinh mat",
    "ban phim", "banh mi", "cai bat", "cai quat", "dong xu", "cay but", "o cam dien", "con chim", "cai o",
    "doi giay", "chia khoa", "con cho", "cuc tay", "em be", "con gau bong", "gau bong", "con meo",
    "may tinh bang", "qua tao", "cai keo", "dong ho", "cay nen", "qua bong", "nguoi", "cai ghe", "chau cay",
]


def run_meta():
    """Thông tin để lặp lại một run, ghi vào summary.json: commit git của code và lệnh đã chạy.
    Commit có đuôi '+dirty' nếu code trong pipeline/ đang khác commit đó (sửa mà chưa commit)."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=Path(__file__).resolve().parent, capture_output=True, text=True)
    try:
        commit = git("rev-parse", "--short", "HEAD").stdout.strip() or None
        if commit and git("diff", "--quiet", "HEAD", "--", ".").returncode:
            commit += "+dirty"
    except OSError:                                       # máy không có git
        commit = None
    stamp = Path(__file__).resolve().parent / "GIT_COMMIT"   # bản đóng gói đưa lên Kaggle (không có .git) ghi sẵn commit
    if not commit and stamp.exists():
        commit = stamp.read_text().strip() + " (kaggle)"
    return {"git_commit": commit, "argv": sys.argv}


def normalize(text: str) -> str:
    text = text.lower().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _phrase_table():
    rows = []
    for cls, names in OBJECT_SYNONYMS.items():
        for n in names:
            rows.append((n.split(), cls))
    rows.sort(key=lambda r: -len(r[0]))
    return rows


_PHRASES = _phrase_table()


def find_objects(tokens):
    """Trả về list (start, end, class) các cụm tên vật trong câu đã normalize (khớp dài nhất trước)."""
    out, i = [], 0
    while i < len(tokens):
        for words, cls in _PHRASES:
            if tokens[i:i + len(words)] == words:
                out.append((i, i + len(words), cls))
                i += len(words)
                break
        else:
            i += 1
    return out


def template_of(text_norm: str) -> str:
    """Khuôn câu: thay tên vật bằng OBJ. Dùng để chia validation theo nhóm."""
    toks = text_norm.split()
    spans = find_objects(toks)
    out, last = [], 0
    for s, e, _ in spans:
        out += toks[last:s] + ["OBJ"]
        last = e
    out += toks[last:]
    return " ".join(out)
