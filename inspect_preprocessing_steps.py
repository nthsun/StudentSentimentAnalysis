
import re
import unicodedata
import pandas as pd
from pathlib import Path
from underthesea import word_tokenize

# ==========================================
# 1. CÁC HÀM XỬ LÝ MINH HỌA
# ==========================================

def normalize_unicode(text):
    return unicodedata.normalize("NFC", text)


def lowercase_text(text):
    return text.lower()


def remove_url_email(text):
    # Loại bỏ URL
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Loại bỏ email
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "",
        text
    )
    return text


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def preserve_negation(text):
    # Không xóa các từ phủ định.
    negation_words = ["không", "chưa", "chẳng"]

    found = [
        word for word in negation_words
        if re.search(rf"\b{word}\b", text)
    ]

    return found


def segment_vietnamese(text):
    return word_tokenize(text)


# ==========================================
# 2. CHUỖI KIỂM THỬ CÓ CHỦ ĐÍCH
# ==========================================

sample = (
    "GIẢNG VIÊN   KHÔNG nhiệt tình!!! "
    "Liên hệ test@example.com "
    "https://example.com"
)

print("\n" + "=" * 75)
print("KIỂM TRA CHI TIẾT 6 BƯỚC TIỀN XỬ LÝ TIẾNG VIỆT")
print("=" * 75)

print("\n[VĂN BẢN BAN ĐẦU]")
print(sample)

# Bước 1: Unicode NFC
step1 = normalize_unicode(sample)
print("\n[BƯỚC 1] CHUẨN HÓA UNICODE NFC")
print("Kết quả:", step1)
print("Unicode NFC:", unicodedata.is_normalized("NFC", step1))

# Bước 2: Chữ thường
step2 = lowercase_text(step1)
print("\n[BƯỚC 2] CHUYỂN ĐỔI CHỮ THƯỜNG")
print("Kết quả:", step2)

# Bước 3: URL và email
step3 = remove_url_email(step2)
print("\n[BƯỚC 3] XỬ LÝ URL VÀ EMAIL")
print("Kết quả:", step3)
print("Còn URL:", bool(re.search(r"https?://|www\.", step3)))
print("Còn email:", bool(re.search(r"\S+@\S+", step3)))

# Bước 4: Khoảng trắng
step4 = normalize_whitespace(step3)
print("\n[BƯỚC 4] CHUẨN HÓA KHOẢNG TRẮNG")
print("Kết quả:", step4)

# Bước 5: Bảo toàn từ phủ định
step5 = step4
negations = preserve_negation(step5)

print("\n[BƯỚC 5] KIỂM TRA TỪ PHỦ ĐỊNH")
print("Kết quả:", step5)
print("Từ phủ định được giữ lại:", negations)

# Bước 6: Tách từ tiếng Việt
step6 = segment_vietnamese(step5)
print("\n[BƯỚC 6] TÁCH TỪ TIẾNG VIỆT - UNDERTHESEA")
print("Kết quả:", step6)
print("Dạng chuỗi:", " ".join(step6))


# ==========================================
# 3. KIỂM TRA MỘT SỐ MẪU UIT-VSFC THỰC TẾ
# ==========================================

print("\n" + "=" * 75)
print("ĐỐI CHIẾU MỘT SỐ MẪU DỮ LIỆU UIT-VSFC")
print("=" * 75)

path = Path("data/alternative/uit_vsfc_1000.csv")

if path.exists():
    df = pd.read_csv(path)

    text_col = next(
        (c for c in ["comments", "text", "feedback_text"]
         if c in df.columns),
        None
    )

    if text_col:
        for i, text in enumerate(df[text_col].dropna().head(5), 1):
            print(f"\n--- MẪU UIT-VSFC {i} ---")
            print("Gốc:", text)

            s1 = normalize_unicode(str(text))
            s2 = lowercase_text(s1)
            s3 = remove_url_email(s2)
            s4 = normalize_whitespace(s3)
            s5 = s4
            s6 = segment_vietnamese(s5)

            print("Bước 1 - Unicode:", s1)
            print("Bước 2 - Chữ thường:", s2)
            print("Bước 3 - URL/email:", s3)
            print("Bước 4 - Khoảng trắng:", s4)
            print("Bước 5 - Từ phủ định:", preserve_negation(s5))
            print("Bước 6 - Tách từ:", " ".join(s6))
    else:
        print("Không tìm thấy cột comments/text/feedback_text.")
else:
    print("Không tìm thấy file:", path)

print("\n" + "=" * 75)
print("HOÀN THÀNH KIỂM TRA")
print("=" * 75)
import json
import os
import joblib
from underthesea import word_tokenize

# 1. Kiểm tra thông tin tệp tin
model_path = 'models/uit_vsfc/final/uit_vsfc_final_model.joblib'
meta_path = 'models/uit_vsfc/final/final_model_metadata.json'

size_kb = os.path.getsize(model_path) / 1024
print(f'Dung lượng mô hình: {size_kb:.2f} KB')

# 2. Tải lại mô hình và metadata
model = joblib.load(model_path)
with open(meta_path, 'r', encoding='utf-8') as f:
  meta = json.load(f)

label_map = meta['labels']
print(
    f"Mô hình được chọn: {meta['selected_candidate']} (CV Macro F1:"
    f" {meta['cv_macro_f1']:.4f})"
)

# 3. Thử nghiệm dự đoán thực tế trên 3 câu mới
test_comments = [
    'Thầy giảng bài rất dễ hiểu và nhiệt tình hỗ trợ sinh viên.',
    'Phòng học quá nóng, máy chiếu bị mờ không nhìn thấy gì.',
    'Môn học này có 3 tín chỉ và thi cuối kỳ bằng hình thức trắc nghiệm.',
]

# Tiền xử lý giống quy trình đã lưu trong metadata (chữ thường + tách từ)
segmented_inputs = [
    word_tokenize(c.lower(), format='text') for c in test_comments
]
predictions = model.predict(segmented_inputs)

print('\n--- KẾT QUẢ KIỂM THỬ DỰ ĐOÁN ---')
for text, pred in zip(test_comments, predictions):
  print(f'Câu gốc: {text}')
  print(f'-> Dự đoán: {label_map[str(pred)]}\n')
