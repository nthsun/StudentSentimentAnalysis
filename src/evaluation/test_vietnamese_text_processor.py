
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.preprocessing.vietnamese_text_processor import (
    clean_text,
    segment_text,
    preprocess_and_segment
)


test_cases = [
    "Giảng viên rất nhiệt tình và thân thiện!",
    "Tôi không hài lòng với chất lượng giảng dạy.",
    "Môn học chưa thực sự tốt.",
    "Nội dung môn học bình thường.",
    "Tôi rất thích môn học này. https://example.com",
    "Liên hệ: student@example.com"
]


print("=" * 70)
print("KIỂM TRA MODULE TIỀN XỬ LÝ TIẾNG VIỆT")
print("=" * 70)

for index, text in enumerate(test_cases, start=1):

    cleaned = clean_text(text)
    segmented = segment_text(cleaned)
    final_text = preprocess_and_segment(text)

    print(f"\nTrường hợp {index}")
    print("Văn bản gốc :", text)
    print("Sau làm sạch:", cleaned)
    print("Sau tách từ :", segmented)

    assert final_text == segmented

print("\nKIỂM TRA TỪ PHỦ ĐỊNH")

negative_text = preprocess_and_segment(
    "Tôi không hài lòng với chất lượng giảng dạy."
)

print(negative_text)

assert "không" in negative_text.split()

print("\nTẤT CẢ KIỂM TRA HOÀN TẤT.")
