
from pathlib import Path
import sys
import joblib
import re

# ============================================================
# 1. CẤU HÌNH
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT / "models" / "uit_vsfc" /
    "final" / "uit_vsfc_final_model.joblib"
)

# Thêm thư mục gốc vào Python path
sys.path.insert(0, str(ROOT))

from src.preprocessing.vietnamese_text_processor import (
    preprocess_and_segment
)

LABEL_NAMES = {
    0: "Negative",
    1: "Positive",
    2: "Neutral"
}

LABEL_VIETNAMESE = {
    0: "Tiêu cực",
    1: "Tích cực",
    2: "Trung tính"
}

# ============================================================
# 2. LOAD MODEL
# ============================================================

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy mô hình: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    print("Đã tải mô hình thành công.")
    print("Mô hình:", type(model).__name__)

    return model


# ============================================================
# 3. TIỀN XỬ LÝ ĐẦU VÀO
# ============================================================

def preprocess_input(text):

    if not isinstance(text, str):
        raise TypeError("Đầu vào phải là chuỗi văn bản.")

    if not text.strip():
        raise ValueError("Vui lòng nhập nội dung.")

    segmented_text = preprocess_and_segment(text)

    if not segmented_text.strip():
        raise ValueError(
            "Không có nội dung hợp lệ sau tiền xử lý."
        )

    # Kiểm tra có token chứa chữ hoặc số hay không
    if not re.search(r"\w", segmented_text, flags=re.UNICODE):
        raise ValueError(
            "Văn bản chỉ chứa dấu câu hoặc ký tự không hợp lệ."
        )

    return segmented_text


# ============================================================
# 4. DỰ ĐOÁN CẢM XÚC
# ============================================================

def predict_sentiment(model, text):

    # Tiền xử lý
    processed_text = preprocess_input(text)

    # Dự đoán nhãn
    prediction = int(model.predict([processed_text])[0])

    # Dự đoán xác suất
    probabilities = model.predict_proba(
        [processed_text]
    )[0]

    # Lấy thứ tự lớp từ mô hình
    classes = model.named_steps["classifier"].classes_

    probability_result = {}

    for index, class_id in enumerate(classes):

        class_id = int(class_id)

        probability_result[class_id] = float(
            probabilities[index]
        )

    result = {
        "original_text": text,
        "processed_text": processed_text,
        "predicted_label": prediction,
        "sentiment": LABEL_NAMES[prediction],
        "sentiment_vietnamese": LABEL_VIETNAMESE[prediction],
        "probabilities": probability_result
    }

    return result


# ============================================================
# 5. HIỂN THỊ KẾT QUẢ
# ============================================================

def display_result(result):

    print("\n" + "=" * 65)
    print("KẾT QUẢ PHÂN TÍCH CẢM XÚC")
    print("=" * 65)

    print("\nVăn bản đầu vào:")
    print(result["original_text"])

    print("\nSau tiền xử lý:")
    print(result["processed_text"])

    print("\nKẾT QUẢ PHÂN LOẠI:")

    print(
        f"{result['sentiment_vietnamese']} "
        f"({result['sentiment']})"
    )

    print("\nXÁC SUẤT DỰ ĐOÁN:")

    for class_id in [0, 1, 2]:

        probability = result["probabilities"][class_id]

        print(
            f"{LABEL_VIETNAMESE[class_id]:12s}: "
            f"{probability * 100:.2f}%"
        )

    print("\n" + "=" * 65)


# ============================================================
# 6. CHƯƠNG TRÌNH CHÍNH
# ============================================================

def main():

    print("=" * 65)
    print("HỆ THỐNG PHÂN LOẠI CẢM XÚC TIẾNG VIỆT")
    print("Dataset: UIT-VSFC")
    print("Model: TF-IDF + Logistic Regression")
    print("=" * 65)

    try:
        model = load_model()

    except Exception as error:
        print("Lỗi tải mô hình:", error)
        return

    while True:

        print("\nNhập câu cần phân tích.")
        print("Nhập 'exit' để thoát.")

        text = input("\nVăn bản: ").strip()

        if text.lower() == "exit":
            print("\nĐã kết thúc chương trình.")
            break

        try:

            result = predict_sentiment(model, text)

            display_result(result)

        except (ValueError, TypeError) as error:

            print("\nKhông thể phân tích:", error)

        except Exception as error:

            print("\nĐã xảy ra lỗi:", error)


if __name__ == "__main__":
    main()
