
from pathlib import Path
import json
import joblib
import sklearn

# ============================================================
# 1. ĐƯỜNG DẪN
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT / "models/uit_vsfc/final/"
    "uit_vsfc_final_model.joblib"
)

METADATA_PATH = (
    ROOT / "models/uit_vsfc/final/"
    "final_model_metadata.json"
)

OUTPUT_DIR = (
    ROOT / "results/tables/uit_vsfc/final_selection"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = {
    0: "Negative",
    1: "Positive",
    2: "Neutral"
}

# ============================================================
# 2. KIỂM TRA FILE
# ============================================================

print("=" * 75)
print("KIỂM TRA MÔ HÌNH CUỐI UIT-VSFC")
print("=" * 75)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy mô hình: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

print("\n[1] THÔNG TIN CHUNG")
print("-" * 75)

print("Model path:", MODEL_PATH)
print("Scikit-learn version:", sklearn.__version__)
print("Model type:", type(model).__name__)

if not hasattr(model, "named_steps"):
    raise TypeError(
        "Mô hình không phải sklearn Pipeline."
    )

print("\nCác bước trong Pipeline:")

for step_name, step in model.named_steps.items():
    print(f"- {step_name}: {type(step).__name__}")

# ============================================================
# 3. KIỂM TRA TF-IDF
# ============================================================

# Tìm bước TfidfVectorizer theo kiểu đối tượng,
# không phụ thuộc tên bước trong Pipeline.

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = None

for step_name, step in model.named_steps.items():
    if isinstance(step, TfidfVectorizer):
        vectorizer = step
        print(f"Tìm thấy TF-IDF tại bước: {step_name}")
        break

if vectorizer is None:
    raise TypeError(
        "Không tìm thấy TfidfVectorizer trong Pipeline."
    )

tfidf_params = vectorizer.get_params()

important_tfidf_params = [
    "analyzer",
    "ngram_range",
    "min_df",
    "max_df",
    "max_features",
    "sublinear_tf",
    "lowercase",
    "strip_accents",
    "dtype",
    "token_pattern"
]

for key in important_tfidf_params:
    print(f"{key}: {tfidf_params.get(key)}")

feature_names = vectorizer.get_feature_names_out()

print("\nTổng số đặc trưng TF-IDF:", len(feature_names))
print("20 đặc trưng đầu tiên:")

for feature in feature_names[:20]:
    print("-", feature)

# ============================================================
# 4. KIỂM TRA LOGISTIC REGRESSION
# ============================================================

print("\n[3] KIỂM TRA LOGISTIC REGRESSION")
print("-" * 75)

if "classifier" not in model.named_steps:
    raise KeyError(
        "Không tìm thấy bước 'classifier' trong Pipeline."
    )

classifier = model.named_steps["classifier"]

print("Classifier:", type(classifier).__name__)

classifier_params = classifier.get_params()

important_classifier_params = [
    "C",
    "penalty",
    "solver",
    "max_iter",
    "class_weight",
    "random_state",
    "fit_intercept",
    "tol"
]

for key in important_classifier_params:
    print(f"{key}: {classifier_params.get(key)}")

print("\nCác nhãn mô hình có thể dự đoán:")

for label in classifier.classes_:
    print(f"{label}: {LABEL_NAMES.get(int(label), 'Unknown')}")

print("\nSố đặc trưng mô hình sử dụng:",
      classifier.n_features_in_)

# ============================================================
# 5. KIỂM TRA PIPELINE CÓ HỖ TRỢ XÁC SUẤT
# ============================================================

print("\n[4] KIỂM TRA KHẢ NĂNG DỰ ĐOÁN")
print("-" * 75)

print("Có predict():", hasattr(model, "predict"))
print("Có predict_proba():", hasattr(model, "predict_proba"))
print("Có decision_function():",
      hasattr(model, "decision_function"))

# ============================================================
# 6. THỬ DỰ ĐOÁN VỚI VĂN BẢN ĐÃ TÁCH TỪ
# ============================================================

print("\n[5] THỬ DỰ ĐOÁN")
print("-" * 75)

sample_texts = [
    "giảng_viên rất nhiệt_tình và thân_thiện",
    "chất_lượng giảng_dạy kém và nội_dung khó_hiểu",
    "nội_dung môn_học bình_thường"
]

predictions = model.predict(sample_texts)

probabilities = None

if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(sample_texts)

for index, text in enumerate(sample_texts):

    predicted_label = int(predictions[index])

    print(f"\nCâu {index + 1}: {text}")
    print("Dự đoán:", LABEL_NAMES[predicted_label])

    if probabilities is not None:
        print("Xác suất:")

        for class_index, label in enumerate(
            classifier.classes_
        ):
            label = int(label)

            print(
                f"  {LABEL_NAMES[label]}: "
                f"{probabilities[index][class_index]:.4f}"
            )

# ============================================================
# 7. KIỂM TRA METADATA
# ============================================================

print("\n[6] KIỂM TRA METADATA")
print("-" * 75)

metadata = {}

if METADATA_PATH.exists():

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    print(json.dumps(
        metadata,
        ensure_ascii=False,
        indent=4
    ))

else:
    print("Không tìm thấy file metadata.")

# ============================================================
# 8. XUẤT BÁO CÁO KIỂM TRA
# ============================================================

report = {
    "model_path": str(MODEL_PATH),
    "model_type": type(model).__name__,
    "sklearn_version": sklearn.__version__,
    "pipeline_steps": [
        {
            "name": name,
            "type": type(step).__name__
        }
        for name, step in model.named_steps.items()
    ],
    "tfidf_parameters": {
        key: str(tfidf_params.get(key))
        for key in important_tfidf_params
    },
    "feature_count": int(len(feature_names)),
    "classifier_type": type(classifier).__name__,
    "classifier_parameters": {
        key: str(classifier_params.get(key))
        for key in important_classifier_params
    },
    "classes": [
        {
            "id": int(label),
            "name": LABEL_NAMES.get(int(label), "Unknown")
        }
        for label in classifier.classes_
    ],
    "supports_predict_proba": hasattr(
        model, "predict_proba"
    ),
    "metadata": metadata
}

report_path = (
    OUTPUT_DIR / "final_model_inspection.json"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        report,
        file,
        ensure_ascii=False,
        indent=4
    )

print("\n" + "=" * 75)
print("HOÀN THÀNH KIỂM TRA MÔ HÌNH")
print("=" * 75)

print("Báo cáo:", report_path)
print("Số đặc trưng:", len(feature_names))
print("Pipeline:", list(model.named_steps.keys()))
print("Classifier:", type(classifier).__name__)
print("Hỗ trợ xác suất:", hasattr(model, "predict_proba"))
