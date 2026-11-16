
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate
)

# ============================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    ROOT / "data/alternative/processed/uit_vsfc_train.csv"
)

MODEL_A_PATH = (
    ROOT / "models/uit_vsfc/advanced/"
    "F2_Word_Unigram_Bigram_LogisticRegression.joblib"
)

MODEL_B_PATH = (
    ROOT / "models/uit_vsfc/advanced/chi2/"
    "CHI1_Word_Unigram_Bigram_LogisticRegression.joblib"
)

OUTPUT_DIR = (
    ROOT / "results/tables/uit_vsfc/final_selection"
)

FINAL_MODEL_DIR = (
    ROOT / "models/uit_vsfc/final"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Chênh lệch Macro F1 dưới ngưỡng này được xem là
# gần tương đương để ưu tiên mô hình đơn giản.
# Đây là quy tắc thực nghiệm, không phải kiểm định
# ý nghĩa thống kê.
TOLERANCE = 0.005

# ============================================================
# 2. ĐỌC DỮ LIỆU
# ============================================================

print("=" * 75)
print("LỰA CHỌN MÔ HÌNH CUỐI CÙNG - UIT-VSFC")
print("=" * 75)

if not TRAIN_PATH.exists():
    raise FileNotFoundError(TRAIN_PATH)

df = pd.read_csv(TRAIN_PATH)

required = ["segmented_comments", "sentiments"]

if not set(required).issubset(df.columns):
    raise ValueError("Thiếu cột dữ liệu bắt buộc.")

if df[required].isnull().any().any():
    raise ValueError("Dữ liệu Train có giá trị thiếu.")

if len(df) != 800:
    raise ValueError(
        f"Train phải có 800 mẫu, hiện có {len(df)}."
    )

X = df["segmented_comments"].astype(str)
y = df["sentiments"].astype(int)

if set(y.unique()) != {0, 1, 2}:
    raise ValueError("Train không có đủ ba nhãn.")

print(f"\nSố mẫu Train: {len(df)}")
print("\nPhân bố nhãn:")
print(y.value_counts().sort_index())

# ============================================================
# 3. NẠP HAI ỨNG VIÊN
# ============================================================

CANDIDATES = {
    "A_No_Chi2": {
        "path": MODEL_A_PATH,
        "description": "TF-IDF + Logistic Regression",
        "complexity": 0
    },
    "B_With_Chi2": {
        "path": MODEL_B_PATH,
        "description": "TF-IDF + Chi-square + Logistic Regression",
        "complexity": 1
    }
}

for name, info in CANDIDATES.items():
    if not info["path"].exists():
        raise FileNotFoundError(
            f"Không tìm thấy mô hình {name}: "
            f"{info['path']}"
        )

# ============================================================
# 4. ĐỊNH NGHĨA CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

# Cùng một bộ Fold cho hai ứng viên
splits = list(cv.split(X, y))

scoring = {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro"
}

summary_rows = []
fold_rows = []
loaded_models = {}

# ============================================================
# 5. ĐÁNH GIÁ TỪNG ỨNG VIÊN
# ============================================================

for name, info in CANDIDATES.items():

    print("\n" + "=" * 75)
    print(f"ỨNG VIÊN: {name}")
    print(info["description"])
    print("=" * 75)

    fitted_model = joblib.load(info["path"])
    loaded_models[name] = fitted_model

    print("\nSiêu tham số mô hình:")
    print(fitted_model.named_steps["classifier"].get_params())

    # clone tạo estimator chưa fit nhưng giữ nguyên
    # các siêu tham số đã lựa chọn.
    # Mỗi Fold sẽ fit lại toàn bộ Pipeline.
    estimator = clone(fitted_model)

    scores = cross_validate(
        estimator,
        X,
        y,
        cv=splits,
        scoring=scoring,
        n_jobs=-1,
        error_score="raise",
        return_train_score=True
    )

    for i in range(5):
        fold_rows.append({
            "candidate": name,
            "fold": i + 1,
            "accuracy": scores["test_accuracy"][i],
            "f1_macro": scores["test_f1_macro"][i],
            "f1_weighted": scores["test_f1_weighted"][i],
            "precision_macro": scores[
                "test_precision_macro"
            ][i],
            "recall_macro": scores[
                "test_recall_macro"
            ][i],
            "train_f1_macro": scores[
                "train_f1_macro"
            ][i]
        })

    mean_f1 = np.mean(scores["test_f1_macro"])
    std_f1 = np.std(
        scores["test_f1_macro"],
        ddof=1
    )

    summary_rows.append({
        "candidate": name,
        "description": info["description"],
        "mean_accuracy": np.mean(
            scores["test_accuracy"]
        ),
        "mean_f1_macro": mean_f1,
        "std_f1_macro": std_f1,
        "mean_f1_weighted": np.mean(
            scores["test_f1_weighted"]
        ),
        "mean_precision_macro": np.mean(
            scores["test_precision_macro"]
        ),
        "mean_recall_macro": np.mean(
            scores["test_recall_macro"]
        ),
        "mean_train_f1_macro": np.mean(
            scores["train_f1_macro"]
        ),
        "complexity": info["complexity"]
    })

    print("\nMacro F1 theo từng Fold:")

    for i, score in enumerate(
        scores["test_f1_macro"], start=1
    ):
        print(f"Fold {i}: {score:.4f}")

    print(f"\nMean Macro F1: {mean_f1:.4f}")
    print(f"Std Macro F1:  {std_f1:.4f}")

# ============================================================
# 6. LƯU KẾT QUẢ TỪNG FOLD
# ============================================================

fold_df = pd.DataFrame(fold_rows)

fold_df.to_csv(
    OUTPUT_DIR / "final_candidate_fold_scores.csv",
    index=False,
    encoding="utf-8-sig"
)

summary_df = pd.DataFrame(summary_rows)

summary_df = summary_df.sort_values(
    "mean_f1_macro",
    ascending=False
).reset_index(drop=True)

summary_df.to_csv(
    OUTPUT_DIR / "final_candidate_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 90)
print("BẢNG SO SÁNH HAI ỨNG VIÊN")
print("=" * 90)

print(
    summary_df[
        [
            "candidate",
            "mean_accuracy",
            "mean_f1_macro",
            "std_f1_macro",
            "mean_train_f1_macro"
        ]
    ].round(4).to_string(index=False)
)

# ============================================================
# 7. SO SÁNH CHÊNH LỆCH TỪNG FOLD
# ============================================================

a_scores = (
    fold_df[fold_df["candidate"] == "A_No_Chi2"]
    .sort_values("fold")["f1_macro"]
    .to_numpy()
)

b_scores = (
    fold_df[fold_df["candidate"] == "B_With_Chi2"]
    .sort_values("fold")["f1_macro"]
    .to_numpy()
)

difference = b_scores - a_scores

difference_df = pd.DataFrame({
    "fold": range(1, 6),
    "f1_without_chi2": a_scores,
    "f1_with_chi2": b_scores,
    "difference_B_minus_A": difference
})

difference_df.to_csv(
    OUTPUT_DIR / "final_candidate_fold_differences.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nChênh lệch Macro F1 từng Fold (B - A):")
print(difference_df.round(4).to_string(index=False))

# ============================================================
# 8. LỰA CHỌN MÔ HÌNH CUỐI
# ============================================================

mean_a = float(np.mean(a_scores))
mean_b = float(np.mean(b_scores))

if abs(mean_a - mean_b) < TOLERANCE:
    selected_name = "A_No_Chi2"
    selection_reason = (
        "Hai mô hình có Macro F1 gần tương đương. "
        "Ưu tiên mô hình không sử dụng Chi-square "
        "vì quy trình đơn giản hơn."
    )
elif mean_a > mean_b:
    selected_name = "A_No_Chi2"
    selection_reason = (
        "Mô hình không Chi-square có Macro F1 cao hơn."
    )
else:
    selected_name = "B_With_Chi2"
    selection_reason = (
        "Mô hình có Chi-square có Macro F1 cao hơn."
    )

selected_model = loaded_models[selected_name]

# ============================================================
# 9. LƯU MÔ HÌNH CUỐI
# ============================================================

# Mô hình đã được GridSearchCV refit trên toàn bộ
# Train 800 mẫu trong thực nghiệm trước.
# Không sử dụng Test để huấn luyện.
final_path = (
    FINAL_MODEL_DIR / "uit_vsfc_final_model.joblib"
)

joblib.dump(selected_model, final_path)

metadata = {
    "dataset": "UIT-VSFC",
    "training_samples": 800,
    "labels": {
        "0": "Negative",
        "1": "Positive",
        "2": "Neutral"
    },
    "selected_candidate": selected_name,
    "selection_reason": selection_reason,
    "cv_macro_f1": float(
        mean_a if selected_name == "A_No_Chi2"
        else mean_b
    ),
    "cv_folds": 5,
    "random_state": RANDOM_STATE,
    "model_path": str(final_path),
    "input_column": "segmented_comments",
    "preprocessing": (
        "Vietnamese text preprocessing and "
        "Underthesea word segmentation"
    ),
    "note": (
        "Fixed-hyperparameter CV diagnostic. "
        "Hyperparameters were previously selected "
        "using the same Train data. "
        "Not an independent test estimate."
    )
}

with open(
    FINAL_MODEL_DIR / "final_model_metadata.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        metadata,
        file,
        ensure_ascii=False,
        indent=4
    )

# ============================================================
# 10. HOÀN THÀNH
# ============================================================

print("\n" + "=" * 75)
print("MÔ HÌNH ĐƯỢC LỰA CHỌN")
print("=" * 75)

print(f"Ứng viên: {selected_name}")
print(f"Lý do: {selection_reason}")
print(f"Đường dẫn: {final_path}")

print("\nĐã xuất:")
print("- final_candidate_fold_scores.csv")
print("- final_candidate_comparison.csv")
print("- final_candidate_fold_differences.csv")
print("- uit_vsfc_final_model.joblib")
print("- final_model_metadata.json")

print("\nKhông sử dụng Test 200 hoặc Official Test.")
