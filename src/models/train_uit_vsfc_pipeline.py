
from pathlib import Path

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold
)


# =====================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# =====================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    ROOT_DIR / "data/alternative/processed/uit_vsfc_train.csv"
)

MODEL_DIR = ROOT_DIR / "models/uit_vsfc/pipeline"
RESULT_DIR = ROOT_DIR / "results/tables/uit_vsfc/pipeline"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. ĐỌC VÀ KIỂM TRA DỮ LIỆU
# =====================================================

print("=" * 65)
print("HUẤN LUYỆN PIPELINE - UIT-VSFC")
print("=" * 65)

if not TRAIN_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy dữ liệu Train: {TRAIN_PATH}"
    )

train_df = pd.read_csv(TRAIN_PATH)

required_columns = [
    "segmented_comments",
    "sentiments"
]

for column in required_columns:
    if column not in train_df.columns:
        raise ValueError(
            f"Thiếu cột dữ liệu: {column}"
        )

train_df = train_df.dropna(
    subset=required_columns
).copy()

X_train = train_df["segmented_comments"].astype(str)
y_train = train_df["sentiments"].astype(int)

if len(train_df) == 0:
    raise ValueError("Tập Train không có dữ liệu.")

if set(y_train.unique()) != {0, 1, 2}:
    raise ValueError(
        "Tập Train phải có đủ ba nhãn 0, 1, 2."
    )

print(f"Số mẫu huấn luyện: {len(train_df)}")

print("\nPhân bố nhãn:")
print(
    y_train.value_counts()
    .sort_index()
    .rename(index={
        0: "Negative",
        1: "Positive",
        2: "Neutral"
    })
)


# =====================================================
# 3. CẤU HÌNH CROSS VALIDATION
# =====================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

SCORING = "f1_macro"


# =====================================================
# 4. KHAI BÁO THUẬT TOÁN VÀ SIÊU THAM SỐ
# =====================================================

model_configs = {
    "Naive Bayes": {
        "estimator": MultinomialNB(),
        "parameters": {
            "classifier__alpha": [0.1, 0.5, 1.0, 2.0]
        }
    },

    "Logistic Regression": {
        "estimator": LogisticRegression(
            max_iter=3000,
            random_state=42
        ),
        "parameters": {
            "classifier__C": [0.1, 1, 5, 10],
            "classifier__class_weight": [None, "balanced"]
        }
    },

    "Linear SVM": {
        "estimator": LinearSVC(
            random_state=42,
            max_iter=10000
        ),
        "parameters": {
            "classifier__C": [0.1, 0.5, 1, 2, 5],
            "classifier__class_weight": [None, "balanced"]
        }
    }
}


# =====================================================
# 5. KHAI BÁO CẤU HÌNH TF-IDF
# =====================================================

tfidf_configs = {
    "unigram": {
        "ngram_range": (1, 1)
    },

    "unigram_bigram": {
        "ngram_range": (1, 2)
    }
}


# =====================================================
# 6. HÀM XÂY DỰNG PIPELINE
# =====================================================

def create_pipeline(classifier, ngram_range):

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                min_df=2,
                max_df=0.95,
                max_features=5000,
                sublinear_tf=True,
                lowercase=False,
                strip_accents=None,
                ngram_range=ngram_range
            )
        ),

        (
            "classifier",
            classifier
        )
    ])

    return pipeline


# =====================================================
# 7. HUẤN LUYỆN VÀ TÌM SIÊU THAM SỐ
# =====================================================

all_results = []
all_cv_details = []

total_experiments = (
    len(model_configs) * len(tfidf_configs)
)

experiment_number = 0

for tfidf_name, tfidf_config in tfidf_configs.items():

    for model_name, model_config in model_configs.items():

        experiment_number += 1

        print("\n" + "=" * 65)
        print(
            f"THỰC NGHIỆM {experiment_number}/{total_experiments}"
        )
        print(f"TF-IDF: {tfidf_name}")
        print(f"Model: {model_name}")
        print("=" * 65)

        pipeline = create_pipeline(
            classifier=model_config["estimator"],
            ngram_range=tfidf_config["ngram_range"]
        )

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=model_config["parameters"],
            scoring=SCORING,
            cv=cv,
            n_jobs=-1,
            refit=True,
            return_train_score=True,
            verbose=1,
            error_score="raise"
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        best_score = grid_search.best_score_

        best_params = grid_search.best_params_

        # Số đặc trưng TF-IDF của mô hình tốt nhất
        feature_count = len(
            best_model.named_steps["tfidf"].get_feature_names_out()
        )

        print("\nKẾT QUẢ TỐT NHẤT:")
        print(f"Model: {model_name}")
        print(f"Configuration: {tfidf_name}")
        print(f"CV Macro F1: {best_score:.4f}")
        print(f"Best Params: {best_params}")
        print(f"Số đặc trưng TF-IDF: {feature_count}")

        # Lưu mô hình tốt nhất
        model_filename = (
            f"pipeline_{tfidf_name}_"
            f"{model_name.lower().replace(' ', '_')}.joblib"
        )

        model_path = MODEL_DIR / model_filename

        joblib.dump(
            best_model,
            model_path
        )

        # Lưu kết quả tổng hợp
        all_results.append({
            "configuration": tfidf_name,
            "model": model_name,
            "cv_macro_f1": best_score,
            "best_params": str(best_params),
            "feature_count": feature_count,
            "model_path": str(model_path)
        })

        # Lưu chi tiết kết quả GridSearchCV
        cv_details = pd.DataFrame(
            grid_search.cv_results_
        )

        cv_details["configuration"] = tfidf_name
        cv_details["model"] = model_name

        all_cv_details.append(cv_details)


# =====================================================
# 8. TỔNG HỢP KẾT QUẢ 6 THỰC NGHIỆM
# =====================================================

results_df = pd.DataFrame(all_results)

results_df = results_df.sort_values(
    by="cv_macro_f1",
    ascending=False
).reset_index(drop=True)

results_df.insert(
    0,
    "rank",
    range(1, len(results_df) + 1)
)

summary_path = (
    RESULT_DIR / "pipeline_cv_comparison.csv"
)

results_df.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig"
)


# =====================================================
# 9. LƯU CHI TIẾT GRID SEARCH
# =====================================================

details_df = pd.concat(
    all_cv_details,
    ignore_index=True
)

details_path = (
    RESULT_DIR / "pipeline_grid_search_details.csv"
)

details_df.to_csv(
    details_path,
    index=False,
    encoding="utf-8-sig"
)


# =====================================================
# 10. HIỂN THỊ BẢNG KẾT QUẢ
# =====================================================

print("\n" + "=" * 75)
print("BẢNG SO SÁNH PIPELINE - STRATIFIED 5-FOLD CV")
print("=" * 75)

print(
    results_df[
        [
            "rank",
            "configuration",
            "model",
            "cv_macro_f1",
            "feature_count",
            "best_params"
        ]
    ].to_string(
        index=False,
        formatters={
            "cv_macro_f1": lambda x: f"{x:.4f}"
        }
    )
)


# =====================================================
# 11. XÁC ĐỊNH MÔ HÌNH TỐT NHẤT
# =====================================================

best_result = results_df.iloc[0]

print("\n" + "=" * 65)
print("PIPELINE TỐT NHẤT")
print("=" * 65)

print(f"Model: {best_result['model']}")
print(f"Configuration: {best_result['configuration']}")
print(f"CV Macro F1: {best_result['cv_macro_f1']:.4f}")
print(f"Best Params: {best_result['best_params']}")


# =====================================================
# 12. KẾT THÚC
# =====================================================

print("\nĐÃ LƯU KẾT QUẢ:")

print(f"- {summary_path}")
print(f"- {details_path}")
print(f"- {MODEL_DIR}")

print("\nHoàn thành huấn luyện Pipeline UIT-VSFC!")
