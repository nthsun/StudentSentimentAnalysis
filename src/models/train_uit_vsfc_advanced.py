# f1 - f5
from pathlib import Path
import json
import warnings

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score

from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# ============================================================
# 1. ĐƯỜNG DẪN VÀ CẤU HÌNH
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    ROOT / "data" / "alternative" / "processed"
    / "uit_vsfc_train.csv"
)

MODEL_DIR = ROOT / "models" / "uit_vsfc" / "advanced"
RESULT_DIR = (
    ROOT / "results" / "tables" / "uit_vsfc" / "advanced"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5

# ============================================================
# 2. ĐỌC VÀ KIỂM TRA DỮ LIỆU TRAIN
# ============================================================

print("=" * 75)
print("THỰC NGHIỆM MỞ RỘNG UIT-VSFC")
print("=" * 75)

if not TRAIN_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy dữ liệu Train: {TRAIN_PATH}"
    )

df = pd.read_csv(TRAIN_PATH)

required_columns = ["segmented_comments", "sentiments"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Thiếu cột: {column}")

df = df.dropna(subset=required_columns).copy()

X = df["segmented_comments"].astype(str)
y = df["sentiments"].astype(int)

expected_labels = {0, 1, 2}

if set(y.unique()) != expected_labels:
    raise ValueError(
        f"Nhãn không hợp lệ: {sorted(y.unique())}"
    )

print(f"\nSố mẫu Train: {len(df)}")
print("\nPhân bố nhãn:")
print(y.value_counts().sort_index())

# ============================================================
# 3. ĐỊNH NGHĨA CÁC PHƯƠNG PHÁP BIỂU DIỄN
# ============================================================

def make_word_vectorizer(ngram_range):
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95,
        max_features=10000,
        sublinear_tf=True,
        lowercase=False,
        strip_accents=None
    )


def make_char_vectorizer():
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        max_df=0.95,
        max_features=15000,
        sublinear_tf=True,
        lowercase=False,
        strip_accents=None
    )


FEATURE_CONFIGS = {
    "F1_Word_Unigram": make_word_vectorizer((1, 1)),

    "F2_Word_Unigram_Bigram": make_word_vectorizer((1, 2)),

    "F3_Word_Unigram_Bigram_Trigram":
        make_word_vectorizer((1, 3)),

    "F4_Character_Ngram": make_char_vectorizer(),

    "F5_Word_Character": FeatureUnion([
        (
            "word",
            make_word_vectorizer((1, 2))
        ),
        (
            "character",
            make_char_vectorizer()
        )
    ])
}

# ============================================================
# 4. ĐỊNH NGHĨA CÁC THUẬT TOÁN
# ============================================================

CLASSIFIERS = {
    "MultinomialNB": {
        "estimator": MultinomialNB(),
        "params": {
            "classifier__alpha": [0.1, 0.5, 1.0, 2.0]
        }
    },

    "ComplementNB": {
        "estimator": ComplementNB(),
        "params": {
            "classifier__alpha": [0.1, 0.5, 1.0, 2.0]
        }
    },

    "LogisticRegression": {
        "estimator": LogisticRegression(
            max_iter=5000,
            random_state=RANDOM_STATE
        ),
        "params": {
            "classifier__C": [0.1, 1, 5, 10],
            "classifier__class_weight": [None, "balanced"]
        }
    },

    "LinearSVM": {
        "estimator": LinearSVC(
            random_state=RANDOM_STATE,
            max_iter=20000
        ),
        "params": {
            "classifier__C": [0.1, 0.5, 1, 2, 5],
            "classifier__class_weight": [None, "balanced"]
        }
    },

    "RandomForest": {
        "estimator": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "params": {
            "classifier__n_estimators": [200, 400],
            "classifier__max_depth": [None, 30],
            "classifier__min_samples_leaf": [1, 2]
        }
    }
}

# ============================================================
# 5. CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

scorer = make_scorer(
    f1_score,
    average="macro",
    zero_division=0
)

results = []
best_models = []

total_experiments = (
    len(FEATURE_CONFIGS) * len(CLASSIFIERS)
)

experiment_number = 0

for feature_name, feature_transformer in FEATURE_CONFIGS.items():

    for model_name, model_config in CLASSIFIERS.items():

        experiment_number += 1

        print("\n" + "=" * 75)
        print(
            f"THỰC NGHIỆM {experiment_number}/{total_experiments}"
        )
        print(f"Feature: {feature_name}")
        print(f"Model: {model_name}")
        print("=" * 75)

        pipeline = Pipeline([
            ("features", feature_transformer),
            ("classifier", model_config["estimator"])
        ])

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=model_config["params"],
            scoring=scorer,
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=0,
            return_train_score=True,
            error_score="raise"
        )

        grid_search.fit(X, y)

        best_model = grid_search.best_estimator_
        best_score = grid_search.best_score_

        best_params = grid_search.best_params_

        safe_name = f"{feature_name}_{model_name}"

        model_path = MODEL_DIR / f"{safe_name}.joblib"

        joblib.dump(best_model, model_path)

        # Lưu chi tiết kết quả GridSearchCV
        detail_df = pd.DataFrame(grid_search.cv_results_)

        detail_path = (
            RESULT_DIR / f"grid_{safe_name}.csv"
        )

        detail_df.to_csv(
            detail_path,
            index=False,
            encoding="utf-8-sig"
        )

        result = {
            "feature": feature_name,
            "model": model_name,
            "cv_macro_f1": best_score,
            "best_params": json.dumps(
                best_params,
                ensure_ascii=False,
                default=str
            ),
            "model_path": str(model_path)
        }

        results.append(result)

        best_models.append({
            "feature": feature_name,
            "model": model_name,
            "score": best_score,
            "path": str(model_path)
        })

        print(f"Best CV Macro F1: {best_score:.4f}")
        print(f"Best parameters: {best_params}")
        print(f"Đã lưu mô hình: {model_path}")

        # Giải phóng bộ nhớ của GridSearchCV
        del grid_search
        del pipeline
        del best_model

# ============================================================
# 6. TỔNG HỢP KẾT QUẢ
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "cv_macro_f1",
    ascending=False
).reset_index(drop=True)

results_df.insert(
    0,
    "rank",
    range(1, len(results_df) + 1)
)

results_df["cv_macro_f1"] = (
    results_df["cv_macro_f1"].round(4)
)

comparison_path = (
    RESULT_DIR / "advanced_cv_comparison.csv"
)

results_df.to_csv(
    comparison_path,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# 7. TỔNG HỢP THEO PHƯƠNG PHÁP ĐẶC TRƯNG
# ============================================================

feature_summary = (
    results_df.groupby("feature")["cv_macro_f1"]
    .agg(["mean", "max", "min"])
    .reset_index()
)

feature_summary = feature_summary.sort_values(
    "max",
    ascending=False
)

feature_summary.to_csv(
    RESULT_DIR / "feature_method_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# 8. TỔNG HỢP THEO THUẬT TOÁN
# ============================================================

model_summary = (
    results_df.groupby("model")["cv_macro_f1"]
    .agg(["mean", "max", "min"])
    .reset_index()
)

model_summary = model_summary.sort_values(
    "max",
    ascending=False
)

model_summary.to_csv(
    RESULT_DIR / "classifier_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# 9. HIỂN THỊ KẾT QUẢ
# ============================================================

print("\n" + "=" * 100)
print("BẢNG XẾP HẠNG 25 TỔ HỢP THỰC NGHIỆM")
print("=" * 100)

print(
    results_df[
        ["rank", "feature", "model", "cv_macro_f1"]
    ].to_string(index=False)
)

print("\n" + "=" * 75)
print("PHƯƠNG PHÁP BIỂU DIỄN TỐT NHẤT")
print("=" * 75)

print(feature_summary.to_string(index=False))

print("\n" + "=" * 75)
print("THUẬT TOÁN TỐT NHẤT")
print("=" * 75)

print(model_summary.to_string(index=False))

best = results_df.iloc[0]

print("\n" + "=" * 75)
print("TỔ HỢP CÓ CV MACRO F1 CAO NHẤT")
print("=" * 75)

print(f"Feature: {best['feature']}")
print(f"Model: {best['model']}")
print(f"CV Macro F1: {best['cv_macro_f1']}")
print(f"Model path: {best['model_path']}")

print("\n" + "=" * 75)
print("HOÀN THÀNH THỰC NGHIỆM")
print("=" * 75)

print(f"Bảng so sánh: {comparison_path}")
print(f"Thư mục mô hình: {MODEL_DIR}")
print(f"Thư mục kết quả: {RESULT_DIR}")

print(
    "\nLưu ý: Chương trình chỉ sử dụng Train 800 mẫu "
    "và Cross-validation. Không sử dụng Test 200."
)
