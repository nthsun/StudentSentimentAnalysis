
from pathlib import Path

import joblib
import pandas as pd

from scipy.sparse import load_npz

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold
)

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "alternative" / "processed"
MODEL_DIR = ROOT / "models" / "uit_vsfc"
RESULT_DIR = ROOT / "results"

METRIC_DIR = RESULT_DIR / "metrics" / "uit_vsfc"
REPORT_DIR = RESULT_DIR / "reports" / "uit_vsfc"
TABLE_DIR = RESULT_DIR / "tables" / "uit_vsfc"

for folder in [MODEL_DIR, METRIC_DIR, REPORT_DIR, TABLE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
CV_FOLDS = 5

LABEL_NAMES = ["Negative", "Positive", "Neutral"]
LABEL_IDS = [0, 1, 2]

CONFIGS = [
    "unigram",
    "unigram_bigram"
]


# ==========================================
# 2. LOAD FEATURES
# ==========================================

def load_features(config):

    prefix = f"uit_vsfc_{config}"

    X_train = load_npz(
        DATA_DIR / f"{prefix}_X_train.npz"
    )

    X_test = load_npz(
        DATA_DIR / f"{prefix}_X_test.npz"
    )

    y_train = pd.read_csv(
        DATA_DIR / f"{prefix}_y_train.csv"
    )["sentiments"]

    y_test = pd.read_csv(
        DATA_DIR / f"{prefix}_y_test.csv"
    )["sentiments"]

    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)
    assert X_train.shape[1] == X_test.shape[1]

    return X_train, X_test, y_train, y_test


# ==========================================
# 3. DEFINE MODELS AND PARAMETERS
# ==========================================

def get_models():

    models = {
        "Naive Bayes": (
            MultinomialNB(),
            {
                "alpha": [0.1, 0.5, 1.0, 2.0]
            }
        ),

        "Logistic Regression": (
            LogisticRegression(
                max_iter=3000,
                random_state=RANDOM_STATE
            ),
            {
                "C": [0.1, 1.0, 5.0, 10.0],
                "class_weight": [None, "balanced"]
            }
        ),

        "Linear SVM": (
            LinearSVC(
                random_state=RANDOM_STATE,
                max_iter=10000
            ),
            {
                "C": [0.1, 0.5, 1.0, 2.0, 5.0],
                "class_weight": [None, "balanced"]
            }
        )
    }

    return models


# ==========================================
# 4. CROSS VALIDATION AND TUNING
# ==========================================

def tune_model(model, param_grid, X_train, y_train):

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
        verbose=0
    )

    search.fit(X_train, y_train)

    return search


# ==========================================
# 5. EVALUATE ON TEST
# ==========================================

def evaluate_model(
    model,
    X_test,
    y_test,
    config,
    model_name
):

    predictions = model.predict(X_test)

    metrics = {
        "configuration": config,
        "model": model_name,
        "accuracy": accuracy_score(
            y_test, predictions
        ),
        "precision_macro": precision_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        ),
        "recall_macro": recall_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        ),
        "f1_macro": f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        ),
        "f1_weighted": f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )
    }

    # Báo cáo chi tiết
    report = classification_report(
        y_test,
        predictions,
        labels=LABEL_IDS,
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0
    )

    report_path = (
        REPORT_DIR /
        f"{config}_{model_name.lower().replace(' ', '_')}.txt"
    )

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(f"Configuration: {config}\n")
        file.write(f"Model: {model_name}\n\n")
        file.write(report)

    # Ma trận nhầm lẫn
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=LABEL_IDS
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=LABEL_NAMES,
        columns=LABEL_NAMES
    )

    matrix_df.to_csv(
        METRIC_DIR /
        f"{config}_{model_name.lower().replace(' ', '_')}_confusion_matrix.csv",
        encoding="utf-8-sig"
    )

    # Lưu dự đoán
    prediction_df = pd.DataFrame({
        "actual": y_test.to_numpy(),
        "predicted": predictions
    })

    prediction_df["actual_label"] = (
        prediction_df["actual"].map(
            dict(zip(LABEL_IDS, LABEL_NAMES))
        )
    )

    prediction_df["predicted_label"] = (
        prediction_df["predicted"].map(
            dict(zip(LABEL_IDS, LABEL_NAMES))
        )
    )

    prediction_df.to_csv(
        METRIC_DIR /
        f"{config}_{model_name.lower().replace(' ', '_')}_predictions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return metrics, report


# ==========================================
# 6. MAIN EXPERIMENT
# ==========================================

def main():

    all_results = []

    print("=" * 65)
    print("HUẤN LUYỆN MÔ HÌNH UIT-VSFC")
    print("=" * 65)

    for config in CONFIGS:

        print("\n" + "#" * 65)
        print(f"TF-IDF CONFIGURATION: {config.upper()}")
        print("#" * 65)

        X_train, X_test, y_train, y_test = load_features(config)

        print("Train:", X_train.shape)
        print("Test :", X_test.shape)

        models = get_models()

        for model_name, (model, param_grid) in models.items():

            print("\n" + "-" * 55)
            print("Đang huấn luyện:", model_name)

            search = tune_model(
                model,
                param_grid,
                X_train,
                y_train
            )

            best_model = search.best_estimator_

            print("Tham số tốt nhất:", search.best_params_)
            print(
                "CV Macro F1:",
                round(search.best_score_, 4)
            )

            # Lưu mô hình tốt nhất của tổ hợp
            model_filename = (
                f"uit_vsfc_{config}_"
                f"{model_name.lower().replace(' ', '_')}.pkl"
            )

            joblib.dump(
                best_model,
                MODEL_DIR / model_filename
            )

            # Đánh giá trên Test
            metrics, report = evaluate_model(
                best_model,
                X_test,
                y_test,
                config,
                model_name
            )

            metrics["cv_macro_f1"] = search.best_score_
            metrics["best_params"] = str(search.best_params_)

            all_results.append(metrics)

            print("\nKết quả Test:")
            print(report)

            print(
                "Test Macro F1:",
                round(metrics["f1_macro"], 4)
            )

    # ======================================
    # 7. SAVE COMPARISON
    # ======================================

    results_df = pd.DataFrame(all_results)

    results_df = results_df.sort_values(
        by="f1_macro",
        ascending=False
    )

    output_file = TABLE_DIR / "model_comparison.csv"

    results_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 65)
    print("HOÀN THÀNH THỰC NGHIỆM")
    print("=" * 65)

    print(
        results_df[
            [
                "configuration",
                "model",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "f1_weighted",
                "cv_macro_f1"
            ]
        ].to_string(index=False)
    )

    print("\nBảng kết quả:", output_file)


if __name__ == "__main__":
    main()
