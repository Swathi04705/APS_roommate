"""
ML Model Evaluation Assignment
Dataset: Roommate compatibility training_data.csv

Compares Logistic Regression (baseline) and Random Forest (candidate improved model)
on the same held-out test set. Uses grouped splitting to keep duplicate student
pairs together, paired bootstrap confidence intervals, and an exact one-sided
McNemar test.

Run:
    python assignment.py
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import binomtest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ----------------------------- CONFIGURATION -----------------------------

RANDOM_STATE = 42
N_BOOTSTRAP = 5000
CONFIDENCE_LEVEL = 0.95
COMPATIBILITY_THRESHOLD = 70

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "training_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_and_prepare_data():
    """Load the CSV, create the binary target, and construct pair groups."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place training_data.csv inside the project's data/ folder."
        )

    df = pd.read_csv(DATA_PATH)
    print("\n" + "=" * 70)
    print("1. DATASET INFORMATION")
    print("=" * 70)
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isna().sum().sum()}")
    print(f"Compatibility threshold: {COMPATIBILITY_THRESHOLD}")

    required_target = "compatibility_score"
    if required_target not in df.columns:
        raise ValueError(f"Required target column '{required_target}' not found.")

    # Convert the score into a binary classification target.
    y = (df[required_target] >= COMPATIBILITY_THRESHOLD).astype(int)

    print("\nClass distribution:")
    print(y.value_counts().sort_index().rename(index={
        0: "Not Compatible (0)", 1: "Compatible (1)"
    }))
    print("\nClass percentages:")
    print((y.value_counts(normalize=True).sort_index() * 100).round(2))

    # A canonical pair key makes (student 1, student 2) and
    # (student 2, student 1) the same group.
    pair_a = df["student1_id"].astype(str)
    pair_b = df["student2_id"].astype(str)
    groups = [
        "|".join(sorted((a, b))) for a, b in zip(pair_a, pair_b)
    ]

    # Exclude IDs, names, free-text bios, and the original target.
    # IDs/names are identifiers rather than generalizable compatibility features.
    columns_to_drop = [
        "student1_id", "student1_name", "student1_bio",
        "student2_id", "student2_name", "student2_bio",
        "compatibility_score",
    ]
    X = df.drop(columns=columns_to_drop, errors="ignore").copy()

    # Normalize column names only for consistent handling; feature values remain.
    categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_columns = X.select_dtypes(include=["number", "bool"]).columns.tolist()

    print(f"\nFeatures used: {X.shape[1]}")
    print(f"Numerical features: {len(numerical_columns)}")
    print(f"Categorical features: {len(categorical_columns)}")
    print(f"Unique student-pair groups: {len(set(groups))}")

    return X, y, np.asarray(groups), numerical_columns, categorical_columns


def make_preprocessor(numerical_columns, categorical_columns):
    """Create preprocessing that is fitted only on training data."""
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def make_grouped_train_test_split(X, y, groups):
    """
    Use the first fold of StratifiedGroupKFold (5 folds).
    This gives approximately 80% training and 20% test data while ensuring
    duplicate records of the same unordered student pair stay in one split.
    """
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    overlap = train_groups.intersection(test_groups)
    if overlap:
        raise RuntimeError("Group leakage detected between train and test sets.")

    print("\n" + "=" * 70)
    print("2. SAMPLING AND DATA SPLIT")
    print("=" * 70)
    print(f"Training records: {len(train_idx)}")
    print(f"Testing records:  {len(test_idx)}")
    print(f"Training class proportions:\n{y_train.value_counts(normalize=True).round(3)}")
    print(f"Testing class proportions:\n{y_test.value_counts(normalize=True).round(3)}")
    print(f"Overlapping pair groups: {len(overlap)}")

    return X_train, X_test, y_train, y_test


def build_models(preprocessor):
    """Define baseline and candidate improved models."""
    baseline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    improved = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    return baseline, improved


def evaluate_model(name, model, X_test, y_test):
    """Calculate standard classification metrics."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    results = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1-score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    }
    return results, predictions


def paired_bootstrap_accuracy_difference(
    y_true, baseline_predictions, improved_predictions,
    n_bootstrap=N_BOOTSTRAP, confidence_level=CONFIDENCE_LEVEL
):
    """
    Paired bootstrap: resample test-row indices and calculate
    accuracy(improved) - accuracy(baseline) on exactly the same sampled rows.
    """
    y_true = np.asarray(y_true)
    baseline_predictions = np.asarray(baseline_predictions)
    improved_predictions = np.asarray(improved_predictions)

    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y_true)
    differences = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        sample_indices = rng.integers(0, n, size=n)
        baseline_acc = accuracy_score(
            y_true[sample_indices], baseline_predictions[sample_indices]
        )
        improved_acc = accuracy_score(
            y_true[sample_indices], improved_predictions[sample_indices]
        )
        differences[i] = improved_acc - baseline_acc

    alpha = 1 - confidence_level
    lower = np.quantile(differences, alpha / 2)
    upper = np.quantile(differences, 1 - alpha / 2)
    observed_difference = (
        accuracy_score(y_true, improved_predictions)
        - accuracy_score(y_true, baseline_predictions)
    )

    return observed_difference, lower, upper, differences


def exact_mcnemar_test(y_true, baseline_predictions, improved_predictions):
    """
    Exact one-sided McNemar test.
    b = baseline wrong, improved correct.
    c = baseline correct, improved wrong.
    H1: improved model has a higher probability of being correct.
    """
    y_true = np.asarray(y_true)
    baseline_predictions = np.asarray(baseline_predictions)
    improved_predictions = np.asarray(improved_predictions)

    baseline_correct = baseline_predictions == y_true
    improved_correct = improved_predictions == y_true

    b = int(np.sum((~baseline_correct) & improved_correct))
    c = int(np.sum(baseline_correct & (~improved_correct)))
    discordant = b + c

    if discordant == 0:
        p_value = 1.0
    else:
        p_value = binomtest(
            k=b, n=discordant, p=0.5, alternative="greater"
        ).pvalue

    return b, c, p_value


def save_confusion_matrix(y_test, predictions, title, filename):
    matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Not Compatible", "Compatible"],
        yticklabels=["Not Compatible", "Compatible"],
    )
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()


def main():
    X, y, groups, numerical_columns, categorical_columns = load_and_prepare_data()

    X_train, X_test, y_train, y_test = make_grouped_train_test_split(X, y, groups)

    # Separate preprocessing objects ensure each pipeline learns transformations
    # from training data only.
    baseline_preprocessor = make_preprocessor(numerical_columns, categorical_columns)
    improved_preprocessor = make_preprocessor(numerical_columns, categorical_columns)
    baseline, improved = build_models(baseline_preprocessor)
    improved.set_params(preprocessor=improved_preprocessor)

    print("\n" + "=" * 70)
    print("3. MODEL TRAINING")
    print("=" * 70)

    baseline.fit(X_train, y_train)
    improved.fit(X_train, y_train)
    print("Logistic Regression training completed.")
    print("Random Forest training completed.")

    print("\n" + "=" * 70)
    print("4. HELD-OUT TEST SET EVALUATION")
    print("=" * 70)

    baseline_results, baseline_predictions = evaluate_model(
        "Logistic Regression (Baseline)", baseline, X_test, y_test
    )
    improved_results, improved_predictions = evaluate_model(
        "Random Forest (Candidate Improved)", improved, X_test, y_test
    )

    metrics_df = pd.DataFrame([baseline_results, improved_results])
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    print("\nBaseline classification report:")
    print(classification_report(
        y_test, baseline_predictions,
        target_names=["Not Compatible", "Compatible"],
        zero_division=0,
    ))
    print("Random Forest classification report:")
    print(classification_report(
        y_test, improved_predictions,
        target_names=["Not Compatible", "Compatible"],
        zero_division=0,
    ))

    save_confusion_matrix(
        y_test, baseline_predictions,
        "Logistic Regression Confusion Matrix",
        "baseline_confusion_matrix.png",
    )
    save_confusion_matrix(
        y_test, improved_predictions,
        "Random Forest Confusion Matrix",
        "improved_confusion_matrix.png",
    )

    # Save paired predictions so the evaluation can be inspected/reproduced.
    prediction_df = pd.DataFrame({
        "actual": np.asarray(y_test),
        "baseline_prediction": baseline_predictions,
        "improved_prediction": improved_predictions,
    })
    prediction_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    print("\n" + "=" * 70)
    print("5. BOOTSTRAP CONFIDENCE INTERVAL")
    print("=" * 70)
    observed_diff, ci_lower, ci_upper, bootstrap_diffs = (
        paired_bootstrap_accuracy_difference(
            y_test, baseline_predictions, improved_predictions
        )
    )
    print(f"Observed accuracy improvement (RF - LR): {observed_diff:.4f}")
    print(f"Observed improvement in percentage points: {observed_diff * 100:.2f}")
    print(
        f"{CONFIDENCE_LEVEL:.0%} bootstrap CI: "
        f"[{ci_lower:.4f}, {ci_upper:.4f}]"
    )

    plt.figure(figsize=(7, 4))
    plt.hist(bootstrap_diffs, bins=40, edgecolor="black", alpha=0.75)
    plt.axvline(0, linestyle="--", color="red", label="No difference")
    plt.axvline(observed_diff, linestyle="-", color="black", label="Observed difference")
    plt.xlabel("Accuracy difference (Random Forest - Logistic Regression)")
    plt.ylabel("Bootstrap frequency")
    plt.title("Paired Bootstrap Distribution of Accuracy Difference")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bootstrap_difference_distribution.png", dpi=300)
    plt.close()

    print("\n" + "=" * 70)
    print("6. EXACT ONE-SIDED McNEMAR HYPOTHESIS TEST")
    print("=" * 70)
    b, c, p_value = exact_mcnemar_test(
        y_test, baseline_predictions, improved_predictions
    )
    alpha = 0.05
    print(f"Baseline wrong, Random Forest correct (b): {b}")
    print(f"Baseline correct, Random Forest wrong (c): {c}")
    print(f"Significance level (alpha): {alpha}")
    print(f"One-sided exact McNemar p-value: {p_value:.6f}")

    print("\nHypotheses:")
    print("H0: The improved model is not more accurate than the baseline model.")
    print("H1: The improved model is more accurate than the baseline model.")

    if p_value < alpha:
        decision = "Reject H0: evidence supports higher accuracy for Random Forest."
    else:
        decision = "Fail to reject H0: insufficient evidence that Random Forest is more accurate."

    ci_supports_positive = ci_lower > 0
    print(f"Statistical decision: {decision}")
    print(
        "Bootstrap CI interpretation: "
        + (
            "The entire 95% interval is above zero."
            if ci_supports_positive
            else "The 95% interval includes zero or negative values."
        )
    )

    summary = pd.DataFrame([{
        "observed_accuracy_difference_RF_minus_LR": observed_diff,
        "bootstrap_95CI_lower": ci_lower,
        "bootstrap_95CI_upper": ci_upper,
        "mcnemar_b_baseline_wrong_rf_correct": b,
        "mcnemar_c_baseline_correct_rf_wrong": c,
        "mcnemar_one_sided_p_value": p_value,
        "significance_level": alpha,
        "decision": decision,
    }])
    summary.to_csv(OUTPUT_DIR / "statistical_test_results.csv", index=False)

    print("\n" + "=" * 70)
    print("7. WHERE PROBABILITY AND STATISTICS ARE USED")
    print("=" * 70)
    print("1. Stratified group splitting: creates training and held-out test samples.")
    print("2. Bootstrap sampling: estimates variation in the paired accuracy difference.")
    print("3. Confidence interval: quantifies uncertainty around the observed improvement.")
    print("4. McNemar test: compares paired correctness outcomes on the same test cases.")
    print("5. P-value and alpha: support the hypothesis-test decision.")

    print("\n" + "=" * 70)
    print("8. OUTPUT FILES")
    print("=" * 70)
    print(f"All results saved in: {OUTPUT_DIR}")
    print(" - model_metrics.csv")
    print(" - test_predictions.csv")
    print(" - statistical_test_results.csv")
    print(" - baseline_confusion_matrix.png")
    print(" - improved_confusion_matrix.png")
    print(" - bootstrap_difference_distribution.png")

    print("\nIMPORTANT:")
    print("Do not claim Random Forest is better unless the metrics and statistical")
    print("tests support that conclusion. Report the actual output values.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
