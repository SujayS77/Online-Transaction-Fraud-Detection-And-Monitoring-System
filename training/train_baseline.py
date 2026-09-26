"""
Baseline fraud classifier - NO Federated Learning, NO GAN yet.

WHY start here instead of jumping straight to the fancy stuff:
you need a known-good, simple baseline to compare everything else against.
When you later add GAN-augmented training data or split this into
Federated clients, the only way to honestly claim "this made it better" is
to have this baseline's numbers written down first. Skipping this step is
exactly what turns a legitimate architecture decision into "I added a GAN
because it sounded impressive" in an interview - don't skip it.

WHY XGBoost specifically for the baseline:
handles the mixed numeric/categorical-encoded features here well with no
scaling required, trains fast even on this small dataset, and has a
built-in, well-understood way to handle class imbalance
(`scale_pos_weight`) - see below.
"""

import sys
import os

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.features import build_features

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "transactions.csv")
MODEL_OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "baseline_xgb.pkl")


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} transactions "
          f"({df['is_fraud'].sum()} fraud, {df['is_fraud'].mean()*100:.2f}%)")

    X = build_features(df)
    y = df["is_fraud"].astype(int)

    # stratify=y keeps the same fraud/normal RATIO in both train and test -
    # without this, a random split could easily leave the test set with
    # almost no fraud examples at all, making the evaluation meaningless.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} ({y_train.sum()} fraud) | "
          f"Test: {len(X_test)} ({y_test.sum()} fraud)")

    # scale_pos_weight tells XGBoost "each fraud example counts as if it
    # were N normal examples" during training - without this, the model
    # can hit 98% accuracy by just predicting "not fraud" every time,
    # since fraud is only ~2% of the data. This is the single most
    # important line in this file - be ready to explain it.
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos
    print(f"scale_pos_weight = {scale_pos_weight:.1f} "
          f"(telling the model to treat each fraud example as ~{scale_pos_weight:.0f}x more important)")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",   # PR-AUC, not accuracy - see note below
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n--- Evaluation ---")
    # WHY NOT just accuracy: with ~2% fraud, predicting "never fraud" gets
    # ~98% accuracy while being completely useless. Precision/recall and
    # PR-AUC (not ROC-AUC) are the honest metrics for imbalanced problems
    # like this - be ready to explain this distinction in an interview.
    print(classification_report(y_test, y_pred, target_names=["normal", "fraud"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"PR-AUC (average precision): {average_precision_score(y_test, y_proba):.4f}")

    os.makedirs(os.path.dirname(MODEL_OUT_PATH), exist_ok=True)
    joblib.dump({"model": model, "feature_names": list(X.columns)}, MODEL_OUT_PATH)
    print(f"\nModel saved to {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()