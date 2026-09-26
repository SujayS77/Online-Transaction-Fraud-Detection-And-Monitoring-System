"""
Feature engineering: turns raw Transaction fields into numeric features
a model can actually learn from.

WHY we can't just feed the raw columns to XGBoost:
- `device_id`, `merchant_id`, `user_id` are high-cardinality IDs (thousands
  of unique values). One-hot encoding them would create thousands of
  useless columns, and a *new* device/merchant at inference time would be
  a category the model never saw during training - the model can't reason
  about "unfamiliarity" from a raw ID alone.
- What actually signals fraud isn't the ID itself, it's the RELATIONSHIP
  between the transaction and the user's own history: is this amount
  unusual FOR THIS USER, is this location unusual FOR THIS USER, have we
  seen this exact device before FOR THIS USER. So instead of encoding raw
  IDs, we engineer relative/behavioral features per user:

    amount_ratio_to_user_avg  -> amount / this user's average amount
                                  (captures the "5-15x spike" fraud pattern)
    device_freq_for_user      -> how many times we've seen this user use
                                  this exact device (low = suspicious;
                                  captures the "never-seen device" pattern)
    is_foreign_location       -> 1 if this location != user's most common
                                  location, else 0
    hour_of_day               -> extracted from timestamp (fraud sometimes
                                  clusters at odd hours - lets the model
                                  find that pattern if it exists)
    card_present              -> already boolean, just cast to int
    transaction_type          -> low-cardinality (4 values) - safe to
                                  one-hot encode directly

This file is used both by the training script and (later) by the
real-time inference service, so features are computed identically in both
places - a very common source of subtle bugs in ML systems if you don't
centralize this.
"""

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- per-user historical stats, computed from the data itself ---
    user_avg_amount = df.groupby("user_id")["amount"].transform("mean")
    df["amount_ratio_to_user_avg"] = df["amount"] / user_avg_amount

    device_counts = df.groupby(["user_id", "device_id"])["device_id"].transform("count")
    df["device_freq_for_user"] = device_counts

    user_common_location = df.groupby("user_id")["location"].transform(
        lambda x: x.mode()[0]
    )
    df["is_foreign_location"] = (df["location"] != user_common_location).astype(int)

    # --- direct transforms ---
    df["hour_of_day"] = pd.to_datetime(df["timestamp"]).dt.hour
    df["card_present"] = df["card_present"].astype(int)

    txn_type_dummies = pd.get_dummies(df["transaction_type"], prefix="type")

    feature_cols = [
        "amount",
        "amount_ratio_to_user_avg",
        "device_freq_for_user",
        "is_foreign_location",
        "hour_of_day",
        "card_present",
    ]
    X = pd.concat([df[feature_cols], txn_type_dummies], axis=1)
    return X


FEATURE_NAMES_NOTE = (
    "Call build_features(df) to get X. "
    "y is simply df['is_fraud'] for labeled/synthetic data - "
    "real data won't have this until analysts confirm flagged cases."
)