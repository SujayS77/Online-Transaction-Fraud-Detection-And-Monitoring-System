"""
Batch dataset builder for offline model training.

WHY this is a separate script from generator/simulate.py, even though it
reuses the exact same UserProfile logic:
simulate.py's job is to mimic a LIVE, never-ending transaction feed (one
event at a time, pushed to Redis, running forever). Training a model needs
a big, finished, static batch of historical data sitting in memory/disk —
a fundamentally different shape of data (a DataFrame you can split into
train/test) than a live stream. Reusing UserProfile means both paths
generate data with identical fraud patterns, so a model trained on this
batch will transfer correctly to what the live stream produces later.

Run this once to produce data/transactions.csv.
"""

import sys
import os
import random

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generator.simulate import UserProfile, FRAUD_RATE
from common.models import Transaction

N_USERS = 300
N_TRANSACTIONS = 20000
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "transactions.csv")


def generate_dataset(n_transactions=N_TRANSACTIONS, n_users=N_USERS, seed=42) -> pd.DataFrame:
    random.seed(seed)
    users = [UserProfile(f"user_{i}") for i in range(n_users)]

    rows = []
    for _ in range(n_transactions):
        user = random.choice(users)
        txn = (user.fraud_transaction() if random.random() < FRAUD_RATE
               else user.normal_transaction())
        rows.append(txn.__dict__)

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    fraud_count = df["is_fraud"].sum()
    print(f"Generated {len(df)} transactions -> {OUTPUT_PATH}")
    print(f"Fraud: {fraud_count} ({fraud_count/len(df)*100:.2f}%) | "
          f"Normal: {len(df) - fraud_count} ({(len(df)-fraud_count)/len(df)*100:.2f}%)")