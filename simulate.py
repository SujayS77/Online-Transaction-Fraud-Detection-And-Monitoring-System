"""
Synthetic transaction generator.

WHY it works this way: each fake "user" has a stable profile (usual spend
range, home location, usual device). Most transactions are drawn from that
profile ("normal" behavior). A small percentage deliberately BREAK the
profile in specific, realistic ways — that's what makes them "fraud" and
also what will make them learnable patterns for a model later, instead of
pure random noise a model could never catch.

Run this file directly to start pushing fake transactions onto the Redis
Stream "transactions" forever, at one every SLEEP_SECONDS.
"""

import random
import time
import sys
import os

import redis

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.models import Transaction

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
STREAM_NAME = "transactions"
SLEEP_SECONDS = float(os.environ.get("GEN_INTERVAL", 0.5))
FRAUD_RATE = 0.02  # ~2% of transactions injected as fraud

BANKS = ["bank_a", "bank_b", "bank_c"]
LOCATIONS = ["MUM-IN", "DEL-IN", "BLR-IN", "NYC-US", "LON-UK", "SIN-SG"]
TXN_TYPES = ["purchase", "withdrawal", "transfer", "online_payment"]


class UserProfile:
    """A stable fake identity so 'normal' behavior means something."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.bank_id = random.choice(BANKS)
        self.home_location = random.choice(LOCATIONS)
        self.usual_device = f"device_{random.randint(1000, 9999)}"
        self.avg_amount = round(random.uniform(200, 5000), 2)

    def normal_transaction(self) -> Transaction:
        amount = round(max(1, random.gauss(self.avg_amount, self.avg_amount * 0.25)), 2)
        return Transaction.new(
            user_id=self.user_id,
            bank_id=self.bank_id,
            merchant_id=f"merchant_{random.randint(1, 500)}",
            amount=amount,
            location=self.home_location,
            device_id=self.usual_device,
            card_present=random.choice([True, False]),
            transaction_type=random.choice(TXN_TYPES),
            is_fraud=False,
        )

    def fraud_transaction(self) -> Transaction:
        """
        Injects one of a few realistic fraud patterns rather than pure
        randomness, so the signal is actually learnable:
          - sudden high-value spike (5-15x the user's normal spend)
          - new device + unfamiliar location combo
          - a foreign location inconsistent with the user's home base
        """
        pattern = random.choice(["spike", "new_device_location"])
        if pattern == "spike":
            amount = round(self.avg_amount * random.uniform(5, 15), 2)
            location, device = self.home_location, self.usual_device
        else:
            amount = round(self.avg_amount * random.uniform(1, 3), 2)
            location = random.choice([l for l in LOCATIONS if l != self.home_location])
            device = f"device_{random.randint(10000, 99999)}"  # never-seen device

        return Transaction.new(
            user_id=self.user_id,
            bank_id=self.bank_id,
            merchant_id=f"merchant_{random.randint(1, 500)}",
            amount=amount,
            location=location,
            device_id=device,
            card_present=False,
            transaction_type=random.choice(TXN_TYPES),
            is_fraud=True,
        )


def main():
    client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    users = [UserProfile(f"user_{i}") for i in range(200)]

    print(f"Pushing transactions to Redis Stream '{STREAM_NAME}' "
          f"every {SLEEP_SECONDS}s. Ctrl+C to stop.")

    while True:
        user = random.choice(users)
        txn = (user.fraud_transaction() if random.random() < FRAUD_RATE
               else user.normal_transaction())

        client.xadd(STREAM_NAME, {"data": txn.to_json()})
        tag = "FRAUD" if txn.is_fraud else "normal"
        print(f"[{tag}] {txn.transaction_id[:8]} user={txn.user_id} "
              f"amount={txn.amount} loc={txn.location}")

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
