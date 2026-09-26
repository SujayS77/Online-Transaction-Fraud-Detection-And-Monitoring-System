"""
Shared data model for a transaction event.

WHY a dataclass + to_json/from_json instead of just passing dicts around:
every part of the system (generator, adapters, FL clients, API) needs the
exact same shape of a transaction. If we let each part build its own dict
with slightly different keys, we get silent bugs later (e.g. "amount" vs
"transaction_amount"). One typed class = one source of truth for the schema.
"""

from dataclasses import dataclass, asdict
import json
import uuid
from datetime import datetime, timezone


@dataclass
class Transaction:
    transaction_id: str
    timestamp: str          # ISO 8601 string (JSON-safe, human-readable)
    user_id: str
    merchant_id: str
    amount: float
    location: str            # simplified as a city/country code for now
    device_id: str
    card_present: bool
    transaction_type: str    # e.g. "purchase", "withdrawal", "transfer"
    bank_id: str             # WHICH simulated bank this belongs to.
                              # This field is what lets us later partition
                              # data across Federated Learning clients —
                              # each bank_id becomes one FL client's shard.
    is_fraud: bool = False    # ground-truth label (only exists because this
                               # is synthetic data — a real feed usually
                               # won't have this until an analyst confirms it)

    @staticmethod
    def new(**kwargs) -> "Transaction":
        """Convenience constructor that fills in id/timestamp automatically."""
        kwargs.setdefault("transaction_id", str(uuid.uuid4()))
        kwargs.setdefault(
            "timestamp", datetime.now(timezone.utc).isoformat()
        )
        return Transaction(**kwargs)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(raw: str) -> "Transaction":
        return Transaction(**json.loads(raw))
