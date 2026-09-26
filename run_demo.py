"""
Proves the whole Phase 1 loop works: generator -> Redis Stream -> adapter.

WHY this script matters even though it "does nothing" with the data yet:
it's the first proof that the adapter pattern (Section 3.2 of the design
doc) actually works — this script only ever talks to `TransactionSource`,
never to Redis directly. When Phase 5 swaps in a real company's database,
THIS loop (and everything built on top of it) will not need to change.

Run generator/simulate.py in one terminal, then this in another.
"""

from common.source import SyntheticSource

if __name__ == "__main__":
    source = SyntheticSource()
    print("Listening for transactions via SyntheticSource... Ctrl+C to stop.\n")

    for txn in source.stream():
        tag = "🚨 FRAUD" if txn.is_fraud else "  normal"
        print(f"{tag} | {txn.transaction_id[:8]} | user={txn.user_id} "
              f"| amount={txn.amount:>10.2f} | loc={txn.location} "
              f"| device={txn.device_id}")
