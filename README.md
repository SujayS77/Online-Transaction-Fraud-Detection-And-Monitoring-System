# FraudGuard — Phase 1: Data Layer

This is the foundation of the full project (see the design doc). It has three parts:

1. **`common/models.py`** — the `Transaction` schema everything else uses.
2. **`common/source.py`** — the `TransactionSource` adapter interface. `SyntheticSource` is the
   implementation we use now; `SQLSource`/`APISource` are stubs to fill in when you point this at
   a real company's data later.
3. **`generator/simulate.py`** — generates fake transactions (98% normal, ~2% fraud patterns) and
   pushes them onto a Redis Stream.

## Setup

```bash
cd fraudguard
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
docker-compose up -d      # starts Redis on localhost:6379
```

## Run it (two terminals)

**Terminal 1 — start generating fake transactions:**
```bash
python generator/simulate.py
```

**Terminal 2 — consume them through the adapter:**
```bash
python run_demo.py
```

You should see the same transactions printed in both terminals — one writing to Redis, one
reading through the `TransactionSource` abstraction. That round trip is the whole point of
Phase 1: nothing downstream will ever talk to Redis directly again.

## What "done" looks like for Phase 1

- [x] Transactions flow generator → Redis → adapter → consumer without errors
- [ ] You can explain, out loud, why `TransactionSource` exists and what swapping `SQLSource` in
      would actually require (should be: implement `stream()`, change zero other files)
- [ ] You've watched it run long enough to see a few `[FRAUD]` tagged transactions and can
      describe the two injected fraud patterns (spike, new-device+new-location)

## Next: Phase 2

Pull ~5,000 transactions off the stream, save to a CSV/DataFrame, and train a baseline XGBoost
classifier — no Federated Learning, no GAN yet. Get real precision/recall numbers on a simple
model before adding complexity. Ask Claude to help set this up when you're ready.
