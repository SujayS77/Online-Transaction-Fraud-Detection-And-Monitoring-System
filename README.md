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

## Phase 2 — Baseline model (no FL, no GAN yet)

```bash
python training/generate_dataset.py   # builds data/transactions.csv (~20k rows, ~2% fraud)
python training/train_baseline.py     # trains + evaluates + saves models/baseline_xgb.pkl
```

- **`common/features.py`** — turns raw transaction fields into per-user relative features
  (e.g. `amount_ratio_to_user_avg`, `device_freq_for_user`) instead of raw high-cardinality IDs.
  Used by both training and (later) the real-time inference service, so features are computed
  identically in both places.
- **`training/generate_dataset.py`** — batch version of the same fraud logic used by
  `generator/simulate.py`, but produces a static CSV instead of a live stream (training needs a
  finished dataset, not an infinite one).
- **`training/train_baseline.py`** — trains an XGBoost classifier with `scale_pos_weight` to
  handle the ~2% fraud imbalance, evaluated on precision/recall/F1/PR-AUC (not accuracy — with
  this little fraud, "always predict normal" would already score ~98% accuracy).

### Baseline results (keep these — you'll compare Phase 4's GAN-augmented version against them)

| Metric | Value |
|---|---|
| Precision (fraud) | 0.68 |
| Recall (fraud) | 0.93 |
| F1 (fraud) | 0.78 |
| ROC-AUC | 0.999 |
| PR-AUC | 0.947 |

Note: the generator's fraud patterns were deliberately given some overlap with normal behavior
(occasional legit new-device use, occasional large legit purchases) — an earlier version without
that overlap scored a *suspicious* 100% across the board, which would read as data leakage to an
interviewer, not skill. Realistic-but-imperfect numbers are the more defensible result.

## Next: Phase 3

Train a GAN (CTGAN) on the fraud-labeled rows only, generate synthetic fraud examples, mix them
into training, and compare against the baseline table above. Ask Claude to help set this up when
you're ready.