# SMC Multi-System Prediction Pipeline

A modelling layer that sits on top of the `MarketStructureParser` and
`mtf_state_builder` which where created on a separate project. It turns the per-timeframe structure output into supervised
labels and trains the three prediction systems described below, plus an M5
forward-window labels — with **empirical baselines** and **leakage-safe ML models**
layered on top of each other.

Instrument target: **Volatility 75 (1s) Index**. Language: **Python** (numpy/pandas core;
LightGBM/XGBoost/scikit-learn optional).

---

## The four systems

| System | Question | Timeframes | Target | Type |
|---|---|---|---|---|
| **1 — Pullback depth** | How far does a pullback retrace toward the strong low/high? | H4, M15 | `pullback_depth` ∈ [0, ~1.5] | regression |
| **2a — Continuation occurrence** | Will the pullback continue or reverse? | H4, M15 | `continuation_occurred` {0,1} | classification |
| **2b — Continuation timing** | How many bars until continuation? | H4, M15 | `bars_to_continuation` | regression |
| **3 — Continuation extent** | How far does the continuation run before reversing/going sideways? | H4, M15 | `cont_extent_atr` (run ÷ ATR) | regression |
| **4a — M5 forward move** | How far does price travel over the next 75 bars? | M5 | `fwd75_net_atr` (and `mfe`/`mae`) | regression |
| **4b — M5 forward direction** | Up or down over the next 75 bars? | M5 | `fwd75_dir` {0,1} | classification |

### Label definitions (precise)

**Pullback depth** (bullish leg; bearish is mirrored):
```
depth = (weak_high − pullback_low) / (weak_high − strong_low)
```
`0` = no retrace, `1` = retraced all the way to the strong low, `>1` = swept beyond it
(usually a reversal). `pullback_low` is the lowest low *after* the weak-high swing top,
up to the resolution bar. The resolution is a continuation **BOS** (depth measured, label
`continuation_occurred=1`) or a reversal **CHOCH** (`=0`).

**Continuation extent** (from the continuation BOS until the next opposite CHOCH):
```
run_distance   = run_extreme − breakout_level        # bullish
cont_extent_atr = run_distance / ATR(at BOS)         # primary target
cont_extent_leg = run_distance / prior_leg_size      # alt normalisation
cont_outcome    = 'extended' if cont_extent_atr ≥ 1 else 'sideways'
```

**M5 forward window** (per bar, next `H`=75 bars):
```
fwd75_mfe = max(high[t+1..t+H]) − close[t]    # max up
fwd75_mae = close[t] − min(low[t+1..t+H])     # max down
fwd75_net = close[t+H] − close[t]
*_atr = above ÷ ATR(t)
```
A non-overlapping **every-75-bar average** summary is also produced
(`labels_m5_block_summary.csv`) for the "...or on average" view.

> A clean structural signal already shows up in the labels: shallow pullbacks
> (depth ≈ 0.03–0.15) tend to *continue*, while deep ones that sweep the strong anchor
> (depth > 1.0) tend to *reverse*. That is the relationship Systems 1–2 learn.

---

## Architecture

```
 OHLC (M5/M15/H4)
      │  MarketStructureParser  (smc_parser.py — your engine)
      ▼
 states.csv / events.csv / pois.csv   per timeframe
      │  mtf_state_builder.py  (your merge)            smc_labeling.py
      ▼                                                     │
 fused M5 master table  ───────────── features ────────────┤ (episodes detected at
 (one row per M5 bar, H4+M15 context)                       │  native TF; features
      │                                                     │  joined at decision_time)
      │                                            labels ──┘
      ▼
 smc_baselines.py   →  conditional mean/median/quantile predictor  (Layer 1)
 smc_models.py      →  LightGBM/XGBoost/sklearn/numpy GBM + purged CV (Layer 2)
 smc_train.py       →  orchestrates everything, writes artifacts + report
```

### Files

| File | Role |
|---|---|
| `smc_parser.py` | Your `MarketStructureParser` (unchanged). |
| `mtf_state_builder.py` | Your multi-timeframe merge (unchanged). |
| `run_parser.py` | Runs the parser over an OHLC frame and exports `states/events/pois.csv` (with OHLC joined into states). |
| `smc_labeling.py` | Episode detection + the four label families. |
| `smc_baselines.py` | Layer-1 empirical conditional baselines. |
| `smc_gbm.py` | Zero-dependency histogram gradient booster (fallback only). |
| `smc_models.py` | Backend selection, feature selection, purged walk-forward CV, metrics. |
| `smc_train.py` | CLI orchestrator. |
| `make_synth_and_run.py` | End-to-end demo on synthetic V75-like data. |

---

## Leakage safety (read this)

This is the part that makes or breaks SMC ML research. Three controls:

1. **Decision-time features.** Episode features are joined with
   `merge_asof(direction="backward")` at the episode's `decision_time`, so a row only
   sees information available up to that bar.
2. **Stationary features only.** `select_feature_columns` drops raw price *levels*
   (`open/high/low/close`, `strong_high`, `weak_low`, POI bounds, …) and all monotonic
   index columns (`*_idx`, `bar_index`) — these leak the passage of time. It keeps
   returns, distances, depths, ATR-normalised values, time-since-event counters and flags.
3. **Purged, embargoed, group-aware walk-forward CV.** Test folds are always in the
   future of their train slice; an `episode_id` never spans train and test; training rows
   inside the test window (+ embargo) are purged.

**Sanity check built in:** on pure random-walk data the models should *not* beat the
empirical baselines. They don't (negative R², AUC ≈ 0.5–0.55 in the synthetic demo). If
you ever see strong metrics on shuffled/synthetic data, suspect leakage first.

---

## Usage

### 1. You already have parser output + a fused table
```bash
python smc_train.py \
  --mtf-csv  out/mtf_state.csv \
  --h4-dir   parser_out/h4 \
  --m15-dir  parser_out/m15 \
  --m5-dir   parser_out/m5 \
  --out-dir  models_out \
  --horizon  75
```

### 2. You have parser output but want the fused table built for you
```bash
python smc_train.py --build-mtf --ohlc-csv m5.csv \
  --h4-dir parser_out/h4 --m15-dir parser_out/m15 --m5-dir parser_out/m5 \
  --out-dir models_out
```

### 3. End-to-end demo (synthetic data, no inputs needed)
```bash
python make_synth_and_run.py
```

### Outputs (`models_out/`)
- `labels_*.csv` — every label table (inspect these first).
- `report.txt`, `metrics.json` — model vs baseline per system.
- `importance_*.csv`, `oof_*.csv` — feature importances and out-of-fold predictions.
- `model_*.pkl` — `{model, features, backend}` for serving.
- `baselines/*.csv` — the interpretable conditional lookup tables.

---

## Dependencies

Core: `numpy`, `pandas` (required).
ML backend (auto-selected, best first): **LightGBM → XGBoost → scikit-learn → built-in numpy GBM**.

```bash
pip install lightgbm scikit-learn   # recommended; LightGBM handles NaNs + is fastest
```

If none are installed the pipeline still runs on the bundled `smc_gbm.py` fallback (slower,
less accurate — fine for wiring/tests, install LightGBM for real work).

---

## Tuning & next steps

- **Data volume matters.** H4 episodes are rare; you need a long history (months–years of
  V75) for the H4 systems to have enough episodes. M15 and M5 fill in faster.
- **Episode definition knobs** live in `build_episodes_for_tf` (ATR window, sideways
  threshold). The pullback start currently anchors on the swing extreme inside the leg;
  adjust `decision_time` there if you want to predict *earlier* in the pullback.
- **Class imbalance / costs.** For Systems 2a/4b consider class weights and a probability
  threshold tuned to your R:R rather than 0.5.
- **Walk-forward retraining.** For live use, retrain on a rolling window and serve the
  pickled model; never fit on data after the bar you are predicting.

> Research tooling only — not trading advice, and nothing here executes orders. Validate
> on your own out-of-sample data before risking capital.
