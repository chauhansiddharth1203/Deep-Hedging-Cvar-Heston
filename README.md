# Risk-Sensitive Deep Hedging under Stochastic Volatility with Variance Instruments

**Siddharth Chauhan (CH21B103)**  
Dual Degree Project — Department of Data Science and Artificial Intelligence  
Indian Institute of Technology Madras

---

## Overview

This repository contains the full implementation for a DDP thesis that trains deep hedging policies under the **Conditional Value-at-Risk (CVaR)** objective using the **Heston stochastic volatility model**, augmented with variance swap and VIX-futures instruments. The central finding is that a **regime-gated VIX policy** with a frozen sigmoid threshold at θ = 3.0 achieves a robust +1.11 ± 0.10 CVaR improvement on the 2024 SPY out-of-training year while closing the 2020 COVID crisis gap to −1.18 ± 0.41 — both satisfying pre-registered decision rules across 5 independent seeds.

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Environment Setup](#environment-setup)
3. [Experiment Replication Guide](#experiment-replication-guide)
   - [Week 1 — Baseline: CVaR Hedging under Heston](#week-1--baseline-cvar-hedging-under-heston)
   - [Week 2 — Jump Robustness: Bates Model](#week-2--jump-robustness-bates-model)
   - [Week 3 — Bootstrap Significance Testing](#week-3--bootstrap-significance-testing)
   - [Week 4 — SPX-Calibrated Parameters](#week-4--spx-calibrated-parameters)
   - [Week 5 — Benchmarking and Interpretability](#week-5--benchmarking-and-interpretability)
   - [Week 6 — Multi-Asset Spread Option Hedging](#week-6--multi-asset-spread-option-hedging)
   - [Week 7 — Ablations, Greeks, and Historical Stress Test](#week-7--ablations-greeks-and-historical-stress-test)
   - [Week 8 — Sim-to-Real Transfer](#week-8--sim-to-real-transfer)
   - [Week 9 — Backtest-Based Training](#week-9--backtest-based-training)
   - [Week 10 — Expanded Real-Market Corpus](#week-10--expanded-real-market-corpus)
   - [Week 11 — VIX Futures as a Vega Instrument](#week-11--vix-futures-as-a-vega-instrument)
   - [Week 12 — Multi-Seed Verification](#week-12--multi-seed-verification)
   - [Week 13 — Regime-Gated Architecture (Main Result)](#week-13--regime-gated-architecture-main-result)
4. [Key Results Summary](#key-results-summary)
5. [Recommended Replication Order](#recommended-replication-order)
6. [Data Dependencies](#data-dependencies)

---

## Repository Structure

```
Deep-Hedging-Cvar-Heston/
│
├── market/                  # Market simulators
│   ├── heston.py            # Heston stochastic volatility simulator
│   ├── heston_with_var_swap.py  # Heston + variance swap payoff
│   ├── heston_calibrated.py # SPX-calibrated Heston parameters
│   ├── heston_stress.py     # Stress-test parameterisation
│   ├── heston_random.py     # Domain-randomised Heston
│   ├── bates.py             # Bates model (Heston + Poisson jumps)
│   ├── bates_with_var_swap.py
│   ├── multi_heston.py      # Correlated two-asset Heston
│   ├── regime_switching_heston_varswap.py
│   ├── neural_sde.py        # Neural SDE generator
│   ├── vix_bootstrap.py     # VIX path block-bootstrap
│   ├── historical_bootstrap.py
│   └── sabr.py              # SABR misspecification test
│
├── policy/                  # Hedging policy networks
│   ├── network.py           # Base policy (stock only, 3-dim state)
│   ├── network_varswap.py   # Stock + variance swap (4-dim state)
│   ├── network_vix.py       # Stock + vega + VIX (5-dim state)
│   ├── network_vix_gated.py # Regime-gated VIX policy (final architecture)
│   ├── network_lstm.py      # LSTM online gate
│   └── network_vix_lstm_gate.py
│
├── training/                # Training loops
│   ├── trainer_cvar.py      # CVaR trainer (base)
│   ├── trainer_variance.py  # Variance objective
│   ├── trainer_entropic.py  # Entropic risk measure
│   ├── trainer_cvar_varswap.py
│   ├── trainer_gated.py     # Regime-gated CVaR trainer
│   ├── trainer_mom.py       # Median-of-Means optimiser (negative result)
│   ├── trainer_nsde_gan.py  # Neural SDE GAN training
│   └── trainer_cvar_*.py    # Specialised trainers
│
├── evaluation/              # Evaluation pipelines
│   ├── evaluate.py          # Base evaluator
│   ├── evaluate_varswap.py
│   ├── evaluate_stress.py
│   └── evaluate_calibrated.py
│
├── baselines/               # Analytical benchmarks
│   ├── delta_hedge.py       # Black-Scholes delta hedge
│   └── heston_delta_vega_hedge.py  # Heston delta-vega analytical hedge
│
├── data/                    # Real-data loaders
│   ├── load_market_data.py  # yfinance SPY/VIX downloader
│   ├── spx_windows.py       # SPX rolling-window extractor
│   └── vix_windows.py       # SPY + VIX joint window loader
│
├── calibration/             # SPX option calibration
│   ├── calibrate_spx.py
│   └── heston_pricer.py
│
├── analysis/
│   └── bootstrap_cvar.py    # Bootstrap CI utilities
│
├── utils/
│   └── bootstrap_ci.py
│
├── experiments/
│   └── tc_sensitivity.py    # Transaction-cost sweep
│
├── risk/
│   ├── cvar.py              # CVaR loss function
│   └── entropic.py          # Entropic risk measure
│
├── instruments/
│   └── options.py           # Option payoff definitions
│
├── main.py                          # Week 1: Base CVaR (stock only)
├── main_variance.py                 # Week 1: Variance objective
├── main_entropic.py                 # Week 1: Entropic objective
├── main_varswap_cvar.py             # Week 1: CVaR + variance swap
├── main_varswap_variance.py         # Week 1: Variance + variance swap
├── main_varswap_entropic.py         # Week 1: Entropic + variance swap
├── main_bates.py                    # Week 2: Bates jump model
├── main_bootstrap.py                # Week 3: Bootstrap CIs
├── main_calibrated.py               # Week 4: SPX-calibrated Heston
├── main_spx_calibration.py          # Week 4: Calibration fitting
├── main_heston_benchmark.py         # Week 5: Delta-vega benchmark
├── main_otm_sweep.py                # Week 5: OTM strike sweep
├── main_hedge_analysis.py           # Week 5: Hedge ratio analysis
├── main_learning_curve.py           # Week 5: Learning curves
├── main_multi_asset.py              # Week 6: Multi-asset spread option
├── main_ablation_mean_cvar.py       # Week 7: Mean-CVaR trade-off
├── main_ablation_design.py          # Week 7: Architecture ablation
├── main_greeks_decomposition.py     # Week 7: Greeks decomposition
├── main_historical_stress.py        # Week 7: Historical stress test
├── main_path_dependent.py           # Week 7: Path-dependent options
├── main_misspecification.py         # Week 7: Model misspecification
├── main_robust_training.py          # Week 7: Domain randomization
├── main_sim_to_real.py              # Week 8: Sim-to-real transfer
├── main_sim_to_real_earlystop.py    # Week 8: Early stopping variant
├── main_backtest_training.py        # Week 9: Backtest-based training
├── main_expanded_corpus.py          # Week 10: Expanded real corpus
├── main_vix_futures.py              # Week 11: Naive VIX adaptation
├── main_vix_futures_v2.py           # Week 11: Leverage-corrected VIX
├── main_vix_multiseed.py            # Week 12: Plain-Adam multi-seed
├── main_vix_gated_multiseed.py      # Week 13: Regime-gated multi-seed
├── main_vix_mom_multiseed.py        # Week 13: MoM multi-seed (negative)
├── main_nsde_hedge.py               # Post-W13: Neural SDE hedge
├── main_nsde_train.py               # Post-W13: Neural SDE training
├── main_regime_switching.py         # Regime-switching Heston
├── main_risk_comparison.py          # Clean risk-measure comparison
├── main_significance_test.py        # Statistical significance
├── compare.py                       # Cross-model comparison
├── compare_varswap.py
├── plot_cvar_confidence_intervals.py
├── plot_objective_comparison.py
└── requirements.txt
```

---

## Environment Setup

**Python version:** 3.9+ recommended (tested on 3.9 and 3.13)

```bash
# Clone the repository
git clone https://github.com/chauhansiddharth1203/Deep-Hedging-Cvar-Heston.git
cd Deep-Hedging-Cvar-Heston

# Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**GPU:** All scripts auto-detect CUDA. Training on CPU is supported but ~5× slower for multi-seed runs. A single A100/V100 epoch of 400 epochs takes ~8 minutes per seed.

**Output directory:** All scripts write results to `results/`. This directory is created automatically and is excluded from version control.

---

## Experiment Replication Guide

All Δ values reported are **CVaR improvement over the Black-Scholes delta-hedge baseline**: positive means the deep hedger achieves a better (less negative) CVaR tail.

---

### Week 1 — Baseline: CVaR Hedging under Heston

**Goal:** Establish that (a) a CVaR-trained deep hedger beats BS delta-hedge on simulated Heston paths, and (b) adding a variance swap as a second instrument further improves CVaR.

**Default hyperparameters:** N=10 000 paths, T=30 steps, lr=3e-4, 200 epochs, α=0.05 (CVaR level).

```bash
# Stock-only, CVaR objective
python main.py

# Stock-only, variance objective
python main_variance.py

# Stock-only, entropic risk measure
python main_entropic.py

# Stock + variance swap, CVaR
python main_varswap_cvar.py

# Stock + variance swap, variance objective
python main_varswap_variance.py

# Stock + variance swap, entropic
python main_varswap_entropic.py
```

**Expected outputs:** `results/deep_hedge_*.pth` (trained weights), `results/pnl_*.pt` (PnL tensors), console table of CVaR values vs delta-hedge.

**Key result:** CVaR + variance swap improves over stock-only CVaR by ~2–3 CVaR units on Heston test paths.

---

### Week 2 — Jump Robustness: Bates Model

**Goal:** Check whether the variance-swap advantage survives when stock prices can jump (Bates = Heston + compound Poisson jumps).

```bash
python main_bates.py
python plot_bates_comparison.py  # produces results/bates_comparison.png
```

**Expected outputs:** `results/bates_cvar.pth`, `results/bates_varswap_cvar.pth`, `results/bates_comparison.png`, `results/bates_vs_heston.png`.

**Key result:** Variance swap advantage persists under jumps; delta hedge degrades more than the deep hedger.

---

### Week 3 — Bootstrap Significance Testing

**Goal:** Confirm all Week 1 CVaR improvements are statistically significant (95% bootstrap CIs do not include zero improvement).

**Prerequisite:** Run all six Week 1 scripts first so the `.pth` weight files exist.

```bash
python main_bootstrap.py
python plot_cvar_confidence_intervals.py
```

**Expected outputs:** Console table of CVaR ± 95% CI for each model, `results/bootstrap_ci_plot.png`.

---

### Week 4 — SPX-Calibrated Parameters

**Goal:** Replace the default synthetic Heston parameters with parameters calibrated to real SPX implied volatility surface, and verify the CVaR advantage holds under realistic calibration.

```bash
# Fit Heston parameters to SPX options (requires internet for yfinance)
python main_spx_calibration.py

# Evaluate the calibrated policy
python main_calibrated.py
```

**Expected outputs:** Calibrated parameter set printed to console, `results/calibrated_policy.pth`.

---

### Week 5 — Benchmarking and Interpretability

**Goal:** Compare the deep hedger against the analytical Heston delta-vega hedge, run an OTM strike sweep, and inspect learned hedge ratios.

```bash
# Heston delta-vega analytical benchmark
python main_heston_benchmark.py

# OTM strike sweep (K/S0 from 0.85 to 1.15)
python main_otm_sweep.py

# Hedge ratio surfaces (delta_S, delta_V vs (S,v))
python main_hedge_analysis.py

# Learning curves (CVaR vs epoch)
python main_learning_curve.py
```

**Expected outputs:** `results/otm_sweep_results.csv`, `results/hedge_delta_S_surface.png`, `results/hedge_delta_V_surface.png`, `results/hedge_vs_bs.png`.

**Key result:** Deep hedger hedge ratios are interpretable — delta_S tracks BS delta; delta_V is positive and correlated with current variance, consistent with vega hedging.

---

### Week 6 — Multi-Asset Spread Option Hedging

**Goal:** Extend the framework to a two-asset spread call under correlated Heston dynamics, with four tradable instruments (S¹, S², VS¹, VS²). Baseline is the Margrabe exchange-option analytical delta.

```bash
python main_multi_asset.py
```

**Expected outputs:** `results/multi_asset_hedge_composition.png`, console table with CVaR improvement across correlation values ρ ∈ {0.0, 0.3, 0.5, 0.7, 0.9}.

**Key result:** Deep hedger beats Margrabe delta by ~30% CVaR at ρ = 0.5; advantage grows at higher correlations.

---

### Week 7 — Ablations, Greeks, and Historical Stress Test

Multiple independent experiments; run in any order.

```bash
# Ablation A: Mean-CVaR trade-off (lambda sweep)
python main_ablation_mean_cvar.py

# Ablation B: Architecture design choices
python main_ablation_design.py

# Greeks decomposition (delta, vega, vanna contributions)
python main_greeks_decomposition.py

# Path-dependent options (Asian, barrier)
python main_path_dependent.py

# Model misspecification (train Heston, test SABR)
python main_misspecification.py

# Domain randomization for robust training
python main_robust_training.py

# Historical stress test on real SPX (2008 GFC, 2020 COVID, 2017 calm)
python main_historical_stress.py

# Statistical significance test
python main_significance_test.py
```

**Key result for stress test:** Policy trained on Heston simulation shows degradation on real crisis data — the sim-to-real gap that motivates Weeks 8–13.

---

### Week 8 — Sim-to-Real Transfer

**Goal:** Train on simulated Heston paths, evaluate on rolling 30-day SPX windows. Honest negative result: mean P&L is positive (implying a speculative bias) and the tail CVaR on real data is worse than on simulation.

```bash
# Standard sim-to-real
python main_sim_to_real.py

# With early stopping on simulated validation set
python main_sim_to_real_earlystop.py
```

**Expected outputs:** `results/sim_to_real_*.png`, console tables comparing simulated vs real-window CVaR.

**Key result:** Persistent sim-to-real gap of ~5–8 CVaR units on crisis windows; zero gap on calm windows. Two failure modes identified: EWMA proxy mismatch and regime coverage gap.

---

### Week 9 — Backtest-Based Training

**Goal:** Replace simulated training with rolling windows from real SPX data (a partial positive result). Shows regime coverage alone closes ~23% of the COVID gap.

```bash
python main_backtest_training.py
```

**Expected outputs:** `results/backtest_training_*.png`, console table with COVID window improvement.

---

### Week 10 — Expanded Real-Market Corpus

**Goal:** Add 2008 GFC, 2018 Volmageddon, 2022 rate-shock windows to training data to test the regime-coverage hypothesis.

```bash
python main_expanded_corpus.py
```

**Expected outputs:** Per-window CVaR table showing failure of the regime-coverage hypothesis: adding crisis data does not fix crisis performance, and degrades calm performance.

---

### Week 11 — VIX Futures as a Vega Instrument

**Goal:** Replace the synthetic variance swap with VIX futures (a real, tradable vega instrument). Two versions: naive and leverage-corrected.

```bash
# Naive VIX adaptation (negative result)
python main_vix_futures.py

# Leverage-corrected VIX policy (the working version)
python main_vix_futures_v2.py
```

**Expected outputs:** Console tables for calm and crisis windows, VIX gate activation plots.

**Key result:** Leverage-corrected VIX policy achieves +1.16 ± 0.06 on 2024 OOT (calm), but −8.05 ± 4.96 on 2020 COVID with high seed variance — the instability that Week 12 characterises.

---

### Week 12 — Multi-Seed Verification

**Goal:** Run the leverage-corrected VIX policy at 5 seeds to characterise the domain of validity precisely. Pre-register decision rules before any Week-13 code runs.

```bash
# 5 seeds, calm mode (2017 + 2024 OOT)
python main_vix_multiseed.py --mode calm --seeds 5 --epochs 400

# 5 seeds, crisis mode (2020 COVID)
python main_vix_multiseed.py --mode covid --seeds 5 --epochs 400
```

**Pre-registered decision rules (from Week 12, unchanged in Week 13):**
- 2024 OOT win is real iff mean Δ > 0 AND seed-std < |mean Δ|
- 2020 COVID closure is real iff mean Δ > −2.5 AND seed-std < 2.0

**Key result:** 2024 OOT: +1.16 ± 0.06 (rule MET). 2020 COVID: −8.05 ± 4.96 (rule FAILS — seed-std far too large). This characterises the problem precisely for Week 13.

---

### Week 13 — Regime-Gated Architecture (Main Result)

Two candidate fixes tested in order; Option B first (falsified), Option A second (confirmed).

#### Option B — Median-of-Means Optimiser (Negative Result)

```bash
# 5 seeds, calm mode
python main_vix_mom_multiseed.py --mode calm --seeds 5 --epochs 400

# 5 seeds, COVID mode
python main_vix_mom_multiseed.py --mode covid --seeds 5 --epochs 400
```

**Expected result:** COVID worsens to −17.88 ± 5.50 (doubled from baseline). MoM is structurally wrong for CVaR: tail gradients are signal, not noise.

#### Option A — Regime-Gated Architecture (Confirmed)

The gate is a frozen sigmoid: `gate(v) = σ((v_norm − θ) / w)` where v_norm = current VIX / window-start VIX. The VIX action is multiplied by `(1 − gate)` — hedging turns off when VIX has spiked beyond threshold.

**Threshold sweep (run first, all other hyperparameters frozen):**

```bash
# θ = 2.0
python main_vix_gated_multiseed.py --mode calm --threshold 2.0 --width 0.3 --seeds 5 --epochs 400

# θ = 2.5
python main_vix_gated_multiseed.py --mode calm --threshold 2.5 --width 0.3 --seeds 5 --epochs 400

# θ = 3.0  ← sweet spot
python main_vix_gated_multiseed.py --mode calm --threshold 3.0 --width 0.3 --seeds 5 --epochs 400

# θ = 3.5
python main_vix_gated_multiseed.py --mode calm --threshold 3.5 --width 0.3 --seeds 5 --epochs 400
```

**Width sensitivity sweep (confirm θ = 3.0 is not width-specific):**

```bash
python main_vix_gated_multiseed.py --mode calm --threshold 3.0 --width 0.15 --seeds 5 --epochs 400
python main_vix_gated_multiseed.py --mode calm --threshold 3.0 --width 0.60 --seeds 5 --epochs 400
python main_vix_gated_multiseed.py --mode calm --threshold 2.5 --width 0.15 --seeds 5 --epochs 400
python main_vix_gated_multiseed.py --mode calm --threshold 2.5 --width 0.60 --seeds 5 --epochs 400
```

**Expected results at θ = 3.0, width = 0.3:**

| Window           | Plain Adam        | Regime-gated θ=3.0 |
|------------------|------------------:|-------------------:|
| 2024 OOT (calm)  | +1.16 ± 0.06      | **+1.11 ± 0.10**   |
| 2020 COVID       | −8.05 ± 4.96      | **−1.18 ± 0.41**   |
| 2008 GFC         | ~−25              | −2.35 ± 0.47       |
| Max seed-std     | 4.96              | 0.47 (↓ 10×)       |

Both pre-registered decision rules are met with strong margins.

**Width-sensitivity result:** At θ = 3.0 the 2024 win and COVID closure hold across widths 0.15, 0.30, 0.60 (4× range). At θ = 2.5 both widths fail. The threshold is the load-bearing parameter; the width is not.

---

## Key Results Summary

| Experiment | Script | Headline Δ CVaR |
|---|---|---|
| Base CVaR vs delta-hedge | `main.py` | ~+1.5 (Heston sim) |
| CVaR + variance swap | `main_varswap_cvar.py` | ~+3.5 (Heston sim) |
| Bates robustness | `main_bates.py` | advantage preserved |
| Multi-asset spread | `main_multi_asset.py` | +30% over Margrabe |
| Sim-to-real transfer | `main_sim_to_real.py` | gap ~5–8 (crisis) |
| VIX leverage-corrected | `main_vix_futures_v2.py` | +1.16 calm, −8.05 COVID |
| MoM optimiser | `main_vix_mom_multiseed.py` | **falsified** |
| **Regime-gated θ=3.0** | `main_vix_gated_multiseed.py` | **+1.11 calm, −1.18 COVID** |

---

## Recommended Replication Order

For a reader who wants to reproduce only the thesis-central results:

1. `main_varswap_cvar.py` — Week 1 headline (variance swap adds value)
2. `main_bootstrap.py` — Week 3 (statistical significance)
3. `main_historical_stress.py` — Week 7 (sim-to-real gap exists)
4. `main_vix_futures_v2.py` — Week 11 (VIX policy, calm win established)
5. `main_vix_multiseed.py --mode calm` then `--mode covid` — Week 12 (problem characterised)
6. `main_vix_mom_multiseed.py --mode covid` — Week 13 Option B (negative result)
7. `main_vix_gated_multiseed.py --threshold 3.0 --width 0.3 --mode calm` — Week 13 Option A (main result)

Total wall-clock on a single GPU: approximately 6–8 hours for steps 5–7.

---

## Data Dependencies

All experiments use either:

- **Simulated data** — generated on-the-fly from Heston/Bates/SABR; no download needed.
- **Real SPX/VIX data** — downloaded automatically via `yfinance` on first run. An internet connection is required. Data is not cached to disk by default.

The following windows are used for out-of-training evaluation:

| Window label     | Date range              | Type        |
|------------------|-------------------------|-------------|
| 2008 GFC         | 2007-09-01 – 2009-06-30 | Crisis      |
| 2017 Calm        | 2017-01-01 – 2017-12-31 | Calm        |
| 2018 Volmageddon | 2018-01-01 – 2018-12-31 | Vol spike   |
| 2020 COVID       | 2020-01-01 – 2020-06-30 | Crisis      |
| 2022 Rate shock  | 2022-01-01 – 2022-12-31 | Stress      |
| 2023 SVB         | 2023-01-01 – 2023-12-31 | Mild stress |
| 2024 Full year   | 2024-01-01 – 2024-12-31 | OOT calm    |

The 2024 window is the primary out-of-training test; all other windows are used for stress evaluation only.
