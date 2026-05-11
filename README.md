# Machine Learning-Based Predictive Handover Management

**CSE476 Mobile Communication Networks — Course Project**
Utku Gökçek and Mert Uğur

A simulation study that compares a traditional RSSI-threshold (LTE A3-event)
handover controller with a Random Forest predictor that proactively switches
serving cells before the current link degrades. The project follows the
course proposal: a synthetic 3×3 cellular grid, Random Waypoint mobility,
3GPP COST-231 path-loss with log-normal shadowing, and side-by-side metrics.

---

## Headline Result (20 users, 14,000 steps, seed = 42)

| Metric                  | Threshold | **ML-Based** | Improvement |
| ----------------------- | --------: | -----------: | ----------: |
| Success rate (%)        |     83.52 |    **89.46** | **+5.94 pp** |
| Ping-pong count         |        20 |        **0** |    **-100%** |
| Unnecessary HOs         |        17 |       **11** |     **-35%** |
| Total handovers         |       205 |      **170** |     **-17%** |
| HO rate (per 100 steps) |      1.46 |     **1.21** |     **-17%** |
| Total interruption (ms) |    10,250 |    **8,500** |     **-17%** |
| Avg serving RSSI (dBm)  |    -55.50 |   **-55.44** |    +0.06 dB |

ML wins on every metric the proposal calls out.

---

## Setup

```bash
# Python 3.10+ recommended
pip install -r requirements.txt
```

`requirements.txt` pins only what the pipeline imports:
`numpy`, `pandas`, `scikit-learn`, `matplotlib`, `joblib`.

## Run

```bash
# Default: 20 users, cached data + model, ~10 s to run
python3 main.py

# Full 60-user evaluation (includes the 12-user held-out test split)
python3 main.py --users 60

# Force regenerate everything from scratch
python3 main.py --regenerate --retrain

# Change which user is rendered in the timeline plots
python3 main.py --plot-user 3
```

The pipeline runs five steps in order:

1. **Mobility & RSSI simulation** — build (or load) the synthetic dataset.
2. **Random Forest training** — user-level train/val/test split, fit RF.
3. **Handover simulation** — run Threshold and ML controllers on the same traces.
4. **Performance evaluation** — compute success rate, ping-pong, unnecessary HOs, interruption, total HOs.
5. **Plot generation** — seven PNG plots saved to `results/`.

---

## Repository Layout

```
.
├── config.py                       # All hyperparameters in one place
├── main.py                         # 5-step pipeline orchestrator
├── requirements.txt                # numpy, pandas, sklearn, matplotlib, joblib
├── data/
│   └── mock_data_generator.py      # Synthetic trace generator + CSV cache
├── simulation/
│   ├── cell_grid.py                # 3×3 base-station layout
│   ├── mobility.py                 # Random Waypoint model
│   └── rssi.py                     # 3GPP COST-231 path-loss + shadowing
├── ml/
│   ├── features.py                 # 13-feature vector (proposal-aligned)
│   ├── trainer.py                  # User-level split + RF fit
│   └── predictor.py                # Inference wrapper (batch + table)
├── handover/
│   ├── threshold_handover.py       # A3-event baseline (hysteresis + TTT)
│   └── ml_handover.py              # ML-based proactive controller
├── evaluation/
│   └── metrics.py                  # Success / ping-pong / unnecessary / interruption
├── visualization/
│   └── plots.py                    # 7 PNG plots, six-panel metric chart
└── results/                        # Generated plots
    ├── 01_cell_grid.png
    ├── 02_rssi_over_time.png
    ├── 03_handover_timeline.png
    ├── 04_metric_comparison.png
    ├── 05_rssi_heatmap.png
    ├── 06_confusion_matrix.png
    └── 07_feature_importance.png
```

---

## Methodology Highlights

### Feature set (proposal-aligned)

Exactly the four feature categories listed in §2 of the proposal:

| Feature              | Type           | Count | Notes                                   |
| -------------------- | -------------- | ----- | --------------------------------------- |
| `current_cell`       | int (0–8)      | 1     | Controller's currently served cell      |
| `speed`              | float (m/s)    | 1     | User speed                              |
| `direction_sin/cos`  | float in [-1,1]| 2     | One "direction" feature, sin/cos encoded to avoid 0/2π wraparound |
| `rssi_0 … rssi_8`    | float (dBm)    | 9     | Noisy RSSI from each base station       |
| **Total**            |                | **13**|                                         |

The label is `next_cell` = noiseless optimal cell at `t + LOOKAHEAD` (5 steps).

### Why `current_cell` uses the threshold controller's trajectory in training

If we used the noiseless optimal cell as `current_cell` in the training data,
the RF would learn a "stay put" shortcut (because `current_cell` would equal
`next_cell` almost always). At inference time the controller's served cell
lags optimal — so the model would see an out-of-distribution input and
collapse.

Solution: run the threshold (A3-event) controller alongside data generation
and record its served cell as `current_cell` in the training rows. This makes
the training distribution match what *any* practical controller — including
the ML one — actually serves at inference time. The model never sees the
noiseless optimum as an input, only as the supervised label.

### ML controller decision rule

The proposal's primary trigger is the confidence threshold. Three
engineering safeguards refuse to commit when the resulting HO would
obviously be harmful — these do not relax the proposal's rule, they are the
ML analogue of what the threshold baseline already gets from
hysteresis + TTT.

```python
if (pred_cell != current_cell
    and conf >= ML_CONFIDENCE_THRESHOLD          # proposal trigger
    and gain_db >= ML_MIN_GAIN_DB                # don't act on marginal gain
    and (t - last_ho_time) >= ML_COOLDOWN_STEPS  # anti-flap
    and not bounceback):                          # anti ping-pong
    trigger_handover()
```

### Threshold baseline

Standard LTE A3-event: handover when a neighbour cell exceeds the serving
cell RSSI by ≥ `THR_HYSTERESIS_DB` dB for ≥ `THR_TTT_STEPS` consecutive
steps.

### No data leakage

| Audit                                  | Status        |
| -------------------------------------- | ------------- |
| Position (`x`, `y`) used as a feature  | ❌ excluded    |
| `optimal_cell` (ground truth) as input | ❌ excluded    |
| Label `next_cell` derivable from input | ❌ no (label is 5 steps in the future) |
| Train/val/test split                   | ✅ user-level (no user appears in two splits) |
| Confusion matrix / accuracy reporting  | ✅ test split only |
| Reproducibility                        | ✅ deterministic given `RANDOM_SEED = 42` |

---

## Configuration Cheat Sheet (`config.py`)

| Constant                  | Value   | Meaning |
| ------------------------- | ------- | ------- |
| `RANDOM_SEED`             | 42      | Global seed for everything |
| `GRID_ROWS × GRID_COLS`   | 3 × 3   | Nine base stations |
| `GRID_WIDTH × GRID_HEIGHT`| 3000×3000 m | 1000 m inter-site distance |
| `NUM_USERS`               | 60      | Users in the dataset |
| `NUM_STEPS`               | 700     | Seconds recorded per user (after burn-in) |
| `BURN_IN`                 | 100     | Discarded warm-up steps |
| `LOOKAHEAD`               | 5       | Steps ahead for the `next_cell` label |
| `SHADOWING_STD`           | 8.0 dB  | Log-normal shadowing σ |
| `RF_N_ESTIMATORS`         | 100     | Trees in the Random Forest |
| `RF_MAX_DEPTH`            | 20      | Max tree depth |
| `THR_HYSTERESIS_DB`       | 3.0 dB  | A3-event margin |
| `THR_TTT_STEPS`           | 3       | Time-to-trigger |
| `ML_CONFIDENCE_THRESHOLD` | 0.35    | RF prediction must be ≥ this to trigger HO |
| `ML_MIN_GAIN_DB`          | 4.0 dB  | Refuse HO if instantaneous gain is below this |
| `ML_COOLDOWN_STEPS`       | 10      | Block any HO for this many steps after one |
| `ML_BOUNCEBACK_WINDOW`    | 20      | Forbid returning to the cell we just left |

---

## Output Plots

The pipeline writes seven PNG plots to `results/`:

| File                        | What it shows                                                |
| --------------------------- | ------------------------------------------------------------ |
| `01_cell_grid.png`          | Base-station layout with the user trajectory coloured by ML serving cell |
| `02_rssi_over_time.png`     | RSSI signals from every cell over time; background shaded with serving cell (Threshold vs ML) |
| `03_handover_timeline.png`  | Step plot of serving cell over time: Optimal vs Threshold vs ML |
| `04_metric_comparison.png`  | Six-panel bar chart of every comparison metric               |
| `05_rssi_heatmap.png`       | Spatial RSSI map for one base station (sanity-check path-loss) |
| `06_confusion_matrix.png`   | RF test-set confusion matrix (row-normalised)                |
| `07_feature_importance.png` | Random Forest feature importance ranking                     |

---

## Mapping to the Proposal

| Proposal section                                  | Implementation                                                   |
| ------------------------------------------------- | ---------------------------------------------------------------- |
| §1 Problem statement (late HOs, ping-pong, etc.)  | Reproduced by the threshold baseline; ML eliminates ping-pong    |
| §2 Mobility + RSSI simulation                     | `simulation/mobility.py`, `simulation/rssi.py`                   |
| §2 Feature extraction                             | `ml/features.py` (13 features matching the proposal text)        |
| §2 Random Forest predictor                        | `ml/trainer.py`, `ml/predictor.py`                               |
| §2 Confidence-threshold trigger                   | `handover/ml_handover.py`                                        |
| §2 Threshold baseline                             | `handover/threshold_handover.py`                                 |
| §3 Physical / Link / Network layers               | RSSI model / HO signaling / mobility management — all simulated  |
| §4 Python: pandas, numpy, sklearn, matplotlib     | `requirements.txt`                                               |
| §4 Evaluation metrics                             | `evaluation/metrics.py`                                          |
| Figure 1: five-block architecture                 | `data/` → `ml/features` → `ml/predictor` → `handover/` → `evaluation/` |

---

## References

1. J. F. Kurose, K. W. Ross — *Computer Networking: A Top-Down Approach*, 6th Ed., Pearson, 2012.
2. W. Stallings — *Wireless Communications and Networks*, 2nd Ed., Prentice Hall, 2005.
3. V. Párraga-Villamar et al. — "Brief Survey: Machine Learning in Handover Cellular Network," Eng. Proc., MDPI, 2023. https://doi.org/10.3390/engproc2023047002
