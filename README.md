# Machine Learning-Based Predictive Handover Management in Mobile Cellular Networks

**CSE476 Mobile Communication Networks — Term Project**
Authors: **Utku Gökçek** and **Mert Uğur**

A simulation framework that compares a traditional **threshold (LTE A3-event)**
handover controller with a **Random Forest** predictive controller in a small
cellular grid. The ML controller predicts the next best serving cell from
RSSI, speed, and direction features and triggers proactive handovers, reducing
ping-pong effects and unnecessary switching.

---

## System Architecture

```
Mobile User /     ->  Feature      ->  Random Forest  ->  Handover        ->  Performance
Trace Generator       Extractor        Predictor          Decision Module     Evaluation
(position, speed,     (speed, dir,     (trained RF:       (ML vs Threshold    (success,
 direction, RSSI)      RSSI, trend)     P(next cell))      baseline)           ping-pong,
                                                                               unnecessary,
                                                                               interruption)
```

Related network layers:
- **Physical:** RSSI / path loss (3GPP COST-231 Urban Macro)
- **Link:** A3-event handover signaling (hysteresis + TTT)
- **Network:** Mobility management and attachment-point changes

---

## Installation

```bash
git clone https://github.com/mertugr/handover-management.git
cd handover-management
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+.

## Quick Start

```bash
# First run: generates the synthetic dataset, trains the RF, runs both controllers
python3 main.py --users 20 --plot-user 0

# Subsequent runs reuse cached traces and model
python3 main.py

# Force-regenerate dataset and retrain model from scratch
python3 main.py --regenerate --retrain
```

Outputs are written to `results/` (seven PNG plots) and the model is cached at
`ml/rf_model.joblib`. Raw traces are cached at `data/traces.csv`.

### CLI Flags

| Flag           | Default | Description                                  |
|----------------|---------|----------------------------------------------|
| `--users N`    | 20      | Number of users to evaluate                  |
| `--plot-user`  | 0       | User ID to visualise in per-user plots       |
| `--regenerate` | off     | Force re-generation of synthetic traces      |
| `--retrain`    | off     | Force re-training of the Random Forest model |

---

## Project Structure

```
handover-management/
├── config.py                       # All simulation / model constants
├── main.py                         # Pipeline entry point (5 steps)
├── simulation/
│   ├── cell_grid.py                # 3x3 base-station layout
│   ├── mobility.py                 # Random Waypoint mobility model
│   └── rssi.py                     # COST-231 path-loss + shadowing
├── data/
│   └── mock_data_generator.py      # Synthetic trace generator (cached)
├── ml/
│   ├── features.py                 # Feature vector definition
│   ├── trainer.py                  # User-level train/val/test + RF fit
│   └── predictor.py                # Online inference wrapper
├── handover/
│   ├── threshold_handover.py       # LTE A3-event baseline
│   └── ml_handover.py              # RF-based proactive controller
├── evaluation/
│   └── metrics.py                  # Success / ping-pong / unnecessary / interruption
└── visualization/
    └── plots.py                    # 7 PNG plots saved to results/
```

---

## How It Works

**1. Data generation.** `NUM_USERS = 60` users are simulated for `NUM_STEPS = 700`
time steps each under a Random Waypoint mobility model on a 3×3 cellular grid
(1 km inter-site distance). At every step, RSSI is computed from all 9 base
stations using the 3GPP COST-231 path-loss model with 8 dB log-normal shadowing.
The record for each step stores position, speed, direction (sin/cos), 9 RSSI
values, 9 RSSI trends (Δ from the previous step), and the optimal cell
`LOOKAHEAD = 5` steps in the future as the supervised label.

**2. Random Forest training.** Users are split into train / validation / test
sets (user-level to avoid temporal leakage). The feature vector has 21 entries
(`speed`, `dir_sin`, `dir_cos`, 9 RSSI, 9 RSSI trend). `current_cell` is
intentionally excluded from features to avoid a distribution shift between
training (ground-truth optimal cell) and inference (the controller's serving
cell, which can be stale). A `RandomForestClassifier` is fit with
`n_estimators=100`, `max_depth=20`, `min_samples_leaf=4`,
`class_weight="balanced_subsample"`.

**3. Handover simulation.** For each user, both controllers are run over the
same RSSI trace:

- **Threshold baseline:** LTE A3 event — handover is triggered when a
  neighbour exceeds the serving cell by `THR_HYSTERESIS_DB = 3.0` dB for
  `THR_TTT_STEPS = 3` consecutive steps.
- **ML controller:** the RF predicts the next best cell each step; a
  handover is triggered only if all hold:
  (i) predicted cell ≠ current cell,
  (ii) confidence ≥ `ML_CONFIDENCE_THRESHOLD = 0.60`,
  (iii) RSSI gain to the predicted cell ≥ `ML_MIN_GAIN_DB = 4.0` dB,
  (iv) cooldown of `ML_COOLDOWN_STEPS = 10` steps since the last HO,
  (v) not a bounceback to the previous cell within
  `ML_BOUNCEBACK_WINDOW = 20` steps.

**4. Evaluation.** Both runs are compared on success rate, ping-pong count,
unnecessary handover count (RSSI gain < 5 dB), total handovers, interruption
time (50 ms/HO), and average serving-cell RSSI.

---

## Sample Results (20 users, 14 000 steps)

| Metric                     | Threshold | ML     | Δ        |
|----------------------------|-----------|--------|----------|
| Success rate (%)           | 83.52     | **90.39**  | **+6.87 pp** |
| Ping-pong count            | 20        | **0**      | **−100 %**   |
| Unnecessary handovers      | 17        | **11**     | **−35 %**    |
| Total handovers            | 205       | 204    | tie      |
| Interruption time (ms)     | 10 250    | 10 200 | tie      |
| Avg serving RSSI (dBm)     | −55.50    | **−55.20** | **+0.30 dB** |

Random Forest test accuracy: **76.4 %**, macro F1: **0.764**.

The ML controller dominates the three metrics highlighted in the proposal
(unnecessary handovers, ping-pong events, interruption time) while also
improving the success rate and maintaining the average serving RSSI.

---

## Generated Plots (`results/`)

| File                           | Contents                                                    |
|--------------------------------|-------------------------------------------------------------|
| `01_cell_grid.png`             | Base-station layout and one user's ML-served trajectory     |
| `02_rssi_over_time.png`        | RSSI curves with serving-cell shading, Threshold vs ML      |
| `03_handover_timeline.png`     | Optimal / Threshold / ML serving cell over time             |
| `04_metric_comparison.png`     | Grouped bar chart of the five performance metrics           |
| `05_rssi_heatmap.png`          | Spatial RSSI coverage for a single base station             |
| `06_confusion_matrix.png`      | RF test-set confusion matrix                                |
| `07_feature_importance.png`    | RF feature importance ranking                               |

---

## References

1. J. F. Kurose and K. W. Ross, *Computer Networking: A Top-Down Approach*, 6th Ed., Pearson, 2012.
2. W. Stallings, *Wireless Communications and Networks*, 2nd Ed., Prentice Hall, 2005.
3. V. Párraga-Villamar et al., "Brief Survey: Machine Learning in Handover Cellular Network," *Eng. Proc.*, MDPI, 2023. https://doi.org/10.3390/engproc2023047002
4. Y. Zheng et al., "GeoLife: A Collaborative Social Networking Service among User, Location and Trajectory," *IEEE Data Eng. Bulletin*, vol. 33, no. 2, 2010.
