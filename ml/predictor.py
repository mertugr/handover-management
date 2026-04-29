"""
Thin inference wrapper around the trained Random Forest classifier.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ml.features import FEATURE_COLS, build_feature_vector
from simulation.cell_grid import NUM_CELLS


class HandoverPredictor:
    """Wraps a trained RandomForestClassifier for online handover prediction."""

    def __init__(self, model: RandomForestClassifier):
        self.model   = model
        self.classes = np.asarray(model.classes_, dtype=np.int64)

    def predict_proba_row(self, features: np.ndarray) -> np.ndarray:
        """Probability over ALL NUM_CELLS cells (missing classes filled with 0)."""
        proba = self.model.predict_proba(features.reshape(1, -1))[0]

        full = np.zeros(NUM_CELLS, dtype=np.float64)
        for idx, cls in enumerate(self.classes):
            if 0 <= cls < NUM_CELLS:
                full[cls] = proba[idx]
        return full

    def predict_next_cell(self, speed: float, direction_rad: float,
                          rssi: np.ndarray, rssi_prev: np.ndarray
                          ) -> tuple[int, float, np.ndarray]:
        """Return (predicted_cell, confidence, full_prob_vector)."""
        feat = build_feature_vector(speed, direction_rad, rssi, rssi_prev)
        proba_full = self.predict_proba_row(feat)
        pred = int(np.argmax(proba_full))
        conf = float(proba_full[pred])
        return pred, conf, proba_full

    def predict_batch(self, speeds: np.ndarray, directions_rad: np.ndarray,
                      rssi_matrix: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
        """Batch RF inference for an entire user trace.

        Per-row predict_proba calls dominate runtime (~14 ms each); a single
        batch call is ~100x faster and produces identical results because the
        mobility trace is independent of any handover decisions.

        Inputs (all length T):
            speeds          : speed in m/s
            directions_rad  : direction in radians
            rssi_matrix     : (T, NUM_CELLS) RSSI snapshots

        Returns:
            pred_cells  : (T,) int64,  argmax cell at each step
            confidences : (T,) float64, confidence of the predicted cell
        """
        T = len(speeds)
        if rssi_matrix.shape != (T, NUM_CELLS):
            raise ValueError(
                f"rssi_matrix shape {rssi_matrix.shape} does not match "
                f"({T}, {NUM_CELLS})"
            )

        # Trend at t = rssi[t] - rssi[t-1], with trend[0] = 0 to match the
        # data generator's first-step convention (mock_data_generator.py).
        rssi_prev = np.empty_like(rssi_matrix)
        rssi_prev[0]  = rssi_matrix[0]
        rssi_prev[1:] = rssi_matrix[:-1]
        trend = rssi_matrix - rssi_prev

        feats = np.empty((T, len(FEATURE_COLS)), dtype=np.float64)
        feats[:, 0] = speeds
        feats[:, 1] = np.sin(directions_rad)
        feats[:, 2] = np.cos(directions_rad)
        feats[:, 3:3 + NUM_CELLS]                = rssi_matrix
        feats[:, 3 + NUM_CELLS:3 + 2 * NUM_CELLS] = trend

        proba = self.model.predict_proba(feats)  # (T, n_classes_seen)

        # Re-map to a full (T, NUM_CELLS) matrix in case some cells were
        # absent from the training labels.
        proba_full = np.zeros((T, NUM_CELLS), dtype=np.float64)
        for idx, cls in enumerate(self.classes):
            if 0 <= cls < NUM_CELLS:
                proba_full[:, cls] = proba[:, idx]

        pred_cells  = np.argmax(proba_full, axis=1).astype(np.int64)
        confidences = proba_full[np.arange(T), pred_cells]
        return pred_cells, confidences
