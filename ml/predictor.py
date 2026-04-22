"""
Thin inference wrapper around the trained Random Forest classifier.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ml.features import build_feature_vector
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
