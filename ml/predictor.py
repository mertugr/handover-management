"""
Thin inference wrapper around the trained Random Forest classifier.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ml.features import FEATURE_COLS, CURRENT_CELL_IDX, build_feature_vector
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

    def predict_next_cell(self, current_cell: int, speed: float,
                          direction_rad: float, rssi: np.ndarray
                          ) -> tuple[int, float, np.ndarray]:
        """Return (predicted_cell, confidence, full_prob_vector)."""
        feat = build_feature_vector(current_cell, speed, direction_rad, rssi)
        proba_full = self.predict_proba_row(feat)
        pred = int(np.argmax(proba_full))
        conf = float(proba_full[pred])
        return pred, conf, proba_full

    def _build_feature_matrix(self, current_cells: np.ndarray,
                              speeds: np.ndarray, directions_rad: np.ndarray,
                              rssi_matrix: np.ndarray) -> np.ndarray:
        T = len(speeds)
        feats = np.empty((T, len(FEATURE_COLS)), dtype=np.float64)
        feats[:, CURRENT_CELL_IDX] = current_cells
        feats[:, 1] = speeds
        feats[:, 2] = np.sin(directions_rad)
        feats[:, 3] = np.cos(directions_rad)
        feats[:, 4:4 + NUM_CELLS] = rssi_matrix
        return feats

    def predict_batch(self, current_cells: np.ndarray, speeds: np.ndarray,
                      directions_rad: np.ndarray, rssi_matrix: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
        """Batch RF inference for an entire user trace under a given
        current_cell sequence.

        Inputs (all length T):
            current_cells   : currently served cell at each step
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
        if len(current_cells) != T:
            raise ValueError(
                f"current_cells length {len(current_cells)} does not match "
                f"speeds length {T}"
            )

        feats = self._build_feature_matrix(current_cells, speeds,
                                            directions_rad, rssi_matrix)

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

    def predict_table(self, speeds: np.ndarray, directions_rad: np.ndarray,
                      rssi_matrix: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
        """Pre-compute predictions for EVERY possible current_cell value.

        At inference, the ML controller's served cell is dynamic (depends on
        its own past decisions), so a single batch call is not enough. This
        helper runs NUM_CELLS batch calls, returning two (T, NUM_CELLS) arrays
        so the controller can look up `[t, current_cell]` in O(1).

        The mobility trace (speed, direction, RSSI) is independent of the
        controller's decisions, so the lookup is bit-identical to per-step
        inference but ~100x faster.
        """
        T = len(speeds)
        preds = np.zeros((T, NUM_CELLS), dtype=np.int64)
        confs = np.zeros((T, NUM_CELLS), dtype=np.float64)
        for c in range(NUM_CELLS):
            cur = np.full(T, c, dtype=np.int64)
            p, k = self.predict_batch(cur, speeds, directions_rad, rssi_matrix)
            preds[:, c] = p
            confs[:, c] = k
        return preds, confs
