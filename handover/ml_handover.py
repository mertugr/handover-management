"""
ML-based proactive handover controller.

Primary trigger (proposal §2): trigger HO when the RF prediction's confidence
exceeds the configured threshold. Three engineering safeguards prevent the
controller from acting on obviously harmful predictions — these do not relax
the proposal's rule, they just refuse to commit when the resulting HO would
be (a) marginal in gain, (b) immediately after another HO, or (c) a
bounceback to the cell we just left. A practical handover controller needs
these the same way the threshold baseline needs hysteresis + TTT.

Gates applied at every step (all must hold to trigger HO):
  * pred_cell != current_cell
  * confidence >= ML_CONFIDENCE_THRESHOLD     ← proposal trigger
  * target RSSI - serving RSSI >= ML_MIN_GAIN_DB
  * (t - last_ho_time) >= ML_COOLDOWN_STEPS
  * target != previously-left cell within ML_BOUNCEBACK_WINDOW steps
"""

from __future__ import annotations

import numpy as np

from config import (ML_CONFIDENCE_THRESHOLD, ML_COOLDOWN_STEPS,
                    ML_MIN_GAIN_DB, ML_BOUNCEBACK_WINDOW)
from ml.predictor import HandoverPredictor


class MLHandoverController:

    def __init__(self,
                 predictor: HandoverPredictor,
                 confidence_threshold: float = ML_CONFIDENCE_THRESHOLD,
                 cooldown_steps:       int   = ML_COOLDOWN_STEPS,
                 min_gain_db:          float = ML_MIN_GAIN_DB,
                 bounceback_window:    int   = ML_BOUNCEBACK_WINDOW):
        self.predictor         = predictor
        self.conf_threshold    = confidence_threshold
        self.cooldown          = cooldown_steps
        self.min_gain_db       = min_gain_db
        self.bounceback_window = bounceback_window

        self.current_cell: int | None = None
        self.user_id                  = None
        self.log: list[dict]          = []

        self._last_ho_time:     int = -10 ** 9
        self._last_source_cell: int = -1
        self._last_source_time: int = -10 ** 9

        # Pre-computed prediction table indexed by [t, current_cell].
        self._pred_table: np.ndarray | None = None
        self._conf_table: np.ndarray | None = None

    def reset(self, initial_cell: int = 0, user_id=None):
        self.current_cell       = initial_cell
        self.user_id            = user_id
        self.log                = []
        self._last_ho_time      = -10 ** 9
        self._last_source_cell  = -1
        self._last_source_time  = -10 ** 9
        self._pred_table        = None
        self._conf_table        = None

    def precompute(self, speeds: np.ndarray, directions_rad: np.ndarray,
                   rssi_matrix: np.ndarray) -> None:
        """Pre-compute (pred_cell, confidence) for every (t, current_cell) pair.

        Because `current_cell` is now an input feature, the controller's
        served cell is dynamic and we can no longer batch-predict the whole
        trace under a single current_cell. predictor.predict_table() runs
        NUM_CELLS batch calls so process_step() is O(1) lookup. Math is
        identical to per-step inference.
        """
        self._pred_table, self._conf_table = self.predictor.predict_table(
            speeds, directions_rad, rssi_matrix
        )

    def process_step(self, t: int, speed: float, direction_rad: float,
                     rssi: np.ndarray) -> tuple[int, bool]:
        """
        Evaluate the ML handover decision at step t.
        Returns (serving_cell, handover_occurred).
        """
        if self._pred_table is None:
            raise RuntimeError(
                "MLHandoverController.precompute() must be called after "
                "reset() and before the first process_step()."
            )

        if self.current_cell is None:
            self.current_cell = int(np.argmax(rssi))

        pred_cell = int(self._pred_table[t, self.current_cell])
        conf      = float(self._conf_table[t, self.current_cell])

        ho_occurred = False

        if pred_cell != self.current_cell:
            gain_db     = float(rssi[pred_cell] - rssi[self.current_cell])
            confident   = conf >= self.conf_threshold       # proposal trigger
            cooldown_ok = (t - self._last_ho_time) >= self.cooldown
            gain_ok     = gain_db >= self.min_gain_db
            bounceback  = (pred_cell == self._last_source_cell
                           and (t - self._last_source_time) < self.bounceback_window)

            if confident and gain_ok and cooldown_ok and not bounceback:
                self.log.append({
                    "user_id":      self.user_id,
                    "time":         t,
                    "from_cell":    self.current_cell,
                    "to_cell":      pred_cell,
                    "serving_rssi": float(rssi[self.current_cell]),
                    "target_rssi":  float(rssi[pred_cell]),
                    "gain_db":      gain_db,
                    "confidence":   conf,
                })
                self._last_source_cell = self.current_cell
                self._last_source_time = t
                self.current_cell      = pred_cell
                self._last_ho_time     = t
                ho_occurred            = True

        return self.current_cell, ho_occurred
