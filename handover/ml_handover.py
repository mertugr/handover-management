"""
ML-based proactive handover controller.

Uses a trained Random Forest to predict the next-best serving cell from
speed, direction, RSSI snapshot, RSSI trend, and the current cell. A
handover is triggered only if all of the following hold:
  * the predicted cell is not the current cell
  * confidence >= ML_CONFIDENCE_THRESHOLD
  * target RSSI - serving RSSI >= ML_MIN_GAIN_DB
  * cooldown has elapsed since the previous handover
  * the target is not a bounceback within ML_BOUNCEBACK_WINDOW

The ML-based strategy is compared with the threshold A3-event baseline
by running both controllers over the same RSSI trace.
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

        self._last_ho_time:      int   = -10 ** 9
        self._last_source_cell:  int   = -1
        self._last_source_time:  int   = -10 ** 9

        # Filled in by precompute(); required before process_step().
        self._cached_preds: np.ndarray | None = None
        self._cached_confs: np.ndarray | None = None

    def reset(self, initial_cell: int = 0, user_id=None):
        self.current_cell       = initial_cell
        self.user_id            = user_id
        self.log                = []
        self._last_ho_time      = -10 ** 9
        self._last_source_cell  = -1
        self._last_source_time  = -10 ** 9
        self._cached_preds      = None
        self._cached_confs      = None

    def precompute(self, speeds: np.ndarray, directions_rad: np.ndarray,
                   rssi_matrix: np.ndarray) -> None:
        """Batch-predict next-cell + confidence for the full user trace.

        Must be called after reset() and before the first process_step(). RF
        per-row inference is ~14 ms; one batch call is ~100x faster, with
        bit-identical results because the mobility trace is independent of
        handover decisions. RF-specific optimization — other controllers
        (threshold, RL) do not need an analogue.
        """
        self._cached_preds, self._cached_confs = self.predictor.predict_batch(
            speeds, directions_rad, rssi_matrix
        )

    def process_step(self, t: int, speed: float, direction_rad: float,
                     rssi: np.ndarray) -> tuple[int, bool]:
        """
        Evaluate the ML handover decision at step t.
        Returns (serving_cell, handover_occurred).
        """
        if self._cached_preds is None:
            raise RuntimeError(
                "MLHandoverController.precompute() must be called after "
                "reset() and before the first process_step()."
            )

        if self.current_cell is None:
            self.current_cell = int(np.argmax(rssi))

        pred_cell = int(self._cached_preds[t])
        conf      = float(self._cached_confs[t])

        ho_occurred = False

        if pred_cell != self.current_cell:
            gain_db      = float(rssi[pred_cell] - rssi[self.current_cell])
            cooldown_ok  = (t - self._last_ho_time) >= self.cooldown
            confident    = conf >= self.conf_threshold
            gain_ok      = gain_db >= self.min_gain_db
            bounceback   = (pred_cell == self._last_source_cell
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
