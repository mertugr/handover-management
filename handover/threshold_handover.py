"""
Baseline LTE A3-event handover controller.
Triggers a handover when a neighbour cell exceeds the serving cell RSSI
by at least `hysteresis` dB for `ttt` consecutive steps.
"""

import numpy as np
from simulation.cell_grid import NUM_CELLS
from config import THR_HYSTERESIS_DB as DEFAULT_HYSTERESIS_DB, \
                   THR_TTT_STEPS     as DEFAULT_TTT_STEPS


class ThresholdHandoverController:

    def __init__(self, hysteresis: float = DEFAULT_HYSTERESIS_DB,
                 ttt: int = DEFAULT_TTT_STEPS):
        self.hysteresis   = hysteresis
        self.ttt          = ttt
        self.current_cell = None
        self.log: list[dict] = []
        self._ttt_counters: dict[int, int] = {}

    def reset(self, initial_cell: int = 0):
        self.current_cell  = initial_cell
        self.log           = []
        self._ttt_counters = {}

    def process_step(self, t: int, rssi: np.ndarray) -> tuple[int, bool]:
        """
        Evaluate the A3-event condition at time step t.
        Returns (serving_cell, handover_occurred).
        """
        if self.current_cell is None:
            self.current_cell = int(np.argmax(rssi))

        serving_rssi = rssi[self.current_cell]

        # Find the strongest neighbour cell
        rssi_copy = rssi.copy()
        rssi_copy[self.current_cell] = -np.inf
        best_cand = int(np.argmax(rssi_copy))
        best_rssi = rssi[best_cand]

        ho_occurred = False

        if best_rssi > serving_rssi + self.hysteresis:
            self._ttt_counters[best_cand] = self._ttt_counters.get(best_cand, 0) + 1

            # Reset counters for cells that no longer qualify
            for cell in list(self._ttt_counters.keys()):
                if cell != best_cand:
                    del self._ttt_counters[cell]

            if self._ttt_counters[best_cand] >= self.ttt:
                self.log.append({
                    "time":         t,
                    "from_cell":    self.current_cell,
                    "to_cell":      best_cand,
                    "serving_rssi": float(serving_rssi),
                    "target_rssi":  float(best_rssi),
                    "gain_db":      float(best_rssi - serving_rssi),
                })
                self.current_cell  = best_cand
                self._ttt_counters = {}
                ho_occurred        = True
        else:
            self._ttt_counters = {}

        return self.current_cell, ho_occurred
