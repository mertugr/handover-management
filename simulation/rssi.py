"""
RSSI computation using the 3GPP COST-231 Urban Macro path-loss model.
Adds log-normal shadowing noise and clips results to a valid dBm range.
"""

import numpy as np
from simulation.cell_grid import BASE_STATIONS, NUM_CELLS
from config import (P_TX_DBM, CABLE_LOSS_DB, ANTENNA_GAIN,
                    PL_CONST, PL_SLOPE, SHADOWING_STD,
                    MIN_RSSI, MAX_RSSI, MIN_DISTANCE)

# Effective isotropic radiated power
EIRP_DBM = P_TX_DBM - CABLE_LOSS_DB + ANTENNA_GAIN  # 56 dBm


def path_loss_db(distance_m: float) -> float:
    """3GPP COST-231 path-loss in dB for a given distance in metres."""
    d_km = max(distance_m, MIN_DISTANCE) / 1000.0
    return PL_CONST + PL_SLOPE * np.log10(d_km)


def rssi_from_cell(position: np.ndarray, cell_id: int,
                   add_noise: bool = True, rng=None) -> float:
    """Return RSSI (dBm) received at position from a single base station."""
    dist = float(np.linalg.norm(position - BASE_STATIONS[cell_id]))
    rssi = EIRP_DBM - path_loss_db(dist)
    if add_noise:
        shadow = (rng.normal(0.0, SHADOWING_STD) if rng is not None
                  else np.random.normal(0.0, SHADOWING_STD))
        rssi += shadow
    return float(np.clip(rssi, MIN_RSSI, MAX_RSSI))


def rssi_all_cells(position: np.ndarray,
                   add_noise: bool = True, rng=None) -> np.ndarray:
    """Return RSSI (dBm) from all NUM_CELLS base stations as an array."""
    return np.array([
        rssi_from_cell(position, i, add_noise=add_noise, rng=rng)
        for i in range(NUM_CELLS)
    ], dtype=np.float64)


def best_cell_by_rssi(position: np.ndarray) -> int:
    """Return the cell ID with the highest noiseless RSSI (ground-truth label)."""
    return int(np.argmax(rssi_all_cells(position, add_noise=False)))
