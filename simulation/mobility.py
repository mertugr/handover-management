"""
Random Waypoint (RWP) mobility model.
Users move toward randomly chosen waypoints, optionally pause on arrival,
then pick a new destination. Small directional noise creates realistic curves.
"""

import numpy as np
from simulation.cell_grid import GRID_WIDTH, GRID_HEIGHT
from config import (MIN_SPEED_MPS, MAX_SPEED_MPS, TIME_STEP_S,
                    PAUSE_PROB, MIN_PAUSE_S, MAX_PAUSE_S,
                    DIR_NOISE_STD, MARGIN)


class MobileUser:
    """Single mobile terminal moving inside the cellular grid."""

    def __init__(self, user_id: int, initial_pos=None, initial_speed=None,
                 rng: np.random.RandomState = None):
        self.user_id = user_id
        self._rng    = rng if rng is not None else np.random.RandomState(user_id)

        if initial_pos is None:
            self.position = np.array([
                self._rng.uniform(MARGIN, GRID_WIDTH  - MARGIN),
                self._rng.uniform(MARGIN, GRID_HEIGHT - MARGIN),
            ], dtype=np.float64)
        else:
            self.position = np.array(initial_pos, dtype=np.float64)

        self.speed     = float(initial_speed) if initial_speed is not None \
                         else self._rng.uniform(MIN_SPEED_MPS, MAX_SPEED_MPS)
        self.direction  = self._rng.uniform(0, 2 * np.pi)
        self._waypoint  = self._sample_waypoint()
        self._paused    = False
        self._pause_rem = 0.0

    def _sample_waypoint(self) -> np.ndarray:
        return np.array([
            self._rng.uniform(MARGIN, GRID_WIDTH  - MARGIN),
            self._rng.uniform(MARGIN, GRID_HEIGHT - MARGIN),
        ], dtype=np.float64)

    def _reflect_boundary(self, pos: np.ndarray) -> np.ndarray:
        """Reflect position off grid walls and reverse the affected direction component."""
        lo = MARGIN
        for dim, limit in enumerate([GRID_WIDTH, GRID_HEIGHT]):
            hi = limit - MARGIN
            if pos[dim] < lo:
                pos[dim] = 2 * lo - pos[dim]
                self.direction = np.pi - self.direction if dim == 0 else -self.direction
            elif pos[dim] > hi:
                pos[dim] = 2 * hi - pos[dim]
                self.direction = np.pi - self.direction if dim == 0 else -self.direction
            pos[dim] = np.clip(pos[dim], lo, limit - MARGIN)
        return pos

    def step(self):
        """Advance the user by one TIME_STEP_S second."""
        if self._paused:
            self._pause_rem -= TIME_STEP_S
            if self._pause_rem <= 0:
                self._paused   = False
                self._waypoint = self._sample_waypoint()
                self.speed     = self._rng.uniform(MIN_SPEED_MPS, MAX_SPEED_MPS)
            return

        to_wp     = self._waypoint - self.position
        dist_wp   = float(np.linalg.norm(to_wp))
        step_dist = self.speed * TIME_STEP_S

        if dist_wp <= step_dist:
            self.position = self._waypoint.copy()
            if self._rng.random() < PAUSE_PROB:
                self._paused    = True
                self._pause_rem = self._rng.uniform(MIN_PAUSE_S, MAX_PAUSE_S)
            else:
                self._waypoint = self._sample_waypoint()
                self.speed     = self._rng.uniform(MIN_SPEED_MPS, MAX_SPEED_MPS)
        else:
            target_dir     = float(np.arctan2(to_wp[1], to_wp[0]))
            noise          = self._rng.normal(0.0, DIR_NOISE_STD)
            self.direction = target_dir + noise
            dx = step_dist * np.cos(self.direction)
            dy = step_dist * np.sin(self.direction)
            self.position  = self._reflect_boundary(self.position + np.array([dx, dy]))
