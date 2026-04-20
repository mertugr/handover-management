"""
Handover performance metrics.
Computes success rate, ping-pong count, unnecessary handover count,
and total interruption time from a completed simulation run.
"""

import numpy as np
from dataclasses import dataclass, asdict

PING_PONG_WINDOW_S     = 10   # handback within this many seconds counts as ping-pong
INTERRUPTION_MS_PER_HO = 50   # estimated interruption per handover (ms)
# A handover whose RSSI gain barely exceeds the 3 dB baseline hysteresis was
# likely triggered by shadowing noise (sigma = 8 dB) rather than a real move,
# so any HO with gain below this threshold is flagged as unnecessary.
UNNECESSARY_GAIN_DB    = 5.0


@dataclass
class HandoverMetrics:
    total_handovers:       int   = 0
    ping_pong_count:       int   = 0
    ping_pong_rate_pct:    float = 0.0
    unnecessary_count:     int   = 0
    unnecessary_rate_pct:  float = 0.0
    total_interruption_ms: float = 0.0
    ho_rate_per_100_steps: float = 0.0
    success_rate_pct:      float = 0.0
    avg_serving_rssi:      float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def compute_metrics(log: list[dict], total_steps: int,
                    served_cells: np.ndarray, true_cells: np.ndarray,
                    rssi_matrix: np.ndarray) -> HandoverMetrics:
    """Compute all metrics from a completed simulation run."""
    m = HandoverMetrics()
    m.total_handovers = len(log)

    if total_steps == 0:
        return m

    m.ho_rate_per_100_steps = (m.total_handovers / total_steps) * 100.0
    m.total_interruption_ms = m.total_handovers * INTERRUPTION_MS_PER_HO

    # Ping-pong: handback to previous cell within the time window.
    # Logs from multiple users are concatenated, so we must skip pairs that
    # cross user boundaries (otherwise negative time diffs pass the check and
    # cause false positives).
    for i in range(1, len(log)):
        prev, curr = log[i - 1], log[i]
        if prev.get("user_id") != curr.get("user_id"):
            continue
        dt = curr["time"] - prev["time"]
        if curr["to_cell"] == prev["from_cell"] and 0 < dt <= PING_PONG_WINDOW_S:
            m.ping_pong_count += 1

    if m.total_handovers > 0:
        m.ping_pong_rate_pct = m.ping_pong_count / m.total_handovers * 100.0

    # Unnecessary: RSSI gain was too small to justify the handover
    for entry in log:
        if entry.get("gain_db", 0.0) < UNNECESSARY_GAIN_DB:
            m.unnecessary_count += 1

    if m.total_handovers > 0:
        m.unnecessary_rate_pct = m.unnecessary_count / m.total_handovers * 100.0

    # Success rate: fraction of steps on the optimal cell
    n = min(len(served_cells), len(true_cells))
    if n > 0:
        m.success_rate_pct = float(
            np.sum(served_cells[:n] == true_cells[:n]) / n * 100.0
        )

    # Average RSSI of the served cell
    if len(rssi_matrix) > 0 and len(served_cells) > 0:
        n = min(len(rssi_matrix), len(served_cells))
        m.avg_serving_rssi = float(
            np.mean([rssi_matrix[t, served_cells[t]] for t in range(n)])
        )

    return m
