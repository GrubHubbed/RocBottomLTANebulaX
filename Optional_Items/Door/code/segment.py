"""Temporal segmentation: find each door open/close cycle in a continuous stream.

Layered so that the cheap, exact method is tried first and the general ones only run
where it cannot cope:

1. **Clock-jump split.** The supplied streams are recorded cycle-by-cycle: samples run at
   a fixed period *inside* a cycle, and the clock jumps 10-60 s between cycles because
   idle time is not sampled at all. Splitting on those jumps recovers the boundaries
   exactly - verified at 110/110 on ``Train.csv`` with zero timing error.

   This is why we do not key off ``Door is opening`` / ``Door is closing``: those flags are
   set a few samples *after* motion begins, so a boundary drawn from them is
   systematically late and loses IoU on every segment.

2. **Activity split.** If a gap-delimited block is far longer than a plausible cycle, the
   recording contains idle rows rather than clock jumps, so the block is sub-split on
   motor activity.

3. **Travel split.** If a block is still too long, the cycles are butted together with no
   idle period at all. Cycles are then separated at door-travel turning points: the leaf
   runs monotonically within a cycle, so a reversal or a position jump marks a boundary.

Every threshold is derived from the stream's own sampling period rather than hard-coded,
because a held-out recording logged at a different rate would otherwise be catastrophic
rather than merely degraded (a fixed 50 ms threshold turns a 60 ms recording into zero
detected segments).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .io import COL_CURRENT, COL_POS, format_datetime

# Cycle geometry in *seconds*, measured from Train.csv (2.7-3.8 s per cycle). Row counts
# are derived from these and the stream's own sampling period, so they survive a
# recording logged at a different rate.
MIN_CYCLE_SECONDS = 1.2
MAX_CYCLE_SECONDS = 5.2

#: A gap must exceed this to end a cycle. Genuine idle periods between cycles run 10-60 s,
#: so anything sub-second is a sensor dropout and must not be treated as a boundary.
MIN_IDLE_GAP_SECONDS = 1.0
#: ...and it must also be this many sampling periods, so a slowly-logged stream still works.
GAP_PERIOD_MULTIPLE = 2.5


def sampling_period(df: pd.DataFrame) -> float:
    """Median sample-to-sample interval in seconds, ignoring inter-cycle gaps."""
    dt = np.diff(df["t"].to_numpy()).astype("timedelta64[ns]").astype(np.int64) / 1e9
    dt = dt[dt > 0]
    if len(dt) == 0:
        return 0.02
    median = float(np.median(dt))
    return median if median > 0 else 0.02


def _travel_split(pos: np.ndarray, min_rows: int) -> list[int]:
    """Cut indices where the door leaf reverses direction or jumps position.

    Within one cycle the leaf travels one way. Two cycles run back to back therefore meet
    at a turning point (open then close) or at a discontinuity (open then open, where the
    position resets). Both are boundaries.
    """
    span = float(pos.max() - pos.min())
    if span <= 0:
        return []
    cuts: set[int] = set()

    # Position discontinuity: the leaf cannot teleport, so a large single-sample step is a
    # new recording starting.
    jumps = np.flatnonzero(np.abs(np.diff(pos)) > 0.5 * span)
    cuts.update(int(j) for j in jumps)

    # Turning points, required to be prominent so ordinary wobble is ignored.
    for signal in (pos, -pos):
        peaks, _ = find_peaks(signal, prominence=0.5 * span, distance=max(min_rows, 1))
        cuts.update(int(p) for p in peaks)

    return sorted(c for c in cuts if min_rows <= c <= len(pos) - min_rows)


def _activity_runs(block: pd.DataFrame, min_rows: int, bridge_rows: int) -> list[tuple[int, int]]:
    """Runs where the motor is drawing current or the leaf is moving."""
    cur = block[COL_CURRENT].to_numpy(dtype=float)
    pos = block[COL_POS].to_numpy(dtype=float)

    baseline = np.percentile(cur, 10)
    threshold = baseline + max(0.15 * (np.percentile(cur, 90) - baseline), 20.0)
    moving = np.abs(np.diff(pos, prepend=pos[0])) > 0
    active = (cur > threshold) | moving
    if not active.any():
        return []

    idx = np.flatnonzero(active)
    runs: list[list[int]] = [[idx[0], idx[0]]]
    for i in idx[1:]:
        if i - runs[-1][1] <= bridge_rows:      # bridge a brief dip inside one cycle
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return [(a, b) for a, b in runs if (b - a + 1) >= min_rows]


def _split_block(
    block: pd.DataFrame, min_rows: int, max_rows: int, bridge_rows: int
) -> list[tuple[int, int, str]]:
    """Split one over-long block into cycles, by activity and then by door travel."""
    spans: list[tuple[int, int, str]] = []
    runs = _activity_runs(block, min_rows, bridge_rows) or [(0, len(block) - 1)]
    for a, b in runs:
        if b - a + 1 <= max_rows:
            spans.append((a, b, "activity"))
            continue
        pos = block[COL_POS].to_numpy(dtype=float)[a : b + 1]
        cuts = _travel_split(pos, min_rows)
        if not cuts:
            spans.append((a, b, "activity"))
            continue
        start = a
        for c in cuts:
            spans.append((start, a + c, "travel"))
            start = a + c + 1
        spans.append((start, b, "travel"))
    return [(a, b, how) for a, b, how in spans if (b - a + 1) >= min_rows]


def segment_stream(
    df: pd.DataFrame,
    gap_seconds: float | None = None,
    min_rows: int | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Find every door cycle in ``df``.

    Returns one row per detected segment with positional row bounds (``i0``/``i1``,
    inclusive), timestamps, and the native-format strings used for submission. Thresholds
    default to values derived from the stream's own sampling period; pass them explicitly
    to override.
    """
    columns = ["segment_id", "i0", "i1", "n_rows", "t_start", "t_end",
               "start_time", "end_time", "duration_s", "split_by"]
    if len(df) == 0:
        return pd.DataFrame(columns=columns)

    period = sampling_period(df)
    if gap_seconds is None:
        gap_seconds = max(GAP_PERIOD_MULTIPLE * period, MIN_IDLE_GAP_SECONDS)
    if min_rows is None:
        min_rows = max(int(MIN_CYCLE_SECONDS / period), 5)
    if max_rows is None:
        max_rows = max(int(MAX_CYCLE_SECONDS / period), min_rows + 1)
    bridge_rows = max(int(0.25 / period), 2)

    t = df["t"].to_numpy()
    dt = np.diff(t).astype("timedelta64[ns]").astype(np.int64) / 1e9
    breaks = np.flatnonzero(dt > gap_seconds)
    block_starts = np.concatenate([[0], breaks + 1])
    block_ends = np.concatenate([breaks, [len(df) - 1]])

    spans: list[tuple[int, int, str]] = []
    for b0, b1 in zip(block_starts, block_ends):
        n = b1 - b0 + 1
        if n <= max_rows:
            if n >= min_rows:
                spans.append((int(b0), int(b1), "time_gap"))
            continue
        for a, b, how in _split_block(df.iloc[b0 : b1 + 1], min_rows, max_rows, bridge_rows):
            spans.append((int(b0 + a), int(b0 + b), how))

    rows = []
    for k, (i0, i1, how) in enumerate(spans, start=1):
        t0, t1 = df["t"].iloc[i0], df["t"].iloc[i1]
        rows.append(
            dict(
                segment_id=f"seg_{k:03d}",
                i0=i0,
                i1=i1,
                n_rows=i1 - i0 + 1,
                t_start=t0,
                t_end=t1,
                start_time=format_datetime(t0),
                end_time=format_datetime(t1),
                duration_s=(t1 - t0).total_seconds(),
                split_by=how,
            )
        )
    return pd.DataFrame(rows, columns=columns)


def iter_segments(df: pd.DataFrame, segments: pd.DataFrame):
    """Yield ``(segment_row, segment_dataframe)`` pairs."""
    for _, seg in segments.iterrows():
        yield seg, df.iloc[int(seg.i0) : int(seg.i1) + 1]
