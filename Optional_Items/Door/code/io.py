"""Loading and timestamp handling for the Door subsystem.

The dataset's native timestamp format is Year-Month-Day-Hour-Minute-Second-Millisecond,
hyphen separated and *not* zero padded (e.g. ``2023-7-5-0-0-3-760``), which breaks
``pd.to_datetime`` defaults - hence the custom parser here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Column names exactly as they appear in Train.csv / Test.csv
COL_TIME = "Datetime"
COL_CURRENT = "Motor current(mA)"
COL_VOLTAGE = "Motor Voltage(10mV)"
COL_EMF = "Motor electrodynamic force"
COL_POS = "Door leaf position"
COL_OPEN_CMD = "Open command"
COL_CLOSE_CMD = "Close command"
COL_OPENING = "Door is opening"
COL_CLOSING = "Door is closing"

SAMPLE_PERIOD_S = 0.02  # 50 Hz


def parse_datetime(s: pd.Series) -> pd.Series:
    """Parse the native 'Y-M-D-h-m-s-ms' format into pandas datetimes."""
    p = s.astype(str).str.strip().str.split("-", expand=True).astype(int)
    base = pd.to_datetime(
        dict(year=p[0], month=p[1], day=p[2], hour=p[3], minute=p[4], second=p[5])
    )
    return base + pd.to_timedelta(p[6], unit="ms")


def format_datetime(ts: pd.Timestamp) -> str:
    """Render a timestamp back in the dataset's native (non zero-padded) format."""
    ts = pd.Timestamp(ts)
    return (
        f"{ts.year}-{ts.month}-{ts.day}-{ts.hour}-{ts.minute}-{ts.second}-"
        f"{int(ts.microsecond // 1000)}"
    )


def load_stream(path: str | Path) -> pd.DataFrame:
    """Load a continuous door telemetry stream and attach a parsed ``t`` column."""
    df = pd.read_csv(path)
    missing = [c for c in (COL_TIME, COL_CURRENT, COL_EMF, COL_POS) if c not in df.columns]
    if missing:
        raise ValueError(f"{Path(path).name} is missing required column(s): {missing}")
    df = df.copy()
    df["t"] = parse_datetime(df[COL_TIME])
    if not df["t"].is_monotonic_increasing:
        df = df.sort_values("t").reset_index(drop=True)
    return df


def load_answers(path: str | Path) -> pd.DataFrame:
    """Load Train_Segments_Answer.csv with parsed start/end timestamps."""
    ans = pd.read_csv(path)
    ans["t_start"] = parse_datetime(ans["start_time"])
    ans["t_end"] = parse_datetime(ans["end_time"])
    return ans


def to_seconds(t: pd.Series | np.ndarray) -> np.ndarray:
    """Convert timestamps to float seconds since epoch (used by the IoU metric)."""
    return pd.to_datetime(pd.Series(t)).astype("int64").to_numpy() / 1e9
