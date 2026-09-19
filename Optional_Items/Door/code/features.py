"""Per-cycle feature extraction.

Every feature is computed from a *single* segment, so nothing leaks across cycles or from
the validation set. Features are grouped by the physics they describe:

* **Load**      - motor current statistics: a door meeting extra resistance draws more current.
* **Speed**     - back-EMF and leaf speed: a dragging door runs slower.
* **Resistance**- current per unit speed / per unit back-EMF. This is the ratio that actually
                  isolates mechanical resistance from ordinary load variation, and it is the
                  feature family that survives the "different doors have different baselines"
                  problem called out in the Info Kit (section 1.2).
* **Roughness** - sample-to-sample variation of current: grit and a jammed rubber strip show up
                  as a rough, stick-slip current trace rather than a uniformly higher one.
* **Shape**     - the cycle's current profile over normalised travel time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from .io import (
    COL_CLOSING,
    COL_CURRENT,
    COL_EMF,
    COL_OPENING,
    COL_POS,
    COL_VOLTAGE,
    SAMPLE_PERIOD_S,
)

EPS = 1e-6


def _safe(x: float) -> float:
    return float(x) if np.isfinite(x) else 0.0


def infer_operation(seg: pd.DataFrame) -> str:
    """Open vs Close, from the direction the leaf actually travels.

    Falls back to the controller status flags only if the leaf barely moves.
    """
    pos = seg[COL_POS].to_numpy(dtype=float)
    delta = pos[-1] - pos[0]
    if abs(delta) > 10:
        return "Open" if delta > 0 else "Close"
    if COL_OPENING in seg.columns and seg[COL_OPENING].mean() > seg.get(COL_CLOSING, 0).mean():
        return "Open"
    return "Close"


def segment_features(seg: pd.DataFrame) -> dict:
    """Reduce one door cycle to a fixed-length feature vector."""
    dt = SAMPLE_PERIOD_S
    cur = seg[COL_CURRENT].to_numpy(dtype=float)
    emf = seg[COL_EMF].to_numpy(dtype=float)
    volt = seg[COL_VOLTAGE].to_numpy(dtype=float)
    pos = seg[COL_POS].to_numpy(dtype=float)
    n = len(cur)

    speed = np.abs(np.diff(pos, prepend=pos[0])) / dt          # leaf speed, units/s
    travel = abs(pos[-1] - pos[0])
    # Steady-travel window: ignore the inrush at start and the latching at the end.
    lo, hi = int(0.2 * n), max(int(0.8 * n), int(0.2 * n) + 1)
    cur_steady = cur[lo:hi]
    emf_steady = emf[lo:hi]
    speed_steady = speed[lo:hi]

    # Normalised-time profile of current (5 equal slices across the cycle).
    slices = np.array_split(cur, 5)

    f = {
        "n_rows": n,
        "duration_s": n * dt,
        # --- load -------------------------------------------------------------
        "cur_mean": cur.mean(),
        "cur_std": cur.std(),
        "cur_max": cur.max(),
        "cur_min": cur.min(),
        "cur_p25": np.percentile(cur, 25),
        "cur_p50": np.percentile(cur, 50),
        "cur_p75": np.percentile(cur, 75),
        "cur_p90": np.percentile(cur, 90),
        "cur_iqr": np.percentile(cur, 75) - np.percentile(cur, 25),
        "cur_rms": float(np.sqrt(np.mean(cur**2))),
        # Distribution shape. Standard motor-current-signature features: a door meeting
        # intermittent resistance produces a skewed, heavy-tailed current distribution
        # rather than simply a higher one.
        "cur_skew": float(skew(cur)) if n > 2 else 0.0,
        "cur_kurtosis": float(kurtosis(cur)) if n > 3 else 0.0,
        "cur_ptp": float(cur.max() - cur.min()),
        "emf_skew": float(skew(emf)) if n > 2 else 0.0,
        "cur_auc": float(cur.sum() * dt),                 # charge drawn over the cycle
        "cur_crest": _safe(cur.max() / (cur.mean() + EPS)),
        "cur_steady_mean": cur_steady.mean(),
        "cur_steady_p90": np.percentile(cur_steady, 90),
        # --- speed ------------------------------------------------------------
        "emf_mean": emf.mean(),
        "emf_max": emf.max(),
        "emf_p50": np.percentile(emf, 50),
        "emf_steady_mean": emf_steady.mean(),
        "emf_std": emf.std(),
        "volt_mean": volt.mean(),
        "volt_max": volt.max(),
        "speed_mean": speed.mean(),
        "speed_max": speed.max(),
        "speed_steady_mean": speed_steady.mean(),
        "travel": travel,
        # --- resistance proxies (the discriminative family) -------------------
        "cur_per_emf": _safe(cur.mean() / (emf.mean() + EPS)),
        "cur_per_emf_steady": _safe(cur_steady.mean() / (emf_steady.mean() + EPS)),
        "cur_per_speed": _safe(cur.mean() / (speed.mean() + EPS)),
        "cur_per_speed_steady": _safe(cur_steady.mean() / (speed_steady.mean() + EPS)),
        "cur_per_travel": _safe(cur.sum() * dt / (travel + EPS)),
        "work_proxy": float(np.sum(cur * emf) * dt),
        "power_proxy_mean": float(np.mean(cur * emf)),
        "emf_per_volt": _safe(emf.mean() / (volt.mean() + EPS)),
        # --- roughness / stick-slip ------------------------------------------
        "cur_absdiff_mean": float(np.mean(np.abs(np.diff(cur)))) if n > 1 else 0.0,
        "cur_absdiff_p90": float(np.percentile(np.abs(np.diff(cur)), 90)) if n > 1 else 0.0,
        "cur_zero_cross_rate": float(np.mean(np.diff(np.sign(np.diff(cur))) != 0)) if n > 2 else 0.0,
        "stall_frac": float(np.mean((speed <= 0) & (cur > np.percentile(cur, 50)))),
        "high_load_frac": float(np.mean(cur > np.percentile(cur, 75))),
        # --- profile shape ----------------------------------------------------
        **{f"cur_slice{i}_mean": float(s.mean()) for i, s in enumerate(slices)},
        **{f"cur_slice{i}_max": float(s.max()) for i, s in enumerate(slices)},
    }
    f = {k: _safe(v) for k, v in f.items()}
    f["is_open"] = 1.0 if infer_operation(seg) == "Open" else 0.0
    return f


def build_feature_table(df: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix for every segment, indexed to match ``segments``."""
    rows = []
    for _, seg in segments.iterrows():
        block = df.iloc[int(seg.i0) : int(seg.i1) + 1]
        feats = segment_features(block)
        feats["segment_id"] = seg.segment_id
        rows.append(feats)
    out = pd.DataFrame(rows).set_index("segment_id")
    out["operation"] = np.where(out["is_open"] > 0.5, "Open", "Close")
    return out


def numeric_feature_names(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c != "operation"]


# --- Stream-relative (self-referencing) normalisation --------------------------------
#
# The Info Kit's core complaint about threshold methods is that "data distributions differ
# among doors; applying a uniform threshold leads to false alarms and missed detection".
# A model trained on absolute milliamps inherits exactly that weakness: recalibrate the
# current sensor by 15% and it starts guessing.
#
# The fix is to judge each cycle against the *other cycles in its own recording* - the same
# door, the same sensor, the same calibration - rather than against absolute values learned
# from a different door. Dividing by the same-operation median over the file makes the
# feature invariant to any global gain on that channel, by construction.
#
# This uses no labels, so it introduces no label leakage; it is applied identically during
# training and at inference (each file normalised against itself). The one assumption it
# makes is that a *minority* of cycles in any given recording are faulty - true here (30 of
# 110) and true of a real door, and safe up to 50% because the statistic is a median.

STREAM_RELATIVE_PREFIXES = ("cur_", "emf_", "volt_", "speed_", "work_", "power_")


def add_stream_relative_features(table: pd.DataFrame, min_cohort: int = 4) -> pd.DataFrame:
    """Append ``self_<feature>`` columns: each cycle over its own file's same-operation median."""
    base_cols = [
        c
        for c in table.columns
        if c.startswith(STREAM_RELATIVE_PREFIXES) and c not in ("operation",)
    ]
    out = table.copy()
    rel = pd.DataFrame(index=table.index, columns=base_cols, dtype=float)
    is_open = table["is_open"] > 0.5
    for mask in (is_open, ~is_open):
        if not mask.any():
            continue
        cohort = table.loc[mask, base_cols]
        # Fall back to the whole file if this operating state is barely represented.
        med = (cohort if len(cohort) >= min_cohort else table[base_cols]).median()
        med = med.replace(0, np.nan)
        rel.loc[mask, :] = cohort.div(med, axis=1).to_numpy()
    rel = rel.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0, 50)
    return pd.concat([out, rel.add_prefix("self_")], axis=1)


#: Feature views the pipeline can be trained on.
#:   ``self_only`` - stream-relative features only. Invariant to sensor calibration and to
#:                   per-door baseline differences; the production default.
#:   ``absolute``  - raw physical units only. Used as the fallback model for short files
#:                   where a file has too few cycles to form a reliable self-baseline.
#:   ``both``      - the union, kept for benchmarking.
FEATURE_VIEWS = ("self_only", "absolute", "both")

#: Below this many cycles in a file, the self-baseline is unreliable and the absolute
#: model is used instead.
MIN_SEGMENTS_FOR_SELF_NORM = 8

_CONTEXT_COLS = ("is_open", "n_rows", "duration_s", "travel", "stall_frac", "high_load_frac")


def build_feature_view(table: pd.DataFrame, view: str = "self_only") -> pd.DataFrame:
    """Select one feature view from a full feature table (must already carry ``self_*``)."""
    if view not in FEATURE_VIEWS:
        raise ValueError(f"unknown feature view {view!r}; expected one of {FEATURE_VIEWS}")
    cols = numeric_feature_names(table)
    if view == "both":
        return table[cols]
    if view == "absolute":
        return table[[c for c in cols if not c.startswith("self_")]]
    keep = [c for c in cols if c.startswith("self_")]
    keep += [c for c in _CONTEXT_COLS if c in cols and c not in keep]
    return table[keep]


def build_features(df: pd.DataFrame, segments: pd.DataFrame, view: str = "self_only") -> pd.DataFrame:
    """End-to-end: raw stream + segments -> model-ready feature matrix for ``view``."""
    table = add_stream_relative_features(build_feature_table(df, segments))
    return build_feature_view(table, view)
