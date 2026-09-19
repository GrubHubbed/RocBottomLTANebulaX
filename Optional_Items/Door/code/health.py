"""Turning per-cycle predictions into maintenance decisions.

The classifier answers "is *this* cycle abnormal". A maintenance team needs something
different: which doors need attention, how urgently, and how much warning they have. Three
layers sit between the two:

**1. Noise suppression.** A single odd cycle is not a fault - a passenger holding the door,
a gust, one noisy reading. A flag is only *confirmed* when it survives:

  * *operating-state normalisation* - a cycle is judged against its own operating state
    (Open vs Close) and its own file's baseline, never a global threshold (this is done
    upstream, in the feature layer);
  * *persistence* - at least ``min_confirmations`` anomalies within a rolling window of
    ``window`` cycles, so isolated spikes never raise a work order;
  * *sensor agreement* - the current channel and the back-EMF channel must tell the same
    story. Extra mechanical resistance raises current *and* slows the door. Current alone
    rising is more consistent with a supply or sensor problem, and is reported as such
    rather than as a door fault.

**2. Event grouping.** Consecutive confirmed cycles are merged into one event, so a door
degrading over twenty cycles produces one ranked item on the screen, not twenty.

**3. Degradation trend and lead time.** The resistance index is tracked over time with a
robust (Theil-Sen) fit and extrapolated to an intervention threshold, giving an estimated
number of cycles of remaining margin - the lead time that makes the detection useful
rather than merely correct.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# The resistance index is "this cycle's steady-state current over its own file's
# same-operation median", so 1.0 is a healthy door by construction.
HEALTHY_INDEX = 1.0
#: Index at which a door is treated as needing intervention before it jams. Set from the
#: labelled data: abnormal cycles sit at ~2.2x baseline, healthy ones at ~1.0x (see the
#: training report), so 1.5x is comfortably above normal variation and below a real fault.
INTERVENTION_INDEX = 1.5

SEVERITY_BANDS = ((2.5, "Critical"), (1.8, "High"), (1.35, "Medium"), (0.0, "Low"))


def resistance_index(features: pd.DataFrame) -> pd.Series:
    """Per-cycle mechanical resistance, as a multiple of that door's own healthy baseline."""
    for col in ("self_cur_steady_mean", "self_cur_per_speed_steady", "self_cur_mean"):
        if col in features.columns:
            return features[col].astype(float).rename("resistance_index")
    # Absolute-view fallback: normalise against this file's own median.
    cur = features["cur_steady_mean"].astype(float)
    return (cur / max(cur.median(), 1e-6)).rename("resistance_index")


def speed_deficit(features: pd.DataFrame) -> pd.Series:
    """How much slower the door ran than its own baseline (>0 means slower)."""
    for col in ("self_emf_per_volt", "self_emf_steady_mean", "self_speed_steady_mean"):
        if col in features.columns:
            return (1.0 - features[col].astype(float)).rename("speed_deficit")
    return pd.Series(0.0, index=features.index, name="speed_deficit")


def severity_band(index: float) -> str:
    for threshold, name in SEVERITY_BANDS:
        if index >= threshold:
            return name
    return "Low"


@dataclass
class DegradationTrend:
    slope_per_cycle: float
    intercept: float
    current_index: float
    cycles_to_threshold: float | None
    threshold: float = INTERVENTION_INDEX
    basis: str = "absolute steady-state current of healthy cycles"
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "slope_per_cycle": self.slope_per_cycle,
            "current_index": self.current_index,
            "cycles_to_threshold": self.cycles_to_threshold,
            "threshold": self.threshold,
            "basis": self.basis,
            "note": self.note,
        }


def _theil_sen(y: np.ndarray) -> tuple[float, float]:
    """Median of pairwise slopes - a couple of extreme cycles cannot invent a trend."""
    n = len(y)
    x = np.arange(n, dtype=float)
    slopes = [
        (y[j] - y[i]) / (x[j] - x[i])
        for i in range(n)
        for j in range(i + 1, n)
    ]
    slope = float(np.median(slopes))
    return slope, float(np.median(y - slope * x))


def degradation_trend(
    absolute_load: pd.Series,
    is_normal: np.ndarray | None = None,
    threshold: float = INTERVENTION_INDEX,
    baseline_frac: float = 0.25,
) -> DegradationTrend:
    """Estimate whether the door's *healthy* operation is drifting worse over time.

    This deliberately runs on **absolute** load (mA), not the stream-relative index. The
    relative index is normalised against the file's own median, which pins its centre at
    1.0 by construction - so a trend fitted to it measures only where faults happen to
    cluster in the recording, never genuine drift. Absolute units can drift, so they are
    what a trend has to be fitted to.

    Only cycles classified Normal are used: a discrete fault is an *event*, handled by
    ``group_events``. What this answers is the different question of whether the door's
    healthy baseline is creeping upward - the early, sub-threshold degradation that buys
    a maintenance team lead time.

    Within a single recording this can only see drift inside that recording. Longer-horizon
    degradation needs the per-run history that ``predict`` appends across files.
    """
    y_all = np.asarray(absolute_load, dtype=float)
    mask = np.ones(len(y_all), dtype=bool) if is_normal is None else np.asarray(is_normal, dtype=bool)
    y = y_all[mask]
    if len(y) < 6:
        return DegradationTrend(0.0, 1.0, 1.0, None, threshold,
                                note="too few healthy cycles to estimate a trend")

    # Baseline = the healthy level early in the recording, so the index starts near 1.0.
    k = max(3, int(len(y) * baseline_frac))
    baseline = float(np.median(y[:k]))
    if baseline <= 0:
        return DegradationTrend(0.0, 1.0, 1.0, None, threshold, note="baseline unavailable")

    index = y / baseline
    slope, intercept = _theil_sen(index)
    current = float(np.median(index[-max(3, k):]))

    if current >= threshold:
        return DegradationTrend(slope, intercept, current, 0.0, threshold,
                                note="healthy-cycle load is already at the intervention level")
    if slope <= 1e-6:
        return DegradationTrend(slope, intercept, current, None, threshold,
                                note="no upward drift - the door's healthy baseline is stable")
    return DegradationTrend(slope, intercept, current, float((threshold - current) / slope),
                            threshold, note="linear extrapolation of the observed upward drift")


@dataclass
class SuppressionConfig:
    window: int = 5               # rolling window, in cycles
    min_confirmations: int = 2    # anomalies needed inside that window
    min_probability: float = 0.5
    require_sensor_agreement: bool = True
    speed_deficit_min: float = 0.02   # back-EMF must be at least this much below baseline


def suppress_noise(
    segments: pd.DataFrame,
    probability: np.ndarray,
    index: pd.Series,
    deficit: pd.Series,
    config: SuppressionConfig | None = None,
) -> pd.DataFrame:
    """Apply persistence and sensor-agreement gating to raw per-cycle flags."""
    cfg = config or SuppressionConfig()
    out = segments.reset_index(drop=True).copy()
    out["probability"] = np.asarray(probability, dtype=float)
    out["resistance_index"] = np.asarray(index, dtype=float)
    out["speed_deficit"] = np.asarray(deficit, dtype=float)
    out["raw_flag"] = out["probability"] >= cfg.min_probability

    # Sensor agreement: a genuine resistance fault loads the motor *and* slows the door.
    agree = (~cfg.require_sensor_agreement) | (out["speed_deficit"] >= cfg.speed_deficit_min)
    out["sensor_agreement"] = agree
    out["evidence"] = np.where(
        out["raw_flag"] & agree,
        "current and back-EMF agree",
        np.where(out["raw_flag"], "current only - check supply/sensor before the door", "-"),
    )

    # Persistence: count raw flags in a trailing window of the same operating state.
    confirmed = np.zeros(len(out), dtype=bool)
    persistence = np.zeros(len(out), dtype=int)
    for op in out["operation"].unique() if "operation" in out.columns else [None]:
        mask = np.ones(len(out), dtype=bool) if op is None else (out["operation"] == op).to_numpy()
        idx = np.flatnonzero(mask)
        flags = out.loc[mask, "raw_flag"].to_numpy()
        for k, pos in enumerate(idx):
            lo = max(0, k - cfg.window + 1)
            count = int(flags[lo : k + 1].sum())
            persistence[pos] = count
            confirmed[pos] = flags[k] and count >= cfg.min_confirmations
    out["persistence"] = persistence
    out["confirmed"] = confirmed & agree.to_numpy()
    out["severity"] = [severity_band(v) for v in out["resistance_index"]]
    out["suppressed"] = out["raw_flag"] & ~out["confirmed"]
    return out


def group_events(flagged: pd.DataFrame, max_gap_cycles: int = 3) -> pd.DataFrame:
    """Merge nearby confirmed cycles into single ranked maintenance events."""
    conf = flagged.index[flagged["confirmed"]].to_numpy()
    if len(conf) == 0:
        return pd.DataFrame(
            columns=["event_id", "start_time", "end_time", "n_cycles", "severity",
                     "peak_resistance_index", "mean_probability", "urgency"]
        )
    groups: list[list[int]] = [[conf[0]]]
    for i in conf[1:]:
        if i - groups[-1][-1] <= max_gap_cycles:
            groups[-1].append(i)
        else:
            groups.append([i])

    rows = []
    for n, g in enumerate(groups, start=1):
        sub = flagged.loc[g]
        peak = float(sub["resistance_index"].max())
        # Urgency blends how bad it is, how persistent it is, and how confident we are.
        urgency = float(
            peak * (1 + 0.1 * np.log1p(len(g))) * sub["probability"].mean()
        )
        rows.append(
            dict(
                event_id=f"event_{n:02d}",
                start_time=sub["start_time"].iloc[0],
                end_time=sub["end_time"].iloc[-1],
                n_cycles=len(g),
                severity=severity_band(peak),
                peak_resistance_index=peak,
                mean_probability=float(sub["probability"].mean()),
                urgency=urgency,
                evidence=sub["evidence"].mode().iat[0] if len(sub) else "-",
            )
        )
    return (
        pd.DataFrame(rows)
        .sort_values("urgency", ascending=False)
        .reset_index(drop=True)
    )


def explain_segment(row: pd.Series, top_features: pd.DataFrame | None = None) -> str:
    """One plain-language sentence a maintainer can act on."""
    idx = row.get("resistance_index", float("nan"))
    deficit = row.get("speed_deficit", 0.0)
    if not row.get("raw_flag", False):
        return (
            f"Normal {str(row.get('operation', '')).lower()} cycle - motor load "
            f"{idx:.2f}x this door's baseline."
        )
    parts = [
        f"Motor load {idx:.2f}x this door's own baseline "
        f"({'+' if idx >= 1 else ''}{(idx - 1) * 100:.0f}%)"
    ]
    if deficit > 0.01:
        parts.append(f"door ran {deficit * 100:.0f}% slower than baseline")
    else:
        parts.append("speed unchanged, so the extra current is not clearly mechanical")
    if not row.get("confirmed", False):
        parts.append(
            "not confirmed - "
            + ("isolated cycle, below the persistence threshold"
               if row.get("sensor_agreement", True)
               else "current and back-EMF disagree")
        )
    return "; ".join(parts) + "."
