"""
SHM (Structural Health Monitoring) — cumulative fatigue damage prediction.
Nebula X Hackathon 2026, Problem Statement 3.

FRAMEWORK-AGNOSTIC. No Streamlit, no Flask, no web framework of any kind. Import it into
whatever the combined app is built on, or run it from the command line.

DEPENDENCIES:  numpy, rainflow          (pandas only if you use predict_to_csv)
    pip install numpy rainflow

------------------------------------------------------------------------------------------
QUICK START
------------------------------------------------------------------------------------------
    from shm_model import predict_damage, predict_batch, interpret

    d = predict_damage("test01.csv")            # -> 0.032513666...
    rows = predict_batch(["test01.csv", "test02.csv"])
    verdict = interpret(d)                      # -> {"status": "good", "headline": ..., ...}

`predict_damage` accepts any of: a file path, an open file object, an uploaded-file object
(anything with .read()), or a numpy array of stress values. So it works the same whether your
app hands it a path on disk or an in-memory upload.

Command line (matches the --input/--output interface PS3 asks predict.py to expose):

    python shm_model.py --input ./Test --output shm_predictions.csv

------------------------------------------------------------------------------------------
WHAT THE MODEL IS
------------------------------------------------------------------------------------------
Physics, not a fitted black box. Each stress time series is rainflow-counted (ASTM E1049),
then Miner's linear damage rule is applied:

    D = sum_i n_i / N_i ,   N_i = C / sigma_i^m      (sigma_i = range_i / 2)
    => D = S(m) / C ,  where  S(m) = sum_i  n_i * sigma_i^m

Only THREE numbers are learned from the 64 training files: the S-N exponent m, the constant C,
and GAMMA for the damage-concentration correction described below.
There is no model file to ship, nothing to load, nothing to go stale — the entire trained state
is the handful of constants below.

m was chosen by cross-validation (a sharp, well-defined minimum at 5.0). C was fitted by
minimising MAPE directly, because MAPE is what the subsystem is scored on.

VALIDATED PERFORMANCE:  leave-one-out CV over the 64 training files gives MAPE = 2.203%,
i.e. a subsystem score of 0.9780 on the official metric max(0, 1 - MAPE). That is an
out-of-fold figure, not training-set performance: every fitted quantity, including the
correction's exponent, is re-chosen inside each fold.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import rainflow

__all__ = [
    "predict_damage", "predict_batch", "interpret", "predict_to_csv",
    "M_EXPONENT", "C_CONSTANT", "GAMMA", "OUTPUT_COLUMNS",
]

# ---------------------------------------------------------------------------------------
# The entire trained model.
# Every constant is carried at full precision deliberately. An earlier version quoted C to
# 6 significant figures, which shifted every prediction by a constant factor of ~3e-7 --
# harmless next to a 2.2% MAPE, but enough to make two implementations disagree in the last
# decimals, which wastes an afternoon when someone tries to reconcile them. Copy these values
# verbatim; do not round them.
# ---------------------------------------------------------------------------------------
M_EXPONENT = 5.0
C_CONSTANT = 731580760.0173708

# --- damage-concentration correction ------------------------------------------------------
# Residual analysis of the pure D = S(m)/C fit found one systematic effect and only one: records
# whose damage is carried by a single dominant rainflow cycle were under-predicted (Spearman
# -0.43 between pred/true and that cycle's share of the damage sum). That is exactly where a
# single power-law S-N curve is least trustworthy. The fit is therefore scaled by
#
#     exp( GAMMA * (top1 - TOP1_REF) ) ,   top1 = largest cycle's damage / total damage
#
# Leave-one-out MAPE 2.544% -> 2.203% (score 0.9746 -> 0.9780), with GAMMA re-chosen inside every
# fold. top1 is clipped to the range it was fitted over so a record more concentrated than
# anything in training is not extrapolated into; the clip binds on no training file.
GAMMA = 0.30
TOP1_REF = 0.167141164449289
TOP1_MIN = 0.048239955808822864
TOP1_MAX = 0.31186730097723686

# Competition output schema — one row per file.
OUTPUT_COLUMNS = ["file_id", "prediction"]

# Percentiles of the 64 labelled TRAINING damage values. The Info Kit states every training
# sample is a healthy operating condition, so these describe what NORMAL loading looks like
# on this line — they are a reference distribution, not damage thresholds.
BASELINE_P25 = 0.045889
BASELINE_MEDIAN = 0.098931
BASELINE_P75 = 0.387633
BASELINE_P95 = 0.780506
BASELINE_MAX = 0.928339


def _load_series(source) -> np.ndarray:
    """Accept a path, a file object, an upload object, or an array. Return 1-D stress values.

    Raises ValueError with an actionable message if the input is not an SHM recording — the
    single most common integration mistake is feeding it a Rail file or a labels file.
    """
    if isinstance(source, np.ndarray):
        data = source
    else:
        name = getattr(source, "name", str(source))
        try:
            data = np.loadtxt(source)
        except Exception:
            raise ValueError(
                f"'{os.path.basename(str(name))}' is not an SHM stress recording. SHM expects "
                f"one raw stress reading per line with NO header (test01.csv ... test16.csv, "
                f"~6 MB each). A file with column headers — a Rail recording, a labels file, or "
                f"a predictions file — will fail here."
            )
    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError(
            f"Expected a single column of stress values, got {data.shape[1]} columns. "
            f"A 129-column file is a Rail Corrugation recording, not SHM."
        )
    return data


def predict_damage(source) -> float:
    """Cumulative fatigue damage D for one recording.

    source: file path, file object, upload object, or 1-D numpy array of stress values.
    returns: float. Miner's rule defines failure at D = 1.0, so D is already a fraction
             of the total fatigue budget.
    """
    data = _load_series(source)
    cycles = list(rainflow.count_cycles(data))
    if not cycles:
        return 0.0
    ranges = np.array([c[0] for c in cycles])
    counts = np.array([c[1] for c in cycles])
    amplitudes = ranges / 2.0
    per_cycle = counts * np.power(amplitudes, M_EXPONENT)
    S = float(np.sum(per_cycle))
    if S <= 0:
        return 0.0
    top1 = float(np.max(per_cycle) / S)
    correction = np.exp(GAMMA * (min(max(top1, TOP1_MIN), TOP1_MAX) - TOP1_REF))
    return S * correction / C_CONSTANT


def predict_batch(sources) -> list[dict]:
    """Run several recordings. Returns a list of dicts with the competition columns plus
    operator-facing extras. Strip to OUTPUT_COLUMNS before writing the submission CSV."""
    rows = []
    for src in sources:
        name = os.path.basename(str(getattr(src, "name", src)))
        data = _load_series(src)
        d = predict_damage(data)
        rows.append({
            "file_id": name,
            "prediction": d,
            # extras — useful for a UI, NOT part of the submission schema
            "pct_of_fatigue_limit": d * 100.0,
            "segments_to_limit": (1.0 / d) if d > 0 else float("inf"),
            "vs_typical_segment": d / BASELINE_MEDIAN,
            "peak_stress": float(np.max(np.abs(data))) if len(data) else 0.0,
        })
    return rows


def interpret(damage: float) -> dict:
    """Turn a damage number into something a maintainer can act on.

    A raw "0.4409" means nothing to a depot. Two framings make it actionable, and both come
    from the given data rather than invented thresholds: what fraction of the fatigue budget
    this segment consumed, and where it sits against the healthy fleet baseline.

    Returns keys: status, band, headline, action, evidence (list of strings).
    `status` is one of good / warning / serious / critical. Pair it with an icon AND a written
    word in the UI — never colour alone, or it fails for colour-blind operators.
    """
    d = float(damage)
    pct = d * 100.0
    segs = (1.0 / d) if d > 0 else float("inf")
    ratio = d / BASELINE_MEDIAN

    if d >= BASELINE_MAX:
        status, band = "critical", "Beyond healthy baseline"
        headline = (f"Loading severity above anything in the healthy reference set "
                    f"({pct:.0f}% of fatigue budget)")
        action = ("Flag for engineering review. Confirm the measurement point is reading "
                  "correctly, then check the route section and load condition it was recorded "
                  "under before treating it as a real structural finding.")
    elif d >= BASELINE_P95:
        status, band = "serious", "Severe loading (top 5%)"
        headline = f"Severe loading — this segment consumed {pct:.0f}% of the fatigue budget"
        action = (f"Schedule an inspection of this measurement point. At this loading rate the "
                  f"Miner's limit is reached in about {segs:.1f} more equivalent segments.")
    elif d >= BASELINE_P75:
        status, band = "warning", "Heavy loading (top 25%)"
        headline = f"Heavy loading — {pct:.0f}% of the fatigue budget consumed in this segment"
        action = ("No immediate action. Keep this measurement point on the watch list and "
                  "re-check on the next download; repeated segments at this level shorten the "
                  "inspection interval.")
    elif d >= BASELINE_P25:
        status, band = "good", "Typical loading"
        headline = f"Normal loading — {pct:.1f}% of the fatigue budget consumed"
        action = "No action. Continue routine monitoring."
    else:
        status, band = "good", "Light loading (bottom 25%)"
        headline = f"Light loading — {pct:.1f}% of the fatigue budget consumed"
        action = "No action. Continue routine monitoring."

    evidence = [
        f"Cumulative damage D = {d:.4f} (Miner's rule; failure is defined at D = 1.0)",
        f"{ratio:.1f}x the median healthy segment (baseline median D = {BASELINE_MEDIAN:.3f})",
        f"At this loading rate, ~{segs:.1f} equivalent segments to reach the D = 1.0 limit",
        f"Healthy-baseline reference range: D = {BASELINE_P25:.3f} (25th pct) to "
        f"{BASELINE_P95:.3f} (95th pct), max observed {BASELINE_MAX:.3f}",
    ]
    return {"status": status, "band": band, "headline": headline,
            "action": action, "evidence": evidence}


def predict_to_csv(input_dir: str, output_path: str) -> str:
    """Run every .csv in input_dir and write the competition CSV. Needs pandas."""
    import pandas as pd
    files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not files:
        raise SystemExit(f"No .csv files found in {input_dir}")
    rows = predict_batch(files)
    df = pd.DataFrame(rows)[OUTPUT_COLUMNS]
    df.to_csv(output_path, index=False)
    print(f"Wrote {output_path}  ({len(df)} rows)")
    return output_path


def main():
    p = argparse.ArgumentParser(description="SHM cumulative fatigue damage prediction.")
    p.add_argument("--input", required=True,
                   help="Folder of SHM stress recordings (test01.csv ... test16.csv)")
    p.add_argument("--output", default="shm_predictions.csv",
                   help="Where to write the predictions CSV")
    a = p.parse_args()
    predict_to_csv(a.input, a.output)


if __name__ == "__main__":
    main()
