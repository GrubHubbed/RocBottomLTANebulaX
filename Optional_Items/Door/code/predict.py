"""Inference: continuous stream in, submission CSV out.

    python -m src.door.predict --input data/raw/Door/Test.csv --output outputs/door_predictions.csv

The submission file carries exactly the three columns the Info Kit specifies
(``start_time``, ``end_time``, ``prediction``), with timestamps in the dataset's native
non-zero-padded format. A second, richer CSV (``*_detailed.csv``) is written alongside it
with probabilities, resistance index, severity and the noise-suppression verdict - that is
what the app displays and what a maintainer would actually read.

The same function powers both the CLI and the app, so the predictions submitted are
produced by exactly the code path demonstrated in the demo video.
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import (
    add_stream_relative_features,
    build_feature_table,
    build_feature_view,
)
from .health import (
    SuppressionConfig,
    degradation_trend,
    explain_segment,
    group_events,
    resistance_index,
    speed_deficit,
    suppress_noise,
)
from .io import load_stream
from .segment import segment_stream

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models/door_model.joblib"
SUBMISSION_COLUMNS = ["start_time", "end_time", "prediction"]


@dataclass
class PredictionResult:
    segments: pd.DataFrame        # rich per-cycle table
    events: pd.DataFrame          # grouped, urgency-ranked maintenance events
    trend: dict                   # degradation trend / lead time
    summary: dict                 # headline counts for the UI

    @property
    def submission(self) -> pd.DataFrame:
        return self.segments[SUBMISSION_COLUMNS].copy()


def load_bundle(model_path: str | Path = DEFAULT_MODEL) -> dict:
    """Load the trained bundle, failing loudly on a library-version mismatch.

    A scikit-learn model is a pickle of that version's internal classes, so loading one
    under a different version fails with a bare ``No module named '_loss'``. That is
    unreadable at 3am during a deployment, so it is caught and explained here.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run:  python -m src.door.train"
        )
    try:
        bundle = joblib.load(path)
    except (ModuleNotFoundError, AttributeError, ImportError) as exc:
        import sklearn

        raise RuntimeError(
            f"Could not load {path.name}: {exc}.\n"
            f"This is almost always a library-version mismatch between training and here "
            f"(running scikit-learn {sklearn.__version__}). Either install the pinned "
            f"versions in requirements.txt, or retrain with `python -m src.door.train`."
        ) from exc

    trained_with = bundle.get("training_metadata", {}).get("library_versions", {})
    if trained_with:
        import sklearn

        if trained_with.get("scikit-learn") != sklearn.__version__:
            warnings.warn(
                f"Model was trained with scikit-learn {trained_with['scikit-learn']} but "
                f"{sklearn.__version__} is installed. Predictions may differ; "
                f"requirements.txt pins the tested versions.",
                RuntimeWarning,
                stacklevel=2,
            )
    return bundle


#: How far above the trained healthy baseline a file's own median may sit before the
#: self-normalisation baseline is treated as unreliable. Normal cycles cluster within a few
#: percent of their baseline, and faults sit at ~2.2x, so a file whose median is 30% high
#: has faults in the middle of its distribution. Set generously: a differently calibrated
#: sensor also shifts this ratio, and a false trip costs only the fallback model.
BASELINE_SUSPECT_RATIO = 1.30


def _check_baseline(table: pd.DataFrame, bundle: dict) -> str | None:
    """Warn when a file's own median looks like it is resting on faulty cycles.

    The stream-relative view assumes a *minority* of cycles in a recording are faulty. When
    that fails the median becomes the fault, every cycle looks normal relative to it, and
    the model under-reports with no outward sign. This compares the file's own median
    against the absolute healthy baseline measured at training time.
    """
    baseline = bundle.get("absolute_baseline") or {}
    if not baseline or "cur_steady_mean" not in table.columns:
        return None
    is_open = table["is_open"] > 0.5
    for name, mask in (("open", is_open), ("close", ~is_open)):
        trained = baseline.get(name)
        if not trained or mask.sum() < 3:
            continue
        observed = float(table.loc[mask, "cur_steady_mean"].median())
        ratio = observed / trained
        if ratio >= BASELINE_SUSPECT_RATIO:
            return (
                f"This recording's typical {name} cycle draws {ratio:.2f}x the current of a "
                f"healthy cycle in training. Either most cycles in it are faulty, or this "
                f"door's sensor is calibrated differently. Judging each cycle against this "
                f"file's own median would hide the fault, so the absolute-units model was "
                f"used instead. Treat these results as needing a human check."
            )
    return None


def predict_stream(
    stream: pd.DataFrame,
    bundle: dict,
    suppression: SuppressionConfig | None = None,
) -> PredictionResult:
    """Segment a continuous stream, classify every cycle, and rank what needs attention."""
    segments = segment_stream(stream)
    if len(segments) == 0:
        empty = pd.DataFrame(columns=SUBMISSION_COLUMNS)
        return PredictionResult(empty, pd.DataFrame(), {}, {"n_segments": 0})

    table = add_stream_relative_features(build_feature_table(stream, segments))

    # A self-baseline needs enough cycles to be meaningful; below that, fall back to the
    # absolute-units model, which needs no baseline.
    min_segments = bundle.get("min_segments_for_self_norm", 8)
    baseline_warning = _check_baseline(table, bundle)
    # Two conditions force the absolute-units model: too few cycles to form a reliable
    # self-baseline, or a self-baseline that looks like it is sitting on faults.
    use_primary = len(segments) >= min_segments and baseline_warning is None
    view = bundle["primary_view"] if use_primary else bundle["fallback_view"]
    model = bundle["classifier"] if use_primary else bundle["fallback_classifier"]
    cols = bundle["feature_columns"] if use_primary else bundle["fallback_feature_columns"]

    X = build_feature_view(table, view).reindex(columns=cols, fill_value=0.0)
    proba = model.predict_proba(X)[:, 1]
    threshold = float(bundle.get("threshold", 0.5))
    labels = np.where(proba >= threshold, bundle["positive_label"], bundle["negative_label"])

    # Unsupervised cross-check (fast-monitoring mode). Higher = more unusual.
    anomaly_score = np.zeros(len(X))
    if use_primary and bundle.get("anomaly_detector") is not None:
        anomaly_score = -bundle["anomaly_detector"].decision_function(X)

    seg = segments.copy()
    seg["operation"] = np.where(table["is_open"].to_numpy() > 0.5, "Open", "Close")
    seg["prediction"] = labels
    idx = resistance_index(X if view != "absolute" else table)
    deficit = speed_deficit(X if view != "absolute" else table)

    flagged = suppress_noise(seg, proba, idx, deficit, suppression)
    flagged["anomaly_score"] = anomaly_score
    flagged["anomaly_flag"] = anomaly_score > -bundle.get("iso_reference", {}).get(
        "normal_score_min", 0.0
    )
    flagged["explanation"] = [explain_segment(r) for _, r in flagged.iterrows()]
    flagged["model_used"] = f"{bundle['model_name'] if use_primary else bundle['fallback_model_name']} ({view})"

    events = group_events(flagged)
    # The trend runs on absolute load, over healthy cycles only - see health.degradation_trend.
    absolute_load = table["cur_steady_mean"].to_numpy(dtype=float)
    flagged["absolute_load_mA"] = absolute_load
    trend = degradation_trend(
        pd.Series(absolute_load),
        is_normal=(flagged["prediction"] == bundle["negative_label"]).to_numpy(),
    ).as_dict()

    n_abnormal = int((flagged["prediction"] == bundle["positive_label"]).sum())
    summary = {
        "n_segments": int(len(flagged)),
        "n_abnormal": n_abnormal,
        "n_normal": int(len(flagged) - n_abnormal),
        "abnormal_rate": float(n_abnormal / len(flagged)),
        "n_confirmed": int(flagged["confirmed"].sum()),
        "n_suppressed": int(flagged["suppressed"].sum()),
        "n_events": int(len(events)),
        "worst_severity": (events["severity"].iat[0] if len(events) else "None"),
        "model_used": flagged["model_used"].iat[0],
        "threshold": threshold,
        "segmentation": flagged["split_by"].value_counts().to_dict(),
        "baseline_warning": baseline_warning,
        "stream_start": str(flagged["start_time"].iat[0]),
        "stream_end": str(flagged["end_time"].iat[-1]),
    }
    return PredictionResult(flagged, events, trend, summary)


def predict_file(
    input_csv: str | Path,
    output_csv: str | Path,
    model_path: str | Path = DEFAULT_MODEL,
    write_detailed: bool = True,
    history_csv: str | Path | None = None,
) -> PredictionResult:
    bundle = load_bundle(model_path)
    result = predict_stream(load_stream(input_csv), bundle)
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.submission.to_csv(out, index=False)
    if write_detailed:
        detail_cols = [
            "segment_id", "start_time", "end_time", "operation", "n_rows", "duration_s",
            "prediction", "probability", "resistance_index", "absolute_load_mA",
            "speed_deficit", "severity",
            "persistence", "confirmed", "suppressed", "sensor_agreement", "anomaly_score",
            "evidence", "explanation", "model_used",
        ]
        cols = [c for c in detail_cols if c in result.segments.columns]
        result.segments[cols].to_csv(out.with_name(out.stem + "_detailed.csv"), index=False)
    if history_csv is not None:
        append_history(result, Path(input_csv).name, history_csv)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict door cycles from a continuous stream.")
    ap.add_argument("--input", required=True, type=Path, help="continuous stream CSV (e.g. Test.csv)")
    ap.add_argument("--output", required=True, type=Path, help="submission CSV to write")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--no-detailed", action="store_true")
    ap.add_argument("--history", type=Path, default=ROOT / "outputs/door_history.csv",
                    help="append this run's summary to a long-term history file")
    args = ap.parse_args()

    result = predict_file(args.input, args.output, args.model, not args.no_detailed, args.history)
    print(json.dumps(result.summary, indent=2, default=str))
    print(f"\nWrote {len(result.submission)} segments -> {args.output}")
    if len(result.events):
        print("\nRanked maintenance events:")
        print(result.events.round(3).to_string(index=False))
    if result.trend.get("cycles_to_threshold") is not None:
        print(f"\nDegradation: {result.trend['note']} "
              f"(index {result.trend['current_index']:.2f}, "
              f"{result.trend['cycles_to_threshold']:.0f} cycles of margin)")



# --- long-term monitoring ------------------------------------------------------------
#
# A single recording can only show drift *inside* that recording. Real degradation is
# visible across days, so every run appends one summary row to a history file. Point the
# app or CLI at the same history file over successive recordings and the cross-file trend
# in absolute motor load becomes the long-horizon signal - the input a sequence model
# (e.g. an LSTM over per-day summaries) would consume later.

HISTORY_COLUMNS = [
    "run_at", "source", "stream_start", "stream_end", "n_segments", "n_abnormal",
    "abnormal_rate", "median_load_mA", "p90_load_mA", "healthy_index", "trend_slope",
    "cycles_to_threshold", "worst_severity",
]


def append_history(result: "PredictionResult", source: str, history_csv: str | Path) -> pd.DataFrame:
    """Append this run's summary to the long-term history file and return the full history."""
    seg = result.segments
    healthy = seg[seg["prediction"] == "Normal"]
    load = healthy["absolute_load_mA"] if "absolute_load_mA" in seg.columns else pd.Series([np.nan])
    row = {
        "run_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "source": source,
        "stream_start": result.summary.get("stream_start"),
        "stream_end": result.summary.get("stream_end"),
        "n_segments": result.summary.get("n_segments"),
        "n_abnormal": result.summary.get("n_abnormal"),
        "abnormal_rate": round(float(result.summary.get("abnormal_rate", 0.0)), 4),
        "median_load_mA": round(float(load.median()), 1) if len(load) else None,
        "p90_load_mA": round(float(load.quantile(0.9)), 1) if len(load) else None,
        "healthy_index": round(float(result.trend.get("current_index", float("nan"))), 3),
        "trend_slope": round(float(result.trend.get("slope_per_cycle", 0.0)), 6),
        "cycles_to_threshold": result.trend.get("cycles_to_threshold"),
        "worst_severity": result.summary.get("worst_severity"),
    }
    path = Path(history_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    history = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=HISTORY_COLUMNS)
    history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    history.to_csv(path, index=False)
    return history


if __name__ == "__main__":
    main()
