"""Door subsystem — abnormal opening/closing resistance detection.

The public API an orchestration layer needs is re-exported here, so the whole integration
is::

    from door import load_stream, load_bundle, predict_stream

    bundle = load_bundle("models/door_model.joblib")   # load once, reuse
    result = predict_stream(load_stream("stream.csv"), bundle)

    result.submission   # DataFrame: start_time, end_time, prediction
    result.segments     # DataFrame: the above + probability, severity, explanation, ...
    result.events       # DataFrame: grouped maintenance events, ranked by urgency
    result.summary      # dict: headline counts, model used, any baseline warning
    result.trend        # dict: degradation trend and remaining margin

See README.md for the field-by-field description and the integration notes.
"""

from .features import build_features, segment_features
from .health import SuppressionConfig, degradation_trend, group_events
from .io import format_datetime, load_stream, parse_datetime
from .metrics import interval_iou, iou_weighted_f1
from .predict import (
    PredictionResult,
    append_history,
    load_bundle,
    predict_file,
    predict_stream,
)
from .segment import segment_stream

__version__ = "1.0.0"

__all__ = [
    # loading
    "load_stream",
    "parse_datetime",
    "format_datetime",
    # the main entry points
    "load_bundle",
    "predict_stream",
    "predict_file",
    "PredictionResult",
    # pieces, if you need them separately
    "segment_stream",
    "build_features",
    "segment_features",
    "SuppressionConfig",
    "group_events",
    "degradation_trend",
    "append_history",
    # scoring
    "iou_weighted_f1",
    "interval_iou",
    "__version__",
]
