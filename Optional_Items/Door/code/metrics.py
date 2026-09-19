"""The official Door metric: IoU-weighted F1 (Info Kit section 4).

Implemented exactly as specified so local validation numbers are directly comparable to
the judges' ``judge_leaderboard.py`` output:

* a prediction may only match a true segment carrying the **same** label;
* candidate pairs need IoU > 0;
* matching is greedy, highest IoU first, one-to-one;
* each match earns its own IoU as credit, not a flat 1.0.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .io import parse_datetime, to_seconds


def interval_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    intersection = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = (a_end - a_start) + (b_end - b_start) - intersection
    return intersection / union if union > 0 else 0.0


@dataclass
class DoorScore:
    score: float
    soft_recall: float
    soft_precision: float
    n_true: int
    n_pred: int
    n_matched: int
    iou_sum: float
    mean_matched_iou: float

    def as_dict(self) -> dict:
        return asdict(self)


def _as_intervals(df: pd.DataFrame, start_col: str, end_col: str, label_col: str):
    start = df[start_col]
    end = df[end_col]
    if not np.issubdtype(np.asarray(start).dtype, np.number):
        start = parse_datetime(start.astype(str)) if start.dtype == object else start
        end = parse_datetime(end.astype(str)) if end.dtype == object else end
        start, end = to_seconds(start), to_seconds(end)
    else:
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    return np.asarray(start), np.asarray(end), df[label_col].to_numpy()


def iou_weighted_f1(
    truth: pd.DataFrame,
    pred: pd.DataFrame,
    truth_cols=("start_time", "end_time", "status"),
    pred_cols=("start_time", "end_time", "prediction"),
) -> DoorScore:
    """Score predicted segments against ground truth."""
    ts, te, tl = _as_intervals(truth, *truth_cols)
    ps, pe, pl = _as_intervals(pred, *pred_cols)
    n_true, n_pred = len(ts), len(ps)

    candidates = []
    for i in range(n_true):
        for j in range(n_pred):
            if tl[i] != pl[j]:
                continue  # wrong label cannot match at all
            iou = interval_iou(ts[i], te[i], ps[j], pe[j])
            if iou > 0:
                candidates.append((iou, i, j))

    candidates.sort(key=lambda c: -c[0])
    used_t: set[int] = set()
    used_p: set[int] = set()
    iou_sum = 0.0
    matched = 0
    for iou, i, j in candidates:
        if i in used_t or j in used_p:
            continue
        used_t.add(i)
        used_p.add(j)
        iou_sum += iou
        matched += 1

    soft_recall = iou_sum / n_true if n_true else 0.0
    soft_precision = iou_sum / n_pred if n_pred else 0.0
    denom = soft_recall + soft_precision
    score = (2 * soft_recall * soft_precision / denom) if denom > 0 else 0.0

    return DoorScore(
        score=score,
        soft_recall=soft_recall,
        soft_precision=soft_precision,
        n_true=n_true,
        n_pred=n_pred,
        n_matched=matched,
        iou_sum=iou_sum,
        mean_matched_iou=(iou_sum / matched) if matched else 0.0,
    )
