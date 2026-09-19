"""
predict.py — Rail corrugation inference.

    python predict.py --input <folder of Test*.csv> --output rail_predictions.csv

Writes one row per input file:
    file_id     e.g. Test1.csv
    prediction  Normal / Side I / Side II

Optional:
    --model   path to the trained model (default: outputs/final/rail_model.joblib)
    --probs   also write per-class probabilities to <output>_probs.csv
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

import rail_model  # noqa: F401  (needed so joblib can unpickle RailModel)
from rail_model import build_features, file_sort_key

DEFAULT_MODEL = Path(__file__).resolve().parent / "outputs" / "final" / "rail_model.joblib"


def main():
    ap = argparse.ArgumentParser(description="Rail corrugation: Normal / Side I / Side II")
    ap.add_argument("--input", required=True, help="folder containing the CSV recordings")
    ap.add_argument("--output", required=True, help="path of the predictions CSV to write")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--probs", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.input)
    paths = sorted(in_dir.glob("*.csv"), key=file_sort_key)
    if not paths:
        sys.exit(f"No CSV files found in {in_dir}")
    if not Path(args.model).exists():
        sys.exit(f"Model not found: {args.model}  (run train_final.py first)")

    model = joblib.load(args.model)
    print(f"Model: {model.config}, {len(model.columns)} features, {len(model.members)} seeds")
    print(f"Extracting features for {len(paths)} files...")
    feats = build_features(paths, with_wl=(model.config == "REF+WL"))

    missing = [c for c in model.columns if c not in feats.columns]
    if missing:
        sys.exit(f"Feature mismatch: {len(missing)} model columns missing, e.g. {missing[:3]}")

    preds = model.predict(feats)
    out = pd.DataFrame({"file_id": feats["filename"], "prediction": preds})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    counts = out["prediction"].value_counts()
    print(f"Wrote {len(out)} predictions -> {args.output}")
    print(counts.to_string())
    n_fault = int(counts.drop("Normal", errors="ignore").sum())
    expected = round(len(out) * 38 / 272)
    if n_fault > 2 * expected + 2:
        print(f"WARNING: {n_fault} faults predicted vs ~{expected} expected from the training "
              f"ratio. Check for a train/test distribution shift before submitting.")

    if args.probs:
        p = model.predict_proba(feats)
        prob_path = Path(args.output).with_name(Path(args.output).stem + "_probs.csv")
        pd.DataFrame({"file_id": feats["filename"], "p_normal": p[:, 0],
                      "p_side_I": p[:, 1], "p_side_II": p[:, 2]}).to_csv(prob_path, index=False)
        print(f"Probabilities -> {prob_path}")


if __name__ == "__main__":
    main()
