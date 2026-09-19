"""
train_final.py — train the submission model on all 272 training files.

Features are rebuilt from the raw CSVs with rail_model.build_features, the same
function predict.py uses, so train and test features cannot drift apart.
The feature table is cached; delete it to force a rebuild.

Pick --config from the step-9 run:
    REF+WL  if it beat REF by >= ~0.02 macro F1 (3+ repeats)
    REF     otherwise

Usage:
    python train_final.py --train-dir <...>/Train --labels <...>/Train_Labels.csv --config REF+WL
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from rail_model import CONFIGS, build_features, file_sort_key, train_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--config", choices=CONFIGS, required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--features-cache", default="outputs/final/train_features.csv")
    ap.add_argument("--out", default="outputs/final/rail_model.joblib")
    args = ap.parse_args()

    labels = pd.read_csv(args.labels)
    cache = Path(args.features_cache)
    if cache.exists():
        print(f"Loading cached training features: {cache}")
        feats = pd.read_csv(cache)
    else:
        paths = sorted((Path(args.train_dir) / f for f in labels["filename"]), key=file_sort_key)
        print(f"Building features for {len(paths)} training files...")
        feats = build_features(paths)
        cache.parent.mkdir(parents=True, exist_ok=True)
        feats.to_csv(cache, index=False)
        print(f"Cached -> {cache}")

    feats = feats.merge(labels, on="filename", how="inner", validate="one_to_one")
    assert len(feats) == len(labels), "some labelled files are missing features"

    model = train_model(feats, feats["label"].values, args.config, n_seeds=args.seeds)

    # Sanity only: training-set fit should be near-perfect. Not a performance estimate.
    acc = np.mean(np.array(model.predict(feats)) == feats["label"].values)
    print(f"Training-set accuracy (sanity check, not a score): {acc:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)
    print(f"Saved {args.out} | config={model.config} features={len(model.columns)} "
          f"seeds={len(model.members)}")


if __name__ == "__main__":
    main()
