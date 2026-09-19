"""
Rail Corrugation — Step 9A: cache raw recordings as float32 .npy

Parsing 10,000 x 129 CSVs with pandas is the slowest part of every run.
This converts each file once (cleaned with step1's clean_rail_file) to
<cache>/<split>/<TrainN>.npy, shape (10000, 129), column order unchanged.

Usage:
    python step9_cache_npy.py --data-root <...>/Rail_Corrugation --cache outputs/cache
"""

import argparse
from pathlib import Path

import numpy as np

from step1_data_cleaning import load_rail_file


def cache_split(src_dir, dst_dir):
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.glob("*.csv"), key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0))
    done = 0
    for f in files:
        out = dst_dir / f"{f.stem}.npy"
        if out.exists():
            continue
        arr = load_rail_file(f).to_numpy(dtype=np.float32)
        np.save(out, arr)
        done += 1
        if done % 25 == 0:
            print(f"  {src_dir.name}: {done} new files cached", flush=True)
    print(f"{src_dir.name}: {len(files)} files ({done} newly cached) -> {dst_dir}")


def load_cached(cache_dir, split, filename):
    """filename like 'Train12.csv'. Returns (10000, 129) float32, memory-mapped."""
    return np.load(Path(cache_dir) / split / f"{Path(filename).stem}.npy", mmap_mode="r")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="folder containing Train/ and Test/")
    ap.add_argument("--cache", default="outputs/cache")
    args = ap.parse_args()
    for split in ("Train", "Test"):
        src = Path(args.data_root) / split
        if src.exists():
            cache_split(src, Path(args.cache) / split)


if __name__ == "__main__":
    main()
