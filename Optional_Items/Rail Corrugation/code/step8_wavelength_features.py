"""
Rail Corrugation — Step 8B: Wavelength-domain features

Why:
    Corrugation is a fixed spatial wavelength on the rail. A wheel rolling at
    speed v over wavelength λ excites vibration at f = v / λ. Fixed-Hz bands
    (0-250, 250-500, ...) therefore describe a DIFFERENT set of wavelengths
    for every speed. Side I training files cluster at 45-49 km/h; the files
    the model misses (Train121 @ 35 km/h, Train180/185 @ 66 km/h) are the
    speed outliers. Re-binning the spectrum by wavelength makes a given
    corrugation land in the same feature regardless of speed.

What it computes, per vibration channel (64), from the Welch PSD:
    - log band power in wavelength bands (mm):
        WL_BANDS = 20-40, 40-63, 63-100, 100-160, 160-250, 250-400
      (roughly half-octave; covers "a few cm to dozens of cm")
    - tonality: peak PSD / median PSD within 20-400 mm.
      Corrugation is periodic -> a narrow spectral peak. Rough/ballast noise is
      broadband. This ratio is dimensionless, so it is insensitive to both speed
      and overall vibration level.
    - peak wavelength (mm) of the strongest component in 20-400 mm.

Aggregated to (all names follow the sideI_/sideII_/I_minus_II convention so
step8_validation.build_mirror_plan mirrors them correctly):
    wl_sideI_<feat>_mean / _max          across the 32 Side I vib sensors
    wl_sideII_<feat>_mean / _max
    wl_<feat>_mean_I_minus_II            (log power & tonality)
    wl_inv_<feat>_car_diff_max / _min    max / min over cars of per-car I - II
                                         (catches single-car localised faults)

Stationary files (speed < 1 m/s) have no defined wavelength; all features are 0.

Usage:
    python step8_wavelength_features.py --train-dir <...>/Train \
        --labels <...>/Train_Labels.csv --out outputs/step8/wavelength_features.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

from step1_data_cleaning import load_rail_file, N_CARS, SAMPLING_RATE
from step3_feature_engineering import extract_speed_features


WL_BANDS_MM = [(20, 40), (40, 63), (63, 100), (100, 160), (160, 250), (250, 400)]
WL_RANGE_MM = (20, 400)
MIN_SPEED_MPS = 1.0
NPERSEG = 2048            # ~4.9 Hz resolution, same as step 3
F_MIN_HZ = 10.0           # ignore DC / very low-frequency drift

SIDE_POS = {"I": [1, 3, 5, 7], "II": [2, 4, 6, 8]}


def channel_wavelength_features(psd, freqs, speed_mps):
    """Features for one channel. freqs/psd from welch; returns dict."""
    ok = freqs >= F_MIN_HZ
    f, p = freqs[ok], psd[ok]
    wl_mm = 1000.0 * speed_mps / f          # wavelength of each frequency bin

    out = {}
    for lo, hi in WL_BANDS_MM:
        m = (wl_mm >= lo) & (wl_mm < hi)
        # integrate over frequency (bins are uniform in f)
        bp = np.trapezoid(p[m], f[m]) if m.sum() >= 2 else (p[m].sum() * (f[1] - f[0]) if m.any() else 0.0)
        out[f"wlpow_{lo}_{hi}"] = np.log10(bp + 1e-12)

    m = (wl_mm >= WL_RANGE_MM[0]) & (wl_mm < WL_RANGE_MM[1])
    if m.sum() >= 3:
        pr = p[m]
        k = int(np.argmax(pr))
        out["tonality"] = np.log10(pr[k] / (np.median(pr) + 1e-15) + 1e-12)
        out["peakwl"] = wl_mm[m][k]
    else:
        out["tonality"] = 0.0
        out["peakwl"] = 0.0
    return out


def file_wavelength_features(df):
    """df = cleaned recording from load_rail_file. Returns one dict."""
    speed = extract_speed_features(df["speed"].values)["estimated_speed_mps"]
    feats = {}

    per = {}  # (side, car) -> list of channel dicts
    if speed >= MIN_SPEED_MPS:
        cols = [f"car{c}_pos{p}_vib" for c in range(1, N_CARS + 1) for p in range(1, 9)]
        freqs, psd = welch(df[cols].to_numpy().T, fs=SAMPLING_RATE, nperseg=NPERSEG, axis=-1)
        for i, col in enumerate(cols):
            car = int(col.split("_")[0][3:])
            pos = int(col.split("_")[1][3:])
            side = "I" if pos in SIDE_POS["I"] else "II"
            per.setdefault((side, car), []).append(
                channel_wavelength_features(psd[i], freqs, speed))

    names = [f"wlpow_{lo}_{hi}" for lo, hi in WL_BANDS_MM] + ["tonality", "peakwl"]
    diff_names = [n for n in names if n != "peakwl"]   # peak wavelength has no meaningful I-II diff

    if not per:
        for s in ("I", "II"):
            for n in names:
                feats[f"wl_side{s}_{n}_mean"] = 0.0
                feats[f"wl_side{s}_{n}_max"] = 0.0
        for n in diff_names:
            feats[f"wl_{n}_mean_I_minus_II"] = 0.0
            feats[f"wl_inv_{n}_car_diff_max"] = 0.0
            feats[f"wl_inv_{n}_car_diff_min"] = 0.0
        feats["wl_valid"] = 0.0
        return feats

    side_arr = {}
    for s in ("I", "II"):
        rows = [d for car in range(1, N_CARS + 1) for d in per[(s, car)]]
        tab = pd.DataFrame(rows)
        side_arr[s] = tab
        for n in names:
            feats[f"wl_side{s}_{n}_mean"] = float(tab[n].mean())
            feats[f"wl_side{s}_{n}_max"] = float(tab[n].max())

    for n in diff_names:
        feats[f"wl_{n}_mean_I_minus_II"] = feats[f"wl_sideI_{n}_mean"] - feats[f"wl_sideII_{n}_mean"]
        car_diffs = [
            np.mean([d[n] for d in per[("I", c)]]) - np.mean([d[n] for d in per[("II", c)]])
            for c in range(1, N_CARS + 1)
        ]
        feats[f"wl_inv_{n}_car_diff_max"] = float(np.max(car_diffs))
        feats[f"wl_inv_{n}_car_diff_min"] = float(np.min(car_diffs))

    feats["wl_valid"] = 1.0
    return feats


def build_wavelength_dataset(data_dir, filenames):
    rows = []
    for i, fn in enumerate(filenames, 1):
        df = load_rail_file(Path(data_dir) / fn)
        rows.append({"filename": fn, **file_wavelength_features(df)})
        if i % 25 == 0:
            print(f"  {i}/{len(filenames)}", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="outputs/step8/wavelength_features.csv")
    args = ap.parse_args()

    labels = pd.read_csv(args.labels)
    out = build_wavelength_dataset(args.train_dir, labels["filename"].tolist())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Saved {out.shape[1] - 1} wavelength features for {len(out)} files -> {args.out}")


if __name__ == "__main__":
    main()
