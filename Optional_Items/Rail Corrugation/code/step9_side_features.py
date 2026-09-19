"""
Rail Corrugation — Step 9B: per-side feature tables

Builds, from the .npy cache:

  side_table.csv   one row per (file, side)  -> 2 rows per file
      own_*   this side's aggregated features
      other_* the opposite side's aggregated features
      d_*     own - other  (for log features this IS the log-ratio)
      car_*   per-car own-minus-other statistics (max, min, median, #cars own>other)
      own_xc_* / other_xc_*   cross-axle (same-rail) correlation features
      speed_kmh, is_side_I, wl_valid, filename, side, label (side-level: 1 = corrugated)

  file_table.csv   one row per file, same features pivoted into the
      sideI_ / sideII_ / _I_minus_II / _car_diff_* naming used by step 8,
      so the 3-class model + mirror augmentation can run on identical features.

Per-channel features (vib and shock, 128 channels):
    fb_<lo>_<hi>   log10 band power, fixed Hz bands (as step 3)
    wl_<lo>_<hi>   log10 band power, wavelength bands in mm (lambda = v / f)
    tonality       log10(peak / median PSD) within 20-600 mm  (periodic wear -> narrow peak)
    peakwl         wavelength (mm) of that peak
    rms            log10 RMS
    kurt           kurtosis
    crest          crest factor (max|x| / rms)

Side aggregation over the 32 sensors: mean, max, median, second-largest.
    median resists one bad sensor; second-largest vs max separates
    "several axle boxes see it" (rail) from "one axle box sees it" (wheel).

Cross-axle correlation (vib only):
    Leading and trailing wheels on the same bogie and side roll over the same
    rail a fixed distance L apart (the axle spacing), i.e. lag = L / v.
    For each assumed bogie pair, the band-passed signals are cross-correlated;
    features are the peak correlation inside the lag window for
    L in AXLE_L_RANGE, minus the peak in a control window (L in CONTROL_L_RANGE),
    because a purely tonal signal correlates at every lag.
    Run with --diagnose-pairs first to check the assumed pairing: true bogie
    pairs should show a tight, plausible implied spacing (~2-3 m) across files.

Stationary files (speed < 1 m/s): wavelength and xcorr features are 0, wl_valid = 0.

Usage:
    python step9_side_features.py --cache outputs/cache --labels <...>/Train_Labels.csv \
        --out outputs/step9
    python step9_side_features.py --cache outputs/cache --labels <...> --diagnose-pairs
    python step9_side_features.py --cache outputs/cache --split Test --out outputs/step9   (no labels)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis

from step3_feature_engineering import extract_speed_features
from step9_cache_npy import load_cached

FS = 10_000
N_CARS = 8
NPERSEG = 2048
F_MIN_HZ = 10.0
MIN_SPEED_MPS = 1.0

FB_BANDS = [(0, 250), (250, 500), (500, 750), (750, 1000), (1000, 1500), (1500, 2000),
            (2000, 2500), (2500, 3000), (3000, 3500), (3500, 4000), (4000, 4500), (4500, 5000)]
WL_BANDS_MM = [(20, 40), (40, 63), (63, 100), (100, 160), (160, 250), (250, 400), (400, 600)]
WL_RANGE_MM = (20, 600)

SIDE_POS = {"I": [1, 3, 5, 7], "II": [2, 4, 6, 8]}
# ASSUMED bogie pairs (same side, same bogie). Verify with --diagnose-pairs.
BOGIE_PAIRS = {"I": [(1, 3), (5, 7)], "II": [(2, 4), (6, 8)]}
AXLE_L_RANGE = (1.5, 3.5)       # m, plausible axle spacing within a bogie
CONTROL_L_RANGE = (0.4, 1.2)    # m, control lags (no physical reason to correlate)
XC_MAX_HZ = 4000.0

AGGS = ("mean", "max", "median", "second")


def feat_names():
    return ([f"fb_{a}_{b}" for a, b in FB_BANDS] + [f"wl_{a}_{b}" for a, b in WL_BANDS_MM]
            + ["tonality", "peakwl", "rms", "kurt", "crest"])


FEATS = feat_names()


def col_index(car, pos, ch):
    """Column index in the raw array (col 0 = speed)."""
    return 1 + ((car - 1) * 8 + (pos - 1)) * 2 + (0 if ch == "vib" else 1)


# ============================================================
# PER-CHANNEL FEATURES
# ============================================================

def channel_features(sig, speed):
    """sig: (n_ch, N). Returns (n_ch, len(FEATS))."""
    n_ch = sig.shape[0]
    out = np.zeros((n_ch, len(FEATS)))
    freqs, psd = welch(sig, fs=FS, nperseg=NPERSEG, axis=-1)
    df = freqs[1] - freqs[0]
    k = 0
    for a, b in FB_BANDS:
        m = (freqs >= a) & (freqs < b)
        out[:, k] = np.log10(psd[:, m].sum(axis=1) * df + 1e-12); k += 1

    ok = freqs >= F_MIN_HZ
    if speed >= MIN_SPEED_MPS:
        wl = 1000.0 * speed / freqs[ok]
        p = psd[:, ok]
        for a, b in WL_BANDS_MM:
            m = (wl >= a) & (wl < b)
            out[:, k] = np.log10(p[:, m].sum(axis=1) * df + 1e-12) if m.any() else 0.0; k += 1
        m = (wl >= WL_RANGE_MM[0]) & (wl < WL_RANGE_MM[1])
        pr = p[:, m]
        if pr.shape[1] >= 3:
            j = np.argmax(pr, axis=1)
            out[:, k] = np.log10(pr[np.arange(n_ch), j] / (np.median(pr, axis=1) + 1e-15) + 1e-12)
            out[:, k + 1] = wl[m][j]
        k += 2
    else:
        k += len(WL_BANDS_MM) + 2

    rms = np.sqrt(np.mean(sig ** 2, axis=1))
    out[:, k] = np.log10(rms + 1e-12)
    out[:, k + 1] = kurtosis(sig, axis=1)
    out[:, k + 2] = np.max(np.abs(sig), axis=1) / (rms + 1e-12)
    return out


# ============================================================
# CROSS-AXLE CORRELATION
# ============================================================

def bandpass_fft(x, speed):
    """Keep only the corrugation wavelength range (20-600 mm) at this speed."""
    X = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(len(x), 1 / FS)
    lo = speed / (WL_RANGE_MM[1] / 1000)
    hi = min(speed / (WL_RANGE_MM[0] / 1000), XC_MAX_HZ)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, n=len(x))


def xcorr_normalized(a, b):
    """Linear (zero-padded) normalized cross-correlation, lags 0..N-1 of b after a."""
    n = len(a)
    A = np.fft.rfft(a, 2 * n)
    B = np.fft.rfft(b, 2 * n)
    c = np.fft.irfft(np.conj(A) * B, 2 * n)[:n]      # c[k] = sum a[t] b[t+k]
    overlap = (n - np.arange(n)) / n                  # undo the shrinking-overlap bias,
    overlap = np.maximum(overlap, 0.25)               # capped so tiny overlaps don't explode
    return c / overlap / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


def window_peak(c, speed, L_range):
    lo = int(np.floor(L_range[0] / speed * FS))
    hi = int(np.ceil(L_range[1] / speed * FS))
    hi = min(hi, len(c) - 1)
    if lo >= hi:
        return 0.0, 0.0
    seg = np.abs(c[lo:hi + 1])
    j = int(np.argmax(seg))
    return float(seg[j]), (lo + j) / FS * speed      # (peak, implied spacing m)


def pair_xc(raw, car, p1, p2, speed):
    a = bandpass_fft(np.asarray(raw[:, col_index(car, p1, "vib")], float), speed)
    b = bandpass_fft(np.asarray(raw[:, col_index(car, p2, "vib")], float), speed)
    # direction of travel unknown: check both orders, keep the stronger
    best = (0.0, 0.0, 0.0)
    for x, y in ((a, b), (b, a)):
        c = xcorr_normalized(x, y)
        pk, L = window_peak(c, speed, AXLE_L_RANGE)
        ctl, _ = window_peak(c, speed, CONTROL_L_RANGE)
        if pk > best[0]:
            best = (pk, L, pk - ctl)
    return best


def side_xc_features(raw, side, speed):
    if speed < MIN_SPEED_MPS:
        return dict(xc_peak_mean=0.0, xc_peak_max=0.0, xc_excess_mean=0.0,
                    xc_excess_max=0.0, xc_L_std=0.0)
    res = [pair_xc(raw, car, p1, p2, speed)
           for car in range(1, N_CARS + 1) for p1, p2 in BOGIE_PAIRS[side]]
    pk, L, ex = map(np.array, zip(*res))
    return dict(xc_peak_mean=pk.mean(), xc_peak_max=pk.max(), xc_excess_mean=ex.mean(),
                xc_excess_max=ex.max(), xc_L_std=L.std())


# ============================================================
# ONE FILE -> TWO SIDE ROWS
# ============================================================

def agg(vals):
    """vals: (n_sensors, n_feat) -> dict agg -> (n_feat,)"""
    s = np.sort(vals, axis=0)
    return {"mean": vals.mean(0), "max": s[-1], "median": np.median(vals, 0), "second": s[-2]}


def file_side_rows(raw):
    speed = extract_speed_features(np.asarray(raw[:, 0]))["estimated_speed_mps"]
    sig = np.asarray(raw[:, 1:], dtype=float).T                 # (128, N)
    F = channel_features(sig, speed)                             # (128, nf)
    F = F.reshape(N_CARS, 8, 2, len(FEATS))                       # car, pos, ch(vib/shock), feat

    per_side = {}
    for s, pos in SIDE_POS.items():
        idx = [p - 1 for p in pos]
        per_side[s] = {
            ch: F[:, idx, ci, :] for ci, ch in enumerate(("vib", "shock"))   # (8 cars, 4, nf)
        }
    xc = {s: side_xc_features(raw, s, speed) for s in ("I", "II")}

    rows = []
    for s, o in (("I", "II"), ("II", "I")):
        r = {"side": s, "is_side_I": float(s == "I"), "speed_kmh": speed * 3.6,
             "wl_valid": float(speed >= MIN_SPEED_MPS)}
        for ch in ("vib", "shock"):
            own_all = per_side[s][ch].reshape(-1, len(FEATS))
            oth_all = per_side[o][ch].reshape(-1, len(FEATS))
            ao, at = agg(own_all), agg(oth_all)
            for a in AGGS:
                for i, f in enumerate(FEATS):
                    r[f"own_{ch}_{f}_{a}"] = ao[a][i]
                    r[f"other_{ch}_{f}_{a}"] = at[a][i]
                    r[f"d_{ch}_{f}_{a}"] = ao[a][i] - at[a][i]
            car_diff = per_side[s][ch].mean(1) - per_side[o][ch].mean(1)       # (8, nf)
            for i, f in enumerate(FEATS):
                cd = car_diff[:, i]
                r[f"car_{ch}_{f}_max"] = cd.max()
                r[f"car_{ch}_{f}_min"] = cd.min()
                r[f"car_{ch}_{f}_median"] = np.median(cd)
                r[f"car_{ch}_{f}_npos"] = float((cd > 0).sum())
        for k, v in xc[s].items():
            r[f"own_{k}"] = v
        for k, v in xc[o].items():
            r[f"other_{k}"] = v
        rows.append(r)
    return rows


# ============================================================
# FILE-LEVEL PIVOT (for the 3-class comparison)
# ============================================================

def to_file_table(side_df):
    """Rename side-I rows into step-8 mirror-compatible names."""
    I = side_df[side_df.side == "I"].set_index("filename")
    II = side_df[side_df.side == "II"].set_index("filename")
    out = {}
    for c in I.columns:
        if c.startswith("own_"):
            out[f"sideI_{c[4:]}"] = I[c]
            out[f"sideII_{c[4:]}"] = II[c]
        elif c.startswith("d_"):
            out[f"{c[2:]}_I_minus_II"] = I[c]
        elif c.startswith("car_") and c.endswith("_max"):
            out[f"{c[:-4]}_car_diff_max"] = I[c]
        elif c.startswith("car_") and c.endswith("_min"):
            out[f"{c[:-4]}_car_diff_min"] = I[c]
        elif c.startswith("car_") and c.endswith("_median"):
            out[f"{c[:-7]}_car_diff_median"] = I[c]
        elif c.startswith("car_") and c.endswith("_npos"):
            out[f"{c[:-5]}_cars_sideI_dominant_count"] = I[c]
    out["speed_kmh"] = I["speed_kmh"]
    out["estimated_speed_mps"] = I["speed_kmh"] / 3.6
    out["wl_valid"] = I["wl_valid"]
    return pd.DataFrame(out, index=I.index).reset_index()


# ============================================================
# DIAGNOSTIC: which positions are bogie-mates?
# ============================================================

def diagnose_pairs(cache, split, labels, n_files=38):
    """On FAULT files only, and only on the corrugated side, report for every
    same-side position pair within a car the implied spacing at the xcorr peak
    (window 0.5-20 m). True bogie-mates should cluster tightly around one
    plausible value (~2-3 m); other pairs should be weaker or scatter."""
    global AXLE_L_RANGE
    saved = AXLE_L_RANGE
    AXLE_L_RANGE = (0.5, 20.0)
    rows = []
    faults = labels[labels["label"] != "Normal"].head(n_files)
    for fn, lab in zip(faults["filename"], faults["label"]):
        raw = load_cached(cache, split, fn)
        v = extract_speed_features(np.asarray(raw[:, 0]))["estimated_speed_mps"]
        if v < 5:
            continue
        for side, pos in SIDE_POS.items():
            if lab != f"Side {side}":
                continue
            for i in range(len(pos)):
                for j in range(i + 1, len(pos)):
                    for car in (1, 4, 8):
                        pk, L, _ = pair_xc(raw, car, pos[i], pos[j], v)
                        rows.append(dict(pair=f"{pos[i]}-{pos[j]}", side=side, peak=pk, L=L))
    AXLE_L_RANGE = saved
    d = pd.DataFrame(rows)
    summ = d.groupby(["side", "pair"]).agg(peak_median=("peak", "median"),
                                          L_median=("L", "median"),
                                          L_iqr=("L", lambda x: x.quantile(.75) - x.quantile(.25)))
    print(summ.round(3).to_string())
    print("\nLook for pairs with high peak, L_median ~2-3 m and small L_iqr. "
          "Edit BOGIE_PAIRS if they are not (1,3),(5,7),(2,4),(6,8).")


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="outputs/cache")
    ap.add_argument("--split", default="Train")
    ap.add_argument("--labels", default="", help="Train_Labels.csv (omit for Test)")
    ap.add_argument("--out", default="outputs/step9")
    ap.add_argument("--diagnose-pairs", action="store_true")
    args = ap.parse_args()

    if args.labels:
        labels = pd.read_csv(args.labels)
    else:
        files = sorted(Path(args.cache, args.split).glob("*.npy"),
                       key=lambda p: int("".join(filter(str.isdigit, p.stem))))
        labels = pd.DataFrame({"filename": [f"{p.stem}.csv" for p in files]})

    if args.diagnose_pairs:
        if "label" not in labels:
            raise SystemExit("--diagnose-pairs needs --labels (it uses fault files only)")
        diagnose_pairs(args.cache, args.split, labels)
        return

    rows = []
    for i, fn in enumerate(labels["filename"], 1):
        for r in file_side_rows(load_cached(args.cache, args.split, fn)):
            r["filename"] = fn
            rows.append(r)
        if i % 25 == 0:
            print(f"  {i}/{len(labels)}", flush=True)
    side_df = pd.DataFrame(rows)

    file_df = to_file_table(side_df)
    if "label" in labels:
        lab = labels.set_index("filename")["label"]
        side_df["file_label"] = side_df["filename"].map(lab)
        side_df["label"] = (side_df["file_label"] == "Side " + side_df["side"]).astype(int)
        file_df = file_df.assign(label=file_df["filename"].map(lab))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tag = args.split.lower()
    side_df.to_csv(out / f"side_table_{tag}.csv", index=False)
    file_df.to_csv(out / f"file_table_{tag}.csv", index=False)
    print(f"side table {side_df.shape}, file table {file_df.shape} -> {out}")


if __name__ == "__main__":
    main()
