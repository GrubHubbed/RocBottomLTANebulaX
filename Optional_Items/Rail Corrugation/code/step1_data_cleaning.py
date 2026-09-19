"""
Rail Corrugation — Step 1: Load, clean, and sanity-check the data.
"""

from pathlib import Path
from unicodedata import numeric

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# DATASET CONSTANTS
# ============================================================

N_CARS = 8
N_POSITIONS = 8

SAMPLING_RATE = 10_000
DURATION_SECONDS = 1

EXPECTED_ROWS = SAMPLING_RATE * DURATION_SECONDS

SIDE_I_POSITIONS = [1, 3, 5, 7]
SIDE_II_POSITIONS = [2, 4, 6, 8]

EXPECTED_COLS = (
    1
    + N_CARS * N_POSITIONS * 2
)

EXPECTED_LABEL_COUNTS = {
    "Normal": 234,
    "Side I": 14,
    "Side II": 24
}


# ============================================================
# COLUMN NAMES
# ============================================================

def build_column_names():

    cols = ["speed"]

    for car in range(1, N_CARS + 1):

        for pos in range(1, N_POSITIONS + 1):

            cols.append(
                f"car{car}_pos{pos}_vib"
            )

            cols.append(
                f"car{car}_pos{pos}_shock"
            )

    assert len(cols) == EXPECTED_COLS

    return cols


COLUMN_NAMES = build_column_names()


# ============================================================
# DATA CLEANING
# ============================================================

def clean_rail_file(df, verbose=False):
    """
    Conservatively clean one recording.

    We fix:
      - non-numeric values
      - infinity
      - missing values

    We detect but DO NOT remove:
      - constant channels
      - extreme vibration/shock values
    """

    df = df.copy()

    # Convert everything to numbers.
    # Anything invalid becomes NaN.
    df = df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Replace infinity with NaN.
    n_inf = int(
        np.isinf(df.to_numpy()).sum()
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    missing_before = int(
        df.isna().sum().sum()
    )

    # Estimate missing samples using surrounding samples.
    df = df.interpolate(
        method="linear",
        axis=0,
        limit_direction="both"
    )

    df = df.ffill().bfill()

    missing_after = int(
        df.isna().sum().sum()
    )

    # Detect sensors that never change.
    constant_cols = [
        col
        for col in df.columns
        if df[col].nunique(dropna=True) <= 1
    ]

    if verbose:

        print("Cleaning report")
        print("----------------")

        print(
            f"Infinite values found: {n_inf}"
        )

        print(
            f"Missing values before: {missing_before}"
        )

        print(
            f"Missing values after: {missing_after}"
        )

        print(
            f"Constant channels: {len(constant_cols)}"
        )

    return df


# ============================================================
# FILE LOADING
# ============================================================

def load_rail_file(
    filepath,
    has_header="auto",
    clean=True
):

    filepath = Path(filepath)

    if has_header == "auto":

        first_row = pd.read_csv(
            filepath,
            nrows=1,
            header=None
        )

        try:

            first_row.astype(float)
            has_header = False

        except (ValueError, TypeError):

            has_header = True

    raw = pd.read_csv(
        filepath,
        header=0 if has_header else None
    )

    if raw.shape[1] != EXPECTED_COLS:

        raise ValueError(
            f"{filepath.name}: expected "
            f"{EXPECTED_COLS} columns, "
            f"got {raw.shape[1]}"
        )

    raw.columns = COLUMN_NAMES

    if clean:
        raw = clean_rail_file(raw)

    return raw


def load_labels(labels_path):

    labels = pd.read_csv(labels_path)

    expected_cols = {
        "filename",
        "label"
    }

    if not expected_cols.issubset(labels.columns):

        raise ValueError(
            "Train_Labels.csv missing expected "
            f"columns {expected_cols}. "
            f"Found {list(labels.columns)}"
        )

    return labels


# ============================================================
# SENSOR HELPERS
# ============================================================

def side_columns(side, channel="vib"):

    if side == "I":

        positions = SIDE_I_POSITIONS

    elif side == "II":

        positions = SIDE_II_POSITIONS

    else:

        raise ValueError(
            "side must be 'I' or 'II'"
        )

    if channel not in {"vib", "shock"}:

        raise ValueError(
            "channel must be 'vib' or 'shock'"
        )

    return [
        f"car{car}_pos{pos}_{channel}"
        for car in range(1, N_CARS + 1)
        for pos in positions
    ]


# ============================================================
# LABEL CHECK
# ============================================================

def check_label_distribution(labels):

    counts = labels["label"].value_counts()

    print("\nLabel distribution:")
    print(counts.to_string())

    print(
        f"\nTotal files: {len(labels)}"
    )

    print("\nExpected distribution:")

    for label, expected in EXPECTED_LABEL_COUNTS.items():

        actual = int(
            counts.get(label, 0)
        )

        status = (
            "OK"
            if actual == expected
            else "MISMATCH"
        )

        print(
            f"{label:8s}: "
            f"expected {expected:3d}, "
            f"found {actual:3d} "
            f"[{status}]"
        )


# ============================================================
# EXTREME VALUE CHECK
# ============================================================

def get_extreme_value_info(df):

    sensor_cols = [
        c for c in df.columns
        if c != "speed"
    ]

    total_extreme = 0
    channels_with_extremes = 0

    for col in sensor_cols:

        values = df[col]

        median = values.median()
        mad = np.median(
            np.abs(values - median)
        )

        if mad == 0:
            continue

        robust_z = (
            0.6745
            * (values - median)
            / mad
        )

        n_extreme = int(
            (np.abs(robust_z) > 10).sum()
        )

        if n_extreme > 0:

            total_extreme += n_extreme
            channels_with_extremes += 1

    return total_extreme, channels_with_extremes


# ============================================================
# SINGLE FILE CHECK
# ============================================================

def sanity_check_file(filepath):

    df = load_rail_file(filepath)

    print(
        f"\n--- {Path(filepath).name} ---"
    )

    print(
        f"Shape: {df.shape} "
        f"(expected "
        f"({EXPECTED_ROWS}, {EXPECTED_COLS}))"
    )

    if df.shape[0] != EXPECTED_ROWS:

        print(
            "! Unexpected row count."
        )

    n_nans = int(
        df.isna().sum().sum()
    )

    print(
        f"NaNs present: {n_nans}"
    )

    print(
        f"Speed: "
        f"min={df['speed'].min():.3f}, "
        f"max={df['speed'].max():.3f}, "
        f"mean={df['speed'].mean():.3f}, "
        f"std={df['speed'].std():.3f}"
    )

    vib_cols = [
        c
        for c in df.columns
        if c.endswith("_vib")
    ]

    shock_cols = [
        c
        for c in df.columns
        if c.endswith("_shock")
    ]

    print(
        "Vibration range: "
        f"{df[vib_cols].min().min():.4f} "
        "to "
        f"{df[vib_cols].max().max():.4f}"
    )

    print(
        "Shock range: "
        f"{df[shock_cols].min().min():.4f} "
        "to "
        f"{df[shock_cols].max().max():.4f}"
    )

    total_extreme, channels_with_extremes = get_extreme_value_info(df)

    print(
        f"Extreme values: {total_extreme} "
        f"across {channels_with_extremes} channels"
    )

    return df


# ============================================================
# FULL DATASET AUDIT
# ============================================================

def audit_dataset(train_dir, labels):

    reports = []

    for _, row in labels.iterrows():

        filename = row["filename"]

        filepath = (
            train_dir / filename
        )

        try:

            df = load_rail_file(
                filepath,
                clean=False
            )

            numeric = df.apply(
                pd.to_numeric,
                errors="coerce"
            )
            total_extreme, channels_with_extremes = (
                get_extreme_value_info(numeric)
            )

            n_missing = int(
                numeric.isna()
                .sum()
                .sum()
            )

            n_inf = int(
                np.isinf(
                    numeric.to_numpy()
                ).sum()
            )

            sensor_cols = [
                col for col in numeric.columns
                if col != "speed"
            ]

            constant_cols = sum(
                numeric[col].nunique(dropna=True) <= 1
                for col in sensor_cols
            )   

            reports.append({
                "filename": filename,
                "label": row["label"],
                "rows": len(df),
                "columns": df.shape[1],
                "missing_values": n_missing,
                "infinite_values": n_inf,
                "constant_channels": constant_cols,
                "extreme_values": total_extreme,
                "channels_with_extremes": channels_with_extremes
            })

        except Exception as e:

            reports.append({

                "filename": filename,

                "label": row["label"],

                "error": str(e)
            })

    return pd.DataFrame(reports)


# ============================================================
# PLOTS
# ============================================================

def plot_class_comparison(
    train_dir,
    labels,
    output_dir,
    side="I",
    n_per_class=2,
    car_to_plot=1
):

    fault_label = f"Side {side}"

    positions = (
        SIDE_I_POSITIONS
        if side == "I"
        else SIDE_II_POSITIONS
    )

    position = positions[0]

    column = (
        f"car{car_to_plot}_"
        f"pos{position}_vib"
    )

    fig, axes = plt.subplots(
        2,
        n_per_class,
        figsize=(5 * n_per_class, 6),
        squeeze=False
    )

    fig.suptitle(
        f"Side {side}: "
        f"Normal vs {fault_label}"
    )

    for row, label in enumerate(
        ["Normal", fault_label]
    ):

        files = labels.loc[
            labels["label"] == label,
            "filename"
        ].head(n_per_class)

        for col_index, filename in enumerate(files):

            df = load_rail_file(
                train_dir / filename
            )

            time = (
                np.arange(len(df))
                / SAMPLING_RATE
            )

            ax = axes[row][col_index]

            ax.plot(
                time,
                df[column],
                linewidth=0.5
            )

            ax.set_title(
                f"{label}: {filename}"
            )

            ax.set_xlabel("Time (s)")

            ax.set_ylabel(
                "Acceleration (m/s²)"
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir
        / f"side_{side}_raw_comparison.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close(fig)

    print(
        f"Saved {output_path}"
    )


# ============================================================
# STEP 1
# ============================================================

def run_step1(
    train_dir,
    labels_path,
    output_dir
):

    print("\n")
    print("=" * 60)
    print("STEP 1 — DATA CHECK")
    print("=" * 60)

    labels = load_labels(
        labels_path
    )

    check_label_distribution(
        labels
    )

    # ----------------------------------------
    # Check one recording from each class
    # ----------------------------------------

    for label in [
        "Normal",
        "Side I",
        "Side II"
    ]:

        matches = labels.loc[
            labels["label"] == label,
            "filename"
        ]

        if matches.empty:
            continue

        sanity_check_file(
            train_dir / matches.iloc[0]
        )

    # ----------------------------------------
    # Full dataset audit
    # ----------------------------------------

    print(
        "\nRunning full dataset audit..."
    )

    audit = audit_dataset(
        train_dir,
        labels
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    audit_path = (
        output_dir
        / "rail_dataset_audit.csv"
    )

    audit.to_csv(
        audit_path,
        index=False
    )

    print(
        f"Audit saved to {audit_path}"
    )

    # ----------------------------------------
    # Raw signal plots
    # ----------------------------------------

    plot_class_comparison(
        train_dir,
        labels,
        output_dir,
        side="I"
    )

    plot_class_comparison(
        train_dir,
        labels,
        output_dir,
        side="II"
    )

    print(
        "\nSTEP 1 COMPLETE"
    )

    return labels