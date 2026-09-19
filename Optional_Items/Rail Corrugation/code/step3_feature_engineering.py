"""
Rail Corrugation — Step 3: Feature Engineering

Goal:
    Convert each 1-second recording into numerical features that
    machine-learning models can use.

This revised version extracts:

1. Time-domain vibration/shock features
2. Frequency-domain features
3. Side-level summaries
4. Side I vs Side II comparison features
5. Per-car features
6. Estimated train-speed features

Why the additions?

Previous experiments showed:
    - vibration features are particularly useful
    - Side I vs Side II differences are useful
    - averaging all 32 sensors can hide local behaviour
    - train speed may affect the observed vibration frequencies
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.signal import welch
from scipy.stats import kurtosis

from step1_data_cleaning import (
    SAMPLING_RATE,
    N_CARS,
    SIDE_I_POSITIONS,
    SIDE_II_POSITIONS,
    load_rail_file,
    side_columns,
)


# ============================================================
# PHYSICAL CONSTANTS
# ============================================================

WHEEL_DIAMETER_M = 0.85
SPEED_SENSOR_TEETH = 90


# ============================================================
# FREQUENCY BANDS
# ============================================================

FREQUENCY_BANDS = [
    (0, 250),
    (250, 500),
    (500, 750),
    (750, 1000),
    (1000, 1500),
    (1500, 2000),
    (2000, 2500),
    (2500, 3000),
    (3000, 3500),
    (3500, 4000),
    (4000, 4500),
    (4500, 5000),
]


# ============================================================
# SPEED ESTIMATION
# ============================================================

def extract_speed_features(speed_signal):
    """
    Estimate train speed from the toothed-wheel sensor.

    The speed sensor toggles between 0 and 1 as teeth pass
    the detector.

    There are 90 teeth on the wheel.

    IMPORTANT:
    Each physical tooth normally produces TWO transitions:

        0 -> 1
        1 -> 0

    Therefore:

        revolutions
            ≈ transitions / (2 * 90)

    Since each recording lasts 1 second:

        revolutions_per_second
            ≈ revolutions

    wheel circumference:

        pi * diameter

    speed:

        revolutions_per_second * circumference

    We also return the raw transition count so the model can
    potentially use it directly.
    """

    signal = np.asarray(
        speed_signal,
        dtype=float
    )

    features = {}

    # --------------------------------------------------------
    # Convert to binary
    # --------------------------------------------------------
    #
    # The documented signal toggles between 0 and 1.
    # Thresholding at 0.5 makes transition counting robust to
    # small numerical noise.

    binary = (
        signal > 0.5
    ).astype(int)

    # --------------------------------------------------------
    # Count changes
    # --------------------------------------------------------

    transitions = np.count_nonzero(
        np.diff(binary) != 0
    )

    # --------------------------------------------------------
    # Estimate wheel revolutions
    # --------------------------------------------------------

    revolutions = (
        transitions
        / (2 * SPEED_SENSOR_TEETH)
    )

    duration_seconds = (
        len(signal)
        / SAMPLING_RATE
    )

    if duration_seconds > 0:

        revolutions_per_second = (
            revolutions
            / duration_seconds
        )

    else:

        revolutions_per_second = 0.0

    wheel_circumference = (
        np.pi
        * WHEEL_DIAMETER_M
    )

    speed_mps = (
        revolutions_per_second
        * wheel_circumference
    )

    speed_kmh = (
        speed_mps
        * 3.6
    )

    features[
        "speed_transition_count"
    ] = transitions

    features[
        "estimated_speed_mps"
    ] = speed_mps

    features[
        "estimated_speed_kmh"
    ] = speed_kmh

    return features


# ============================================================
# TIME-DOMAIN FEATURES
# ============================================================

def extract_time_features(signal):

    signal = np.asarray(
        signal,
        dtype=float
    )

    features = {}

    rms = np.sqrt(
        np.mean(signal ** 2)
    )

    features["rms"] = rms

    features["std"] = (
        np.std(signal)
    )

    features["max_abs"] = (
        np.max(
            np.abs(signal)
        )
    )

    features["peak_to_peak"] = (
        np.max(signal)
        - np.min(signal)
    )

    features["kurtosis"] = (
        kurtosis(
            signal,
            fisher=False
        )
    )

    if rms > 0:

        features["crest_factor"] = (
            features["max_abs"]
            / rms
        )

    else:

        features["crest_factor"] = 0.0

    return features


# ============================================================
# FREQUENCY FEATURES
# ============================================================

def extract_frequency_features(signal):

    signal = np.asarray(
        signal,
        dtype=float
    )

    frequencies, psd = welch(
        signal,
        fs=SAMPLING_RATE,
        nperseg=2048
    )

    features = {}

    # --------------------------------------------------------
    # Dominant frequency
    # --------------------------------------------------------

    dominant_index = (
        np.argmax(psd)
    )

    features[
        "dominant_frequency"
    ] = frequencies[
        dominant_index
    ]

    # --------------------------------------------------------
    # Total power
    # --------------------------------------------------------

    total_power = np.trapezoid(
        psd,
        frequencies
    )

    features[
        "total_power"
    ] = total_power

    # --------------------------------------------------------
    # Spectral centroid
    # --------------------------------------------------------
    #
    # Think of this as the "centre of gravity" of the
    # frequency spectrum.

    psd_sum = np.sum(psd)

    if psd_sum > 0:

        features[
            "spectral_centroid"
        ] = (
            np.sum(
                frequencies * psd
            )
            / psd_sum
        )

    else:

        features[
            "spectral_centroid"
        ] = 0.0

    # --------------------------------------------------------
    # Frequency band power
    # --------------------------------------------------------

    for low, high in FREQUENCY_BANDS:

        mask = (
            (frequencies >= low)
            & (frequencies < high)
        )

        if np.count_nonzero(mask) >= 2:

            band_power = np.trapezoid(
                psd[mask],
                frequencies[mask]
            )

        else:

            band_power = 0.0

        features[
            f"power_{low}_{high}"
        ] = band_power

        # Relative band power:
        #
        # What FRACTION of the signal's total power lies
        # inside this frequency range?
        #
        # This may be less sensitive to overall amplitude.

        features[
            f"relative_power_{low}_{high}"
        ] = (
            band_power
            / (total_power + 1e-12)
        )

    return features


# ============================================================
# SINGLE SENSOR
# ============================================================

def extract_sensor_features(
    signal
):

    features = {}

    features.update(
        extract_time_features(
            signal
        )
    )

    features.update(
        extract_frequency_features(
            signal
        )
    )

    return features


# ============================================================
# SUMMARISE MULTIPLE SENSORS
# ============================================================

def summarize_sensor_features(
    sensor_feature_list,
    prefix
):
    """
    Given features from several sensors, calculate:

        mean
        maximum
        standard deviation

    Example:

        32 Side I vibration sensors
                 ↓
        calculate RMS for each
                 ↓
        mean RMS
        maximum RMS
        variation in RMS
    """

    sensor_df = pd.DataFrame(
        sensor_feature_list
    )

    output = {}

    for feature in sensor_df.columns:

        values = sensor_df[
            feature
        ]

        output[
            f"{prefix}_{feature}_mean"
        ] = values.mean()

        output[
            f"{prefix}_{feature}_max"
        ] = values.max()

        output[
            f"{prefix}_{feature}_std"
        ] = values.std()

    return output


# ============================================================
# SIDE FEATURES
# ============================================================

def extract_side_features(
    df,
    side,
    channel
):

    columns = side_columns(
        side,
        channel=channel
    )

    sensor_features = []

    for column in columns:

        features = (
            extract_sensor_features(
                df[column].values
            )
        )

        sensor_features.append(
            features
        )

    return summarize_sensor_features(
        sensor_features,
        prefix=(
            f"{channel}_side{side}"
        )
    )


# ============================================================
# PER-CAR FEATURES
# ============================================================

def extract_per_car_features(
    df,
    channel="vib"
):
    """
    Preserve some spatial information.

    Previously:

        all 32 Side I sensors
            -> one summary

    Now we ALSO calculate summaries for each car.

    Example:

        Car 1 Side I
        Car 2 Side I
        ...
        Car 8 Side I

    This allows the model to notice patterns that are strong
    on only part of the train.
    """

    output = {}

    for car in range(
        1,
        N_CARS + 1
    ):

        for side, positions in [

            (
                "I",
                SIDE_I_POSITIONS
            ),

            (
                "II",
                SIDE_II_POSITIONS
            ),
        ]:

            sensor_features = []

            for position in positions:

                column = (
                    f"car{car}_"
                    f"pos{position}_"
                    f"{channel}"
                )

                features = (
                    extract_sensor_features(
                        df[column].values
                    )
                )

                sensor_features.append(
                    features
                )

            prefix = (
                f"{channel}_"
                f"car{car}_"
                f"side{side}"
            )

            # To avoid exploding the feature count too much,
            # keep only a focused subset at car level.

            car_df = pd.DataFrame(
                sensor_features
            )

            for feature in [
                "rms",
                "std",
                "total_power",
                "power_0_250",
                "power_250_500",
                "power_500_750",
                "power_750_1000",
                "spectral_centroid",
            ]:

                values = (
                    car_df[feature]
                )

                output[
                    f"{prefix}_{feature}_mean"
                ] = values.mean()

                output[
                    f"{prefix}_{feature}_max"
                ] = values.max()

    return output


# ============================================================
# SIDE COMPARISON FEATURES
# ============================================================

def add_side_comparison_features(
    features
):
    """
    Compare corresponding Side I and Side II features.

    These were important in our previous model.

    Example:

        Side I vibration power
        -
        Side II vibration power
    """

    output = dict(
        features
    )

    channels = [
        "vib",
        "shock"
    ]

    base_features = [
        "rms_mean",
        "rms_max",
        "std_mean",
        "std_max",
        "max_abs_mean",
        "total_power_mean",
        "total_power_max",
        "spectral_centroid_mean",
        "power_0_250_mean",
        "power_250_500_mean",
        "power_500_750_mean",
        "power_750_1000_mean",
        "relative_power_0_250_mean",
        "relative_power_250_500_mean",
        "relative_power_500_750_mean",
        "relative_power_750_1000_mean",
    ]

    for channel in channels:

        for feature in base_features:

            side_I_key = (
                f"{channel}_"
                f"sideI_"
                f"{feature}"
            )

            side_II_key = (
                f"{channel}_"
                f"sideII_"
                f"{feature}"
            )

            if (
                side_I_key not in features
                or
                side_II_key not in features
            ):
                continue

            side_I = (
                features[
                    side_I_key
                ]
            )

            side_II = (
                features[
                    side_II_key
                ]
            )

            # Difference

            output[
                f"{channel}_"
                f"{feature}_"
                f"I_minus_II"
            ] = (
                side_I
                - side_II
            )

            # Ratio

            output[
                f"{channel}_"
                f"{feature}_"
                f"I_div_II"
            ] = (
                side_I
                / (
                    side_II
                    + 1e-12
                )
            )

    return output


# ============================================================
# PER-CAR SIDE COMPARISONS
# ============================================================

def add_per_car_side_comparisons(
    features,
    channel="vib"
):
    """
    Compare Side I vs Side II WITHIN each car.

    Example:

        Car 3 Side I RMS
        -
        Car 3 Side II RMS

    This preserves side localisation while accounting for
    car-to-car differences.
    """

    output = dict(
        features
    )

    feature_names = [
        "rms_mean",
        "rms_max",
        "total_power_mean",
        "total_power_max",
        "power_0_250_mean",
        "power_250_500_mean",
        "power_500_750_mean",
        "power_750_1000_mean",
    ]

    for car in range(
        1,
        N_CARS + 1
    ):

        for feature in feature_names:

            side_I_key = (
                f"{channel}_"
                f"car{car}_"
                f"sideI_"
                f"{feature}"
            )

            side_II_key = (
                f"{channel}_"
                f"car{car}_"
                f"sideII_"
                f"{feature}"
            )

            if (
                side_I_key not in features
                or
                side_II_key not in features
            ):
                continue

            side_I = (
                features[
                    side_I_key
                ]
            )

            side_II = (
                features[
                    side_II_key
                ]
            )

            output[
                f"{channel}_"
                f"car{car}_"
                f"{feature}_"
                f"I_minus_II"
            ] = (
                side_I
                - side_II
            )

            output[
                f"{channel}_"
                f"car{car}_"
                f"{feature}_"
                f"I_div_II"
            ] = (
                side_I
                / (
                    side_II
                    + 1e-12
                )
            )

    return output


# ============================================================
# COMPLETE FILE
# ============================================================

def extract_file_features(
    filepath
):

    df = load_rail_file(
        filepath
    )

    features = {}

    # --------------------------------------------------------
    # SPEED
    # --------------------------------------------------------

    features.update(
        extract_speed_features(
            df["speed"].values
        )
    )

    # --------------------------------------------------------
    # SIDE-LEVEL VIBRATION
    # --------------------------------------------------------

    features.update(
        extract_side_features(
            df,
            side="I",
            channel="vib"
        )
    )

    features.update(
        extract_side_features(
            df,
            side="II",
            channel="vib"
        )
    )

    # --------------------------------------------------------
    # SIDE-LEVEL SHOCK
    # --------------------------------------------------------

    features.update(
        extract_side_features(
            df,
            side="I",
            channel="shock"
        )
    )

    features.update(
        extract_side_features(
            df,
            side="II",
            channel="shock"
        )
    )

    # --------------------------------------------------------
    # PER-CAR VIBRATION FEATURES
    # --------------------------------------------------------
    #
    # We start with vibration only because Step 5 showed
    # vibration was substantially more informative.

    features.update(
        extract_per_car_features(
            df,
            channel="vib"
        )
    )

    # --------------------------------------------------------
    # SIDE COMPARISONS
    # --------------------------------------------------------

    features = (
        add_side_comparison_features(
            features
        )
    )

    features = (
        add_per_car_side_comparisons(
            features,
            channel="vib"
        )
    )

    return features


# ============================================================
# COMPLETE TRAINING DATASET
# ============================================================

def build_feature_dataset(
    train_dir,
    labels
):

    rows = []

    total = len(
        labels
    )

    for count, (_, row) in enumerate(
        labels.iterrows(),
        start=1
    ):

        filename = (
            row["filename"]
        )

        print(
            f"[{count:3d}/{total}] "
            f"Extracting {filename}"
        )

        features = (
            extract_file_features(
                train_dir
                / filename
            )
        )

        features[
            "filename"
        ] = filename

        features[
            "label"
        ] = row["label"]

        rows.append(
            features
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# STEP 3
# ============================================================

def run_step3(
    train_dir,
    labels,
    output_dir
):

    print()
    print("=" * 60)
    print(
        "STEP 3 — FEATURE ENGINEERING"
    )
    print("=" * 60)

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    features = (
        build_feature_dataset(
            train_dir,
            labels
        )
    )

    output_path = (
        output_dir
        / "rail_features.csv"
    )

    features.to_csv(
        output_path,
        index=False
    )

    # --------------------------------------------------------
    # VALIDATE OUTPUT
    # --------------------------------------------------------

    feature_columns = [
        c
        for c in features.columns
        if c not in {
            "filename",
            "label"
        }
    ]

    numeric = (
        features[
            feature_columns
        ]
    )

    n_nan = int(
        numeric
        .isna()
        .sum()
        .sum()
    )

    n_inf = int(
        np.isinf(
            numeric.to_numpy()
        ).sum()
    )

    print()
    print(
        f"Recordings: "
        f"{len(features)}"
    )

    print(
        f"ML features: "
        f"{len(feature_columns)}"
    )

    print(
        f"NaN values: {n_nan}"
    )

    print(
        f"Infinite values: {n_inf}"
    )

    print()
    print(
        "Estimated speed summary:"
    )

    print(
        features[
            "estimated_speed_kmh"
        ].describe()
    )

    print()
    print(
        f"Saved features to:\n"
        f"{output_path}"
    )

    print()
    print("=" * 60)
    print(
        "STEP 3 COMPLETE"
    )
    print("=" * 60)

    return features