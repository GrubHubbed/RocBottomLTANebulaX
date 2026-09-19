"""
Rail Corrugation — Step 7: Enhanced Feature Engineering

Purpose:
    Address the root causes of subpar Side I detection and Side II
    overfitting identified in Step 6:

    1. Car-Level Spatial Invariance:
       Localized corrugation on a single car (e.g. Train170 on Car 8)
       is diluted by whole-train averaging. We compute permutation-invariant
       order statistics across all 8 cars:
         - Max car differential (max_k [Car_k_SideI - Car_k_SideII])
         - Min car differential (min_k [Car_k_SideI - Car_k_SideII])
         - Median car differential (robust against single-car sensor spikes like Train100)
         - Car dominance vote count (number of cars where Side I > Side II)
         - Top-2 cars mean ratio (mean of 2 highest car ratios)

    2. Dimensionless Normalized Asymmetry Index:
       (Side I - Side II) / (Side I + Side II + eps)
       Bounded in [-1, +1]. Normalizes out high-speed train baseline
       vibration for high-speed runs like Train185 and Train202.

    3. Speed-Normalized Power Features:
       Vibration power normalized by estimated speed squared (v^2)
       to decouple speed kinetic energy from rail surface defect energy.

Saves:
    outputs/step7/rail_features_enhanced.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CONSTANTS
# ============================================================

EPSILON = 1e-6
N_CARS = 8

CAR_METRICS_TO_POOL = [
    "rms_max",
    "rms_mean",
    "std_mean",
    "std_max",
    "total_power_mean",
    "total_power_max",
    "power_0_250_mean",
    "power_0_250_max",
    "power_250_500_mean",
    "power_250_500_max",
    "spectral_centroid_mean",
]

WHOLE_TRAIN_ASYMMETRY_METRICS = [
    "rms_mean",
    "rms_max",
    "std_mean",
    "std_max",
    "total_power_mean",
    "total_power_max",
    "power_0_250_mean",
    "power_250_500_mean",
    "power_500_750_mean",
    "power_750_1000_mean",
    "spectral_centroid_mean",
]


# ============================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================

def compute_car_invariant_features(df, channel="vib"):
    """
    Compute order-invariant spatial statistics across the 8 train cars.

    Instead of relying on positional features like 'car8_vib_...' (which
    overfits to the exact car positions in the 14 training examples),
    we pool across cars to detect localized defects regardless of which
    car passes over the corrugation first.
    """
    new_features = {}

    for metric in CAR_METRICS_TO_POOL:
        car_diffs = []
        car_ratios = []
        car_asyms = []

        for car in range(1, N_CARS + 1):
            col_I = f"{channel}_car{car}_sideI_{metric}"
            col_II = f"{channel}_car{car}_sideII_{metric}"

            if col_I in df.columns and col_II in df.columns:
                val_I = df[col_I].values
                val_II = df[col_II].values

                diff = val_I - val_II
                ratio = val_I / (val_II + EPSILON)
                asym = (val_I - val_II) / (val_I + val_II + EPSILON)

                car_diffs.append(diff)
                car_ratios.append(ratio)
                car_asyms.append(asym)

        if not car_diffs:
            continue

        # Stack into shape (n_samples, n_cars)
        diff_matrix = np.column_stack(car_diffs)
        ratio_matrix = np.column_stack(car_ratios)
        asym_matrix = np.column_stack(car_asyms)

        prefix = f"{channel}_inv_{metric}"

        # 1. Max car differential: catches localized corrugation (Train170 Car 8)
        new_features[f"{prefix}_car_diff_max"] = np.max(diff_matrix, axis=1)

        # 2. Min car differential: catches localized Side II corrugation
        new_features[f"{prefix}_car_diff_min"] = np.min(diff_matrix, axis=1)

        # 3. Median car differential: robust consensus, immune to single-car joint impacts (Train100 Car 5)
        new_features[f"{prefix}_car_diff_median"] = np.median(diff_matrix, axis=1)

        # 4. Standard deviation across cars: measures spatial localization / defect concentration
        new_features[f"{prefix}_car_diff_std"] = np.std(diff_matrix, axis=1)

        # 5. Car Dominance Vote Count: how many of the 8 cars show Side I > Side II
        new_features[f"{prefix}_cars_sideI_dominant_count"] = np.sum(diff_matrix > 0, axis=1).astype(float)

        # 6. Decisive dominance count (Side I at least 25% higher than Side II)
        new_features[f"{prefix}_cars_sideI_decisive_count"] = np.sum(ratio_matrix > 1.25, axis=1).astype(float)

        # 7. Decisive Side II dominance count
        new_features[f"{prefix}_cars_sideII_decisive_count"] = np.sum(ratio_matrix < 0.80, axis=1).astype(float)

        # 8. Mean of top-2 highest car ratios (severe localized defects)
        sorted_ratios = np.sort(ratio_matrix, axis=1)
        new_features[f"{prefix}_top2_cars_ratio_mean"] = np.mean(sorted_ratios[:, -2:], axis=1)
        new_features[f"{prefix}_bottom2_cars_ratio_mean"] = np.mean(sorted_ratios[:, :2], axis=1)

        # 9. Max car asymmetry
        new_features[f"{prefix}_car_asym_max"] = np.max(asym_matrix, axis=1)
        new_features[f"{prefix}_car_asym_min"] = np.min(asym_matrix, axis=1)
        new_features[f"{prefix}_car_asym_median"] = np.median(asym_matrix, axis=1)

    return pd.DataFrame(new_features, index=df.index)


def compute_normalized_asymmetry_features(df, channels=["vib", "shock"]):
    """
    Compute dimensionless normalized asymmetry index for whole-train features:
        A = (Side_I - Side_II) / (Side_I + Side_II + eps)
    Bounded in [-1, +1]. Prevents high-speed vibration scaling from masking
    relative side imbalances.
    """
    new_features = {}

    for channel in channels:
        for metric in WHOLE_TRAIN_ASYMMETRY_METRICS:
            col_I = f"{channel}_sideI_{metric}"
            col_II = f"{channel}_sideII_{metric}"

            if col_I in df.columns and col_II in df.columns:
                val_I = df[col_I].values
                val_II = df[col_II].values

                asym = (val_I - val_II) / (val_I + val_II + EPSILON)
                new_features[f"{channel}_{metric}_asymmetry_index"] = asym

    return pd.DataFrame(new_features, index=df.index)


def compute_speed_normalized_features(df):
    """
    Normalize total vibration power and RMS by estimated train speed.
    Vibration power on rail tracks approximately scales with v^2.
    """
    new_features = {}

    if "estimated_speed_mps" in df.columns:
        speed = df["estimated_speed_mps"].values
        speed_safe = np.maximum(speed, 1.0)  # avoid division by zero
        speed_sq = speed_safe ** 2

        for channel in ["vib", "shock"]:
            for side in ["sideI", "sideII"]:
                power_col = f"{channel}_{side}_total_power_mean"
                rms_col = f"{channel}_{side}_rms_mean"

                if power_col in df.columns:
                    new_features[f"{channel}_{side}_power_per_speed_sq"] = df[power_col].values / speed_sq

                if rms_col in df.columns:
                    new_features[f"{channel}_{side}_rms_per_speed"] = df[rms_col].values / speed_safe

        # Asymmetry in speed-normalized power
        p_I = new_features.get("vib_sideI_power_per_speed_sq")
        p_II = new_features.get("vib_sideII_power_per_speed_sq")
        if p_I is not None and p_II is not None:
            new_features["vib_power_per_speed_sq_diff"] = p_I - p_II
            new_features["vib_power_per_speed_sq_asym"] = (p_I - p_II) / (p_I + p_II + EPSILON)

    return pd.DataFrame(new_features, index=df.index)


# ============================================================
# PIPELINE STEP ENTRY POINT
# ============================================================

def run_step7_features(base_features_path=None, output_dir=None):
    """
    Generate and save the enhanced feature set.
    """
    print()
    print("=" * 60)
    print("STEP 7A: ENHANCED FEATURE ENGINEERING")
    print("=" * 60)

    if base_features_path is None:
        base_features_path = Path("outputs/step3/rail_features.csv")
    else:
        base_features_path = Path(base_features_path)

    if output_dir is None:
        output_dir = Path("outputs/step7")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not base_features_path.exists():
        raise FileNotFoundError(
            f"Base feature file not found at: {base_features_path}\n"
            "Please ensure Step 3 has been run."
        )

    print(f"Loading base features from: {base_features_path}")
    df = pd.read_csv(base_features_path)
    print(f"Base features shape: {df.shape}")

    print("\nComputing spatial permutation-invariant car features...")
    vib_inv_df = compute_car_invariant_features(df, channel="vib")
    print(f"  Extracted {vib_inv_df.shape[1]} invariant vibration features")

    print("\nComputing dimensionless normalized asymmetry indices...")
    asym_df = compute_normalized_asymmetry_features(df)
    print(f"  Extracted {asym_df.shape[1]} normalized asymmetry features")

    print("\nComputing speed-normalized energy features...")
    speed_norm_df = compute_speed_normalized_features(df)
    print(f"  Extracted {speed_norm_df.shape[1]} speed-normalized features")

    # Combine cleanly without DataFrame fragmentation warnings
    enhanced_df = pd.concat(
        [df, vib_inv_df, asym_df, speed_norm_df],
        axis=1
    )

    out_csv = output_dir / "rail_features_enhanced.csv"
    enhanced_df.to_csv(out_csv, index=False)

    n_new = vib_inv_df.shape[1] + asym_df.shape[1] + speed_norm_df.shape[1]
    print(f"\nSuccessfully generated enhanced feature dataset:")
    print(f"  Total samples:  {enhanced_df.shape[0]}")
    print(f"  Total features: {enhanced_df.shape[1] - 2} (Base: {df.shape[1] - 2}, New: {n_new})")
    print(f"  Saved to:       {out_csv}")

    return enhanced_df


if __name__ == "__main__":
    run_step7_features()
