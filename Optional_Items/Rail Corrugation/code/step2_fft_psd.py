"""
Rail Corrugation — Step 2: Frequency-Domain Signal Analysis

Goal:
    Investigate whether Normal, Side I, and Side II recordings have
    different frequency patterns.

Main idea:
    Raw vibration data tells us acceleration over TIME.

    PSD (Power Spectral Density) tells us how much vibration energy
    exists at different FREQUENCIES.

Optimisations in this version:
    1. Each CSV is loaded only once.
    2. Welch PSD is calculated for all 128 sensor channels at once.
    3. PSD results are cached in memory.
    4. All plots reuse the cache instead of re-reading CSV files.

This makes Step 2 substantially faster than repeatedly calculating
PSDs separately for every side/channel combination.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from step1_data_cleaning import (
    SAMPLING_RATE,
    load_rail_file,
    side_columns,
)


# ============================================================
# SETTINGS
# ============================================================

# Welch divides each 10,000-sample signal into smaller chunks.
#
# Larger value:
#     better frequency resolution
#     fewer chunks to average
#
# Smaller value:
#     smoother PSD
#     lower frequency resolution
#
# 2048 is a sensible starting point for this dataset.
WELCH_SEGMENT_SIZE = 2048


# ============================================================
# PSD HELPERS
# ============================================================

def psd_to_db(psd):
    """
    Convert PSD into decibels (dB).

    Why?

    PSD values can differ by very large amounts.
    Converting to dB compresses the scale and makes plots
    much easier to read.

    The tiny 1e-12 prevents log10(0).
    """

    return 10 * np.log10(
        psd + 1e-12
    )


# ============================================================
# CALCULATE ALL PSDs FOR ONE FILE
# ============================================================

def calculate_file_psds(df):
    """
    Calculate PSDs for ALL 128 vibration/shock channels at once.

    Input DataFrame:

        10,000 rows
        x
        129 columns

    Column 1 is speed.

    The remaining 128 columns are:

        64 vibration channels
        64 shock channels

    Instead of doing:

        sensor 1 -> Welch
        sensor 2 -> Welch
        sensor 3 -> Welch
        ...

    we send all 128 sensors into scipy.signal.welch at once.

    This is called VECTORIZATION.

    It avoids lots of slow Python loops.

    Returns:

        frequencies

        psd_by_column:
            dictionary mapping sensor column name -> PSD array
    """

    sensor_columns = [
        column
        for column in df.columns
        if column != "speed"
    ]

    # --------------------------------------------------------
    # Convert DataFrame into NumPy matrix
    # --------------------------------------------------------
    #
    # Shape:
    #
    #     10,000 samples
    #          ×
    #     128 sensors

    signals = df[
        sensor_columns
    ].to_numpy()

    # --------------------------------------------------------
    # Calculate PSD
    # --------------------------------------------------------
    #
    # axis=0 means:
    #
    #     calculate frequency information DOWN each column
    #
    # because every column represents one sensor.

    frequencies, psds = welch(
        signals,
        fs=SAMPLING_RATE,
        nperseg=WELCH_SEGMENT_SIZE,
        axis=0
    )

    # psds now has shape approximately:
    #
    #     1025 frequency bins
    #            ×
    #     128 sensors

    # --------------------------------------------------------
    # Store PSDs by sensor name
    # --------------------------------------------------------

    psd_by_column = {}

    for index, column in enumerate(
        sensor_columns
    ):

        psd_by_column[column] = (
            psds[:, index]
        )

    return frequencies, psd_by_column


# ============================================================
# AVERAGE PSD FOR ONE SIDE
# ============================================================

def get_side_average_psd(
    psd_by_column,
    side,
    channel="vib"
):
    """
    Calculate the average PSD across the 32 sensors belonging
    to one side.

    Side I:
        positions 1, 3, 5, 7
        across all 8 cars

    Side II:
        positions 2, 4, 6, 8
        across all 8 cars

    Example:

        Side I vibration

        sensor 1 PSD ─┐
        sensor 2 PSD ─┤
        sensor 3 PSD ─┤
        ...            ├──> average PSD
        sensor 32 PSD ─┘
    """

    columns = side_columns(
        side,
        channel=channel
    )

    psds = np.array([
        psd_by_column[column]
        for column in columns
    ])

    # Shape:
    #
    #     32 sensors × frequency bins
    #
    # axis=0 averages across the 32 sensors.

    average_psd = np.mean(
        psds,
        axis=0
    )

    return average_psd


# ============================================================
# BUILD PSD CACHE
# ============================================================

def build_psd_cache(
    train_dir,
    labels
):
    """
    Load every training CSV ONCE and calculate everything
    needed for Step 2.

    For every recording we store:

        Side I vibration average PSD
        Side II vibration average PSD

        Side I shock average PSD
        Side II shock average PSD

    After this function finishes, plotting does NOT need
    to read the CSV files again.
    """

    print()
    print("Building PSD cache...")
    print(
        "Each CSV will be loaded only once."
    )

    cache = {}

    frequencies = None

    total = len(labels)

    for count, (_, row) in enumerate(
        labels.iterrows(),
        start=1
    ):

        filename = row["filename"]
        label = row["label"]

        print(
            f"[{count:3d}/{total}] "
            f"{filename}"
        )

        # ----------------------------------------------------
        # LOAD CSV ONCE
        # ----------------------------------------------------

        df = load_rail_file(
            train_dir / filename
        )

        # ----------------------------------------------------
        # CALCULATE ALL 128 PSDs AT ONCE
        # ----------------------------------------------------

        frequencies, psd_by_column = (
            calculate_file_psds(df)
        )

        # ----------------------------------------------------
        # Store the four side/channel summaries
        # ----------------------------------------------------

        cache[filename] = {

            "label": label,

            "side_I_vib":
                get_side_average_psd(
                    psd_by_column,
                    side="I",
                    channel="vib"
                ),

            "side_II_vib":
                get_side_average_psd(
                    psd_by_column,
                    side="II",
                    channel="vib"
                ),

            "side_I_shock":
                get_side_average_psd(
                    psd_by_column,
                    side="I",
                    channel="shock"
                ),

            "side_II_shock":
                get_side_average_psd(
                    psd_by_column,
                    side="II",
                    channel="shock"
                ),
        }

    print()
    print(
        f"PSD cache complete: "
        f"{len(cache)} files processed."
    )

    return frequencies, cache


# ============================================================
# GET PSD FROM CACHE
# ============================================================

def get_cached_psd(
    file_data,
    side,
    channel
):
    """
    Retrieve one PSD from the cache.

    Example:

        side="I"
        channel="vib"

    retrieves:

        side_I_vib
    """

    key = (
        f"side_{side}_{channel}"
    )

    return file_data[key]


# ============================================================
# CLASS AVERAGE
# ============================================================

def calculate_class_average_from_cache(
    cache,
    label,
    side,
    channel
):
    """
    Calculate the average PSD across every recording
    belonging to one class.

    Example:

        all Normal recordings
                ↓
        Side I vibration PSD
                ↓
        average
                ↓
        typical Normal Side I spectrum
    """

    psds = []

    for file_data in cache.values():

        if file_data["label"] != label:
            continue

        psd = get_cached_psd(
            file_data,
            side,
            channel
        )

        psds.append(psd)

    if not psds:

        raise ValueError(
            f"No files found with label "
            f"'{label}'"
        )

    psds = np.array(psds)

    return np.mean(
        psds,
        axis=0
    )


# ============================================================
# NORMAL vs FAULT PLOT
# ============================================================

def plot_normal_vs_fault(
    frequencies,
    cache,
    output_dir,
    side,
    channel
):
    """
    Compare:

        Normal recordings

    against:

        recordings with corrugation on the selected side.

    Example:

        side = "I"

    compares:

        Normal
        vs
        Side I corrugation
    """

    fault_label = (
        f"Side {side}"
    )

    print(
        f"Creating {channel} plot: "
        f"Normal vs {fault_label}"
    )

    # --------------------------------------------------------
    # NORMAL
    # --------------------------------------------------------

    normal_psd = (
        calculate_class_average_from_cache(
            cache,
            label="Normal",
            side=side,
            channel=channel
        )
    )

    # --------------------------------------------------------
    # FAULT
    # --------------------------------------------------------

    fault_psd = (
        calculate_class_average_from_cache(
            cache,
            label=fault_label,
            side=side,
            channel=channel
        )
    )

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        frequencies,
        psd_to_db(normal_psd),
        label="Normal"
    )

    plt.plot(
        frequencies,
        psd_to_db(fault_psd),
        label=f"{fault_label} corrugation"
    )

    plt.xlabel(
        "Frequency (Hz)"
    )

    plt.ylabel(
        "Average PSD (dB)"
    )

    plt.title(
        f"Side {side} {channel}: "
        f"Normal vs {fault_label}"
    )

    # --------------------------------------------------------
    # NYQUIST FREQUENCY
    # --------------------------------------------------------
    #
    # Sampling rate = 10,000 Hz
    #
    # Highest usable frequency:
    #
    #     10,000 / 2 = 5,000 Hz

    plt.xlim(
        0,
        SAMPLING_RATE / 2
    )

    plt.legend()

    plt.tight_layout()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_path = (
        output_dir
        / (
            f"side_{side}_{channel}_"
            f"normal_vs_fault.png"
        )
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    print(
        f"Saved {output_path}"
    )


# ============================================================
# SIDE I vs SIDE II INSIDE ONE FAULT FILE
# ============================================================

def plot_sides_for_fault_file(
    frequencies,
    cache,
    output_dir,
    fault_label,
    channel="vib"
):
    """
    Select one faulty recording and compare:

        Side I sensors
        vs
        Side II sensors

    This helps us see whether the fault appears more strongly
    on the side that is actually labelled as corrugated.
    """

    filename = None
    file_data = None

    # Find first recording with requested label.

    for candidate_filename, candidate_data in (
        cache.items()
    ):

        if (
            candidate_data["label"]
            == fault_label
        ):

            filename = (
                candidate_filename
            )

            file_data = (
                candidate_data
            )

            break

    if file_data is None:

        raise ValueError(
            f"No file found for "
            f"'{fault_label}'"
        )

    # --------------------------------------------------------
    # GET BOTH SIDES
    # --------------------------------------------------------

    side_I_psd = get_cached_psd(
        file_data,
        side="I",
        channel=channel
    )

    side_II_psd = get_cached_psd(
        file_data,
        side="II",
        channel=channel
    )

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        frequencies,
        psd_to_db(side_I_psd),
        label="Side I sensors"
    )

    plt.plot(
        frequencies,
        psd_to_db(side_II_psd),
        label="Side II sensors"
    )

    plt.xlabel(
        "Frequency (Hz)"
    )

    plt.ylabel(
        "Average PSD (dB)"
    )

    plt.title(
        f"{filename} — "
        f"{fault_label} — "
        f"{channel}"
    )

    plt.xlim(
        0,
        SAMPLING_RATE / 2
    )

    plt.legend()

    plt.tight_layout()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    safe_label = (
        fault_label
        .lower()
        .replace(" ", "_")
    )

    output_path = (
        output_dir
        / (
            f"{safe_label}_"
            f"{channel}_sides.png"
        )
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    print(
        f"Saved {output_path}"
    )


# ============================================================
# SAVE CLASS-AVERAGE PSD DATA
# ============================================================

def save_average_psd_data(
    frequencies,
    cache,
    output_dir
):
    """
    Save the class-average PSD curves as a NumPy .npz file.

    Why?

    Step 3 may want to reuse frequency information.

    Saving these arrays means we don't necessarily need to
    recompute everything just to inspect the class averages.

    .npz is NumPy's compressed format for storing arrays.
    """

    output_path = (
        output_dir
        / "class_average_psds.npz"
    )

    np.savez_compressed(

        output_path,

        frequencies=frequencies,

        normal_side_I_vib=
            calculate_class_average_from_cache(
                cache,
                "Normal",
                "I",
                "vib"
            ),

        normal_side_II_vib=
            calculate_class_average_from_cache(
                cache,
                "Normal",
                "II",
                "vib"
            ),

        side_I_vib=
            calculate_class_average_from_cache(
                cache,
                "Side I",
                "I",
                "vib"
            ),

        side_II_vib=
            calculate_class_average_from_cache(
                cache,
                "Side II",
                "II",
                "vib"
            ),

        normal_side_I_shock=
            calculate_class_average_from_cache(
                cache,
                "Normal",
                "I",
                "shock"
            ),

        normal_side_II_shock=
            calculate_class_average_from_cache(
                cache,
                "Normal",
                "II",
                "shock"
            ),

        side_I_shock=
            calculate_class_average_from_cache(
                cache,
                "Side I",
                "I",
                "shock"
            ),

        side_II_shock=
            calculate_class_average_from_cache(
                cache,
                "Side II",
                "II",
                "shock"
            ),
    )

    print(
        f"Saved PSD data to "
        f"{output_path}"
    )


# ============================================================
# STEP 2
# ============================================================

def run_step2(
    train_dir,
    labels,
    output_dir
):
    """
    Run the complete Step 2 analysis.

    Process:

        1. Load each CSV once.
        2. Calculate PSDs.
        3. Cache results.
        4. Compare Normal vs Side I.
        5. Compare Normal vs Side II.
        6. Compare vibration and shock separately.
        7. Compare both sides inside individual fault files.
    """

    print()
    print("=" * 60)
    print(
        "STEP 2 — FREQUENCY ANALYSIS"
    )
    print("=" * 60)

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # BUILD CACHE
    # ========================================================

    frequencies, cache = (
        build_psd_cache(
            train_dir,
            labels
        )
    )

    # ========================================================
    # VIBRATION ANALYSIS
    # ========================================================

    print()
    print(
        "Creating vibration comparisons..."
    )

    plot_normal_vs_fault(
        frequencies,
        cache,
        output_dir,
        side="I",
        channel="vib"
    )

    plot_normal_vs_fault(
        frequencies,
        cache,
        output_dir,
        side="II",
        channel="vib"
    )

    # ========================================================
    # SHOCK ANALYSIS
    # ========================================================

    print()
    print(
        "Creating shock comparisons..."
    )

    plot_normal_vs_fault(
        frequencies,
        cache,
        output_dir,
        side="I",
        channel="shock"
    )

    plot_normal_vs_fault(
        frequencies,
        cache,
        output_dir,
        side="II",
        channel="shock"
    )

    # ========================================================
    # SIDE COMPARISONS
    # ========================================================

    print()
    print(
        "Comparing sides inside "
        "fault recordings..."
    )

    plot_sides_for_fault_file(
        frequencies,
        cache,
        output_dir,
        fault_label="Side I",
        channel="vib"
    )

    plot_sides_for_fault_file(
        frequencies,
        cache,
        output_dir,
        fault_label="Side II",
        channel="vib"
    )

    # ========================================================
    # SAVE CLASS-AVERAGE PSDs
    # ========================================================

    save_average_psd_data(
        frequencies,
        cache,
        output_dir
    )

    print()
    print("=" * 60)
    print(
        "STEP 2 COMPLETE"
    )
    print("=" * 60)

    # Returning these will also allow later steps to reuse
    # the data during the SAME execution of main.py.
    return frequencies, cache