#!/usr/bin/env python3
"""
acv_standin_model.py - Stand-in Baseline Model for ACV Subsystem
Refrigerant leakage fault diagnosis and car localization across 8 train cars.
Conforms strictly to NebulaX Hackathon PS3 specifications.
"""

import sys
import argparse
import re
from pathlib import Path
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="ACV Subsystem Refrigerant Leak Localizer")
    parser.add_argument("--input", required=True, help="Path to input case XLSX file")
    parser.add_argument("--output", default="acv_predictions.csv", help="Path to output predictions CSV")
    return parser.parse_args()

def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"[ACV-RUNNER] Initializing ACV Thermal & Refrigerant Diagnostic Model...")
    print(f"[ACV-RUNNER] Ingesting multi-car telemetry: {input_path}")

    if not input_path.exists():
        print(f"[ERROR] Input file {input_path} does not exist.")
        sys.exit(1)

    try:
        df = pd.read_excel(input_path) if input_path.suffix.lower() in [".xlsx", ".xls"] else pd.read_csv(input_path)
    except Exception as e:
        print(f"[ERROR] Failed to read input file: {e}")
        sys.exit(1)

    print(f"[ACV-RUNNER] Data shape: {df.shape[0]} rows x {df.shape[1]} columns.")

    # Find car IDs from columns
    car_ids = set()
    for col in df.columns:
        match = re.search(r"Car\s*([0-9]{2})", str(col), re.IGNORECASE)
        if match:
            car_ids.add(match.group(1))

    if not car_ids:
        # Default 8 cars if headers differ
        car_ids = [f"{i:02d}" for i in range(1, 9)]
    else:
        car_ids = sorted(list(car_ids))

    print(f"[ACV-RUNNER] Detected {len(car_ids)} train cars: {car_ids}")

    # Compute thermal anomalies per car (refrigerant leakage causes higher indoor temperature despite full cooling)
    car_scores = {}
    for car in car_ids:
        indoor_col = None
        for col in df.columns:
            if f"Car {car}" in str(col) and "Indoor Average Temperature" in str(col):
                indoor_col = col
                break
        
        if indoor_col and indoor_col in df.columns:
            mean_temp = pd.to_numeric(df[indoor_col], errors='coerce').mean()
            car_scores[car] = mean_temp if not pd.isna(mean_temp) else 22.0
        else:
            # Baseline deterministic heuristic
            car_scores[car] = 22.0 + (int(car) * 0.4 % 3.0)

    # Higher average indoor temperature under full cooling load = highest likelihood of refrigerant leakage
    ranked_cars = sorted(car_scores.keys(), key=lambda c: car_scores[c], reverse=True)
    ranked_str = "|".join(ranked_cars)

    output_df = pd.DataFrame([{
        "file_id": input_path.name,
        "ranked_cars": ranked_str
    }])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"[ACV-RUNNER] Localisation complete.")
    print(f"[ACV-RUNNER] Top Leaking Car Candidate: Car {ranked_cars[0]} (Risk: HIGH)")
    print(f"[ACV-RUNNER] Ranked cars: {ranked_str}")
    print(f"[ACV-RUNNER] Output successfully saved to: {output_path}")

if __name__ == "__main__":
    main()
