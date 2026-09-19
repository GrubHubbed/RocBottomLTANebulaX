"""
Rail Corrugation Detection Pipeline — unified entry point (Steps 1-9).

Step 9 is integrated as the current experimental path:
    9A cache raw Train/Test CSVs to cleaned float32 .npy once
    9B diagnose bogie-pair assumptions
    9C build per-side / file-level feature tables
    9D evaluate REF / F3 / PS / PSN and ablations

Important:
- Step 9 side features REPLACE the old Step 8 wavelength-feature pass.
  We therefore do not run step8_wavelength_features.py in the Step 9 path.
- Step 9 evaluation imports Step 8's validated mirror/model helpers, so
  step8_validation.py remains required.
- The existing Step 8 final trainer is kept available for the legacy Step 8
  model only. Do not use it for the final Step 9 model until the winning
  Step 9 configuration has been selected and the trainer is adapted.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from step1_data_cleaning import load_labels, run_step1
from step2_fft_psd import run_step2
from step3_feature_engineering import run_step3
from step4_model_training import run_step4
from step5_experiments import run_step5
from step6_xgboost_refinement import run_step6
from step7_enhanced_features import run_step7_features
from step7_model_refinement import run_step7_models


DEFAULT_DATA_ROOT = Path(
    r"C:\Users\Ray\Desktop\hackathon\nebula"
    r"\NebulaX-Hackathon-ProblemStatement"
    r"\PS3\02_Datasets\Rail_Corrugation"
)

OUTPUT_ROOT = Path("outputs")
STEP1_OUTPUT = OUTPUT_ROOT / "step1"
STEP2_OUTPUT = OUTPUT_ROOT / "step2"
STEP3_OUTPUT = OUTPUT_ROOT / "step3"
STEP4_OUTPUT = OUTPUT_ROOT / "step4"
STEP5_OUTPUT = OUTPUT_ROOT / "step5"
STEP6_OUTPUT = OUTPUT_ROOT / "step6"
STEP7_OUTPUT = OUTPUT_ROOT / "step7"
STEP8_OUTPUT = OUTPUT_ROOT / "step8"
STEP9_OUTPUT = OUTPUT_ROOT / "step9"
CACHE_OUTPUT = OUTPUT_ROOT / "cache"

FEATURE_FILE = STEP3_OUTPUT / "rail_features.csv"
ENHANCED_FEATURE_FILE = STEP7_OUTPUT / "rail_features_enhanced.csv"

# Legacy Step 8 outputs. Step 9 does NOT use the old wavelength feature file.
STEP8_SUMMARY_FILE = STEP8_OUTPUT / "step8_summary.csv"
FINAL_MODEL_FILE = STEP8_OUTPUT / "rail_model.joblib"

SIDE_TABLE_TRAIN = STEP9_OUTPUT / "side_table_train.csv"
FILE_TABLE_TRAIN = STEP9_OUTPUT / "file_table_train.csv"
SIDE_TABLE_TEST = STEP9_OUTPUT / "side_table_test.csv"
FILE_TABLE_TEST = STEP9_OUTPUT / "file_table_test.csv"
STEP9_RESULTS = STEP9_OUTPUT / "step9_results.csv"


def dataset_paths(data_root):
    root = Path(data_root)
    return root / "Train", root / "Test", root / "Train_Labels.csv"


def require_file(path, instruction):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}\n{instruction}")
    return path


def require_dir(path, instruction):
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Required directory not found: {path}\n{instruction}")
    return path


def get_labels(labels_path):
    return load_labels(labels_path)


def get_features():
    require_file(FEATURE_FILE, "Run Step 3 first: python main.py --step 3")
    print(f"Loading saved features from: {FEATURE_FILE}")
    return pd.read_csv(FEATURE_FILE)


def ensure_enhanced_features():
    """Build Step 7 enhanced features only if they do not already exist."""
    if not ENHANCED_FEATURE_FILE.exists():
        require_file(FEATURE_FILE, "Run Step 3 first: python main.py --step 3")
        run_step7_features(base_features_path=FEATURE_FILE, output_dir=STEP7_OUTPUT)
    return ENHANCED_FEATURE_FILE


def run_python_script(script_name, args):
    script = Path(__file__).resolve().with_name(script_name)
    require_file(script, f"Put {script_name} in the same folder as main.py.")
    cmd = [sys.executable, str(script), *map(str, args)]
    print("\nRunning:", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ============================================================
# LEGACY STEP 8
# ============================================================
# Step 9 replaces step8_wavelength_features.py. Step 8 is retained only so
# the previous baseline can still be reproduced if needed.


def run_step8_validation(repeats=5, quick=False):
    ensure_enhanced_features()
    args = [
        "--features", ENHANCED_FEATURE_FILE,
        "--out", STEP8_OUTPUT,
        "--repeats", repeats,
    ]
    if quick:
        args.append("--quick")

    # Deliberately NO --extra-features:
    # Step 9 owns wavelength extraction now, avoiding duplicate processing.
    run_python_script("step8_validation.py", args)


def validated_thresholds():
    """Use nested thresholds only when nested validation beat untuned validation."""
    if not STEP8_SUMMARY_FILE.exists():
        return (1 / 3, 1 / 3)

    summary = pd.read_csv(STEP8_SUMMARY_FILE)
    col = "config" if "config" in summary.columns else "name"
    nested = summary[summary[col].astype(str).str.startswith("E_")] if col in summary.columns else pd.DataFrame()
    if nested.empty:
        return (1 / 3, 1 / 3)

    row = nested.iloc[0]
    nested_score = row.get("nested_macro")
    untuned_score = row.get("macro_pooled")
    if pd.notna(nested_score) and pd.notna(untuned_score) and nested_score > untuned_score:
        return float(row["t_sideI_mean"]), float(row["t_sideII_mean"])
    return (1 / 3, 1 / 3)


def run_step8_train():
    """
    Train the legacy Step 8 model without the superseded wavelength table.

    This is NOT the final Step 9 trainer. Once Step 9 identifies the winning
    configuration, adapt the final trainer to that configuration.
    """
    ensure_enhanced_features()
    t1, t2 = validated_thresholds()
    args = [
        "--features", ENHANCED_FEATURE_FILE,
        "--t-side-i", t1,
        "--t-side-ii", t2,
        "--out", FINAL_MODEL_FILE,
    ]
    run_python_script("step8_train_final.py", args)


def run_step8(part="all", repeats=5, quick=False):
    if part in {"all", "validation"}:
        run_step8_validation(repeats=repeats, quick=quick)
    if part in {"all", "train"}:
        run_step8_train()


# ============================================================
# STEP 9
# ============================================================


def run_step9_cache(data_root):
    """
    Cache Train/Test CSVs once.

    step9_cache_npy.py skips an output .npy if it already exists, so this is
    safe to rerun and prevents repeated pandas parsing in later Step 9 work.
    """
    require_dir(Path(data_root) / "Train", "Check --data-root.")
    run_python_script(
        "step9_cache_npy.py",
        ["--data-root", data_root, "--cache", CACHE_OUTPUT],
    )


def run_step9_diagnose(labels_path):
    """
    Diagnose BOGIE_PAIRS before committing to the cross-axle features.

    Inspect the printed table. True pairs should have a strong peak,
    L_median around 2-3 m, and a small L_iqr. If not, edit BOGIE_PAIRS in
    step9_side_features.py before building the feature tables.
    """
    require_file(labels_path, "Train_Labels.csv is required for pair diagnosis.")
    require_dir(CACHE_OUTPUT / "Train", "Run Step 9 cache first.")
    run_python_script(
        "step9_side_features.py",
        [
            "--cache", CACHE_OUTPUT,
            "--split", "Train",
            "--labels", labels_path,
            "--out", STEP9_OUTPUT,
            "--diagnose-pairs",
        ],
    )


def run_step9_features(labels_path, split="Train"):
    """
    Build the Step 9 side/file tables.

    This single extraction contains:
      - fixed-Hz energy features
      - wavelength features through 600 mm
      - mean/max/median/second-largest aggregation
      - per-car contrasts
      - own/other and d_* contrasts (log-ratios for log-energy features)
      - cross-axle correlation

    Therefore no separate Step 8 wavelength extraction is run.
    """
    split = split.capitalize()
    require_dir(CACHE_OUTPUT / split, f"Run Step 9 cache first; missing cached {split} data.")

    args = [
        "--cache", CACHE_OUTPUT,
        "--split", split,
        "--out", STEP9_OUTPUT,
    ]
    if split == "Train":
        require_file(labels_path, "Train_Labels.csv is required for Train features.")
        args += ["--labels", labels_path]

    run_python_script("step9_side_features.py", args)


def run_step9_evaluate(repeats=5, only=""):
    """
    Evaluate REF, F3, PS, PSN and ablations on identical file-level splits.

    REF uses Step 7 enhanced features as the old-feature baseline.
    F3/PS/PSN/ablations use the new Step 9 tables.
    """
    require_file(SIDE_TABLE_TRAIN, "Run Step 9 feature extraction first.")
    require_file(FILE_TABLE_TRAIN, "Run Step 9 feature extraction first.")
    ensure_enhanced_features()

    args = [
        "--step9-dir", STEP9_OUTPUT,
        "--old-features", ENHANCED_FEATURE_FILE,
        "--repeats", repeats,
    ]
    if only:
        args += ["--only", only]

    run_python_script("step9_evaluate.py", args)


def run_step9(data_root, labels_path, part="all", repeats=5, only=""):
    """
    Step 9 orchestration.

    'all' runs cache -> diagnose -> Train feature tables -> evaluation.
    The diagnostic is printed before feature extraction, but an automated
    'all' run cannot know whether BOGIE_PAIRS should be edited. For the first
    real-data run, prefer:
        python main.py --step 9 --step9-part cache
        python main.py --step 9 --step9-part diagnose
        # inspect output / edit BOGIE_PAIRS if needed
        python main.py --step 9 --step9-part features
        python main.py --step 9 --step9-part evaluate

    Test features are intentionally separate because they are not needed for
    model selection:
        python main.py --step 9 --step9-part test
    """
    if part in {"all", "cache"}:
        run_step9_cache(data_root)

    if part in {"all", "diagnose"}:
        run_step9_diagnose(labels_path)

    if part in {"all", "features"}:
        run_step9_features(labels_path, split="Train")

    if part in {"all", "evaluate"}:
        run_step9_evaluate(repeats=repeats, only=only)

    if part == "test":
        run_step9_features(labels_path, split="Test")


# ============================================================
# PIPELINE ROUTING
# ============================================================


def run_single_step(step, data_root, train_dir, labels_path, args):
    if step == 1:
        run_step1(train_dir=train_dir, labels_path=labels_path, output_dir=STEP1_OUTPUT)

    elif step == 2:
        run_step2(train_dir=train_dir, labels=get_labels(labels_path), output_dir=STEP2_OUTPUT)

    elif step == 3:
        run_step3(train_dir=train_dir, labels=get_labels(labels_path), output_dir=STEP3_OUTPUT)

    elif step == 4:
        run_step4(features=get_features(), output_dir=STEP4_OUTPUT)

    elif step == 5:
        run_step5(features=get_features(), output_dir=STEP5_OUTPUT)

    elif step == 6:
        run_step6(features=get_features(), output_dir=STEP6_OUTPUT)

    elif step == 7:
        require_file(FEATURE_FILE, "Run Step 3 first: python main.py --step 3")
        run_step7_features(base_features_path=FEATURE_FILE, output_dir=STEP7_OUTPUT)
        run_step7_models(
            enhanced_features_path=ENHANCED_FEATURE_FILE,
            output_dir=STEP7_OUTPUT,
        )

    elif step == 8:
        run_step8(
            part=args.step8_part,
            repeats=args.repeats,
            quick=args.quick,
        )

    elif step == 9:
        run_step9(
            data_root=data_root,
            labels_path=labels_path,
            part=args.step9_part,
            repeats=args.repeats,
            only=args.step9_only,
        )

    else:
        raise ValueError(f"Unknown step: {step}")


def run_all(data_root, train_dir, labels_path, args):
    """
    Full development pipeline.

    We retain Steps 1-7 because Step 9's REF comparison needs the Step 7
    enhanced feature table. We intentionally skip the old Step 8 wavelength
    extraction/final packaging because Step 9 supersedes the wavelength pass
    and the final model should only be packaged after Step 9 selects a winner.
    """
    print("\n" + "=" * 60)
    print("RAIL CORRUGATION — FULL DEVELOPMENT PIPELINE (STEPS 1-7 + 9)")
    print("=" * 60)

    labels = run_step1(
        train_dir=train_dir,
        labels_path=labels_path,
        output_dir=STEP1_OUTPUT,
    )
    run_step2(train_dir=train_dir, labels=labels, output_dir=STEP2_OUTPUT)
    features = run_step3(train_dir=train_dir, labels=labels, output_dir=STEP3_OUTPUT)
    run_step4(features=features, output_dir=STEP4_OUTPUT)
    run_step5(features=features, output_dir=STEP5_OUTPUT)
    run_step6(features=features, output_dir=STEP6_OUTPUT)

    run_step7_features(base_features_path=FEATURE_FILE, output_dir=STEP7_OUTPUT)
    run_step7_models(
        enhanced_features_path=ENHANCED_FEATURE_FILE,
        output_dir=STEP7_OUTPUT,
    )

    # Do not run legacy Step 8 wavelength extraction here.
    # Step 9 contains its own broader wavelength representation.
    run_step9(
        data_root=data_root,
        labels_path=labels_path,
        part="all",
        repeats=args.repeats,
        only=args.step9_only,
    )

    print("\n" + "=" * 60)
    print("FULL DEVELOPMENT PIPELINE COMPLETE")
    print(f"Step 9 results: {STEP9_RESULTS}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Rail Corrugation Detection Pipeline — unified Steps 1-9"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--step",
        type=int,
        choices=range(1, 10),
        help="Run one pipeline step (1-9).",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run Steps 1-7 then the Step 9 experimental pipeline.",
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Rail_Corrugation root containing Train/, Test/, and Train_Labels.csv.",
    )

    parser.add_argument(
        "--step8-part",
        choices=["all", "validation", "train"],
        default="all",
        help="Legacy Step 8: run validation, train, or both. Old wavelength pass is retired.",
    )

    parser.add_argument(
        "--step9-part",
        choices=["all", "cache", "diagnose", "features", "evaluate", "test"],
        default="all",
        help=(
            "Step 9 sub-part. For the first real-data run, use cache -> diagnose -> "
            "features -> evaluate separately so BOGIE_PAIRS can be checked."
        ),
    )

    parser.add_argument(
        "--step9-only",
        default="",
        help="Optional comma list passed to step9_evaluate.py, e.g. PS,PSN,F3.",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Repeated-CV repeats for Step 8/9 evaluation (default: 5).",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Legacy Step 8 validation only: skip nested threshold tuning.",
    )

    args = parser.parse_args()

    train_dir, test_dir, labels_path = dataset_paths(args.data_root)

    # Validate only what the selected route actually needs.
    if args.all or args.step in {1, 2, 3, 8, 9}:
        require_file(
            labels_path,
            f"Check --data-root. Expected labels at: {labels_path}",
        )

    if args.all or args.step in {1, 2, 3}:
        require_dir(
            train_dir,
            f"Check --data-root. Expected Train directory at: {train_dir}",
        )

    if args.step == 9 and args.step9_part in {"all", "cache"}:
        require_dir(train_dir, f"Expected Train directory at: {train_dir}")
        # Test is optional to the cache script, but warn by failing early only
        # for explicit Test feature generation.
    if args.step == 9 and args.step9_part == "test":
        require_dir(test_dir, f"Expected Test directory at: {test_dir}")

    if args.all:
        run_all(args.data_root, train_dir, labels_path, args)
    elif args.step is not None:
        run_single_step(
            args.step,
            args.data_root,
            train_dir,
            labels_path,
            args,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
