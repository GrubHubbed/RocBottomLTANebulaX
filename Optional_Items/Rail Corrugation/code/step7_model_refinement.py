"""
Rail Corrugation — Step 7: Model Refinement & Imbalance Mitigation

Purpose:
    Systematically evaluate and benchmark the advanced techniques
    designed to solve Side I underperformance and Side II overfitting:

    1. Class-Weighted XGBoost Training (sample_weight)
    2. Enhanced Invariant & Dimensionless Asymmetry Features
    3. Physics-Based Bilateral Symmetry Data Augmentation (inside training folds)
    4. Tree Regularization (max_depth=3, min_child_weight=3, colsample=0.6)
    5. Out-of-Fold Decision Threshold Tuning
    6. Blended Ensemble (Regularized XGBoost + Balanced ExtraTrees)

Outputs saved to:
    outputs/step7/
        - step7_experiment_summary.csv
        - step7_fold_results.csv
        - step7_file_error_analysis.csv
        - step7_predictions_by_file.csv
        - step7_feature_importance.csv
        - step7_confusion_matrix.png
        - step7_side_f1_comparison.png
        - rail_corrugation_step7_champion.joblib
"""

from pathlib import Path
import time
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 3  # 15 evaluations per experiment

CLASS_NAMES = ["Normal", "Side I", "Side II"]
LABEL_TO_INT = {"Normal": 0, "Side I": 1, "Side II": 2}
INT_TO_LABEL = {0: "Normal", 1: "Side I", 2: "Side II"}
EPSILON = 1e-6


# ============================================================
# BILATERAL SYMMETRY MIRRORING
# ============================================================

def create_mirrored_batch(X_in, y_in):
    """
    Apply physical bilateral mirror symmetry to rail telemetry features.

    In railway physics, track corrugation on Side I and Side II are mirror
    images of each other. Swapping lateral channels and inverting cross-rail
    differentials turns Side I defects into valid Side II defects and vice versa.

    IMPORTANT: Applied STRICTLY to training folds to prevent data leakage.
    Validation folds remain 100% untouched.
    """
    X_mir = X_in.copy()
    feature_cols = list(X_in.columns)
    swapped = set()

    for c in feature_cols:
        if c in swapped:
            continue

        # 1. Lateral Side I <-> Side II sensor columns
        if "sideI" in c and "sideII" not in c:
            c_opp = c.replace("sideI", "sideII")
            if c_opp in feature_cols:
                tmp = X_mir[c].copy()
                X_mir[c] = X_mir[c_opp]
                X_mir[c_opp] = tmp
                swapped.add(c)
                swapped.add(c_opp)

        elif "sideI_decisive_count" in c:
            c_opp = c.replace("sideI_decisive_count", "sideII_decisive_count")
            if c_opp in feature_cols:
                tmp = X_mir[c].copy()
                X_mir[c] = X_mir[c_opp]
                X_mir[c_opp] = tmp
                swapped.add(c)
                swapped.add(c_opp)

        elif "car_diff_max" in c:
            c_opp = c.replace("car_diff_max", "car_diff_min")
            if c_opp in feature_cols:
                tmp_max = -X_mir[c_opp].copy()
                tmp_min = -X_mir[c].copy()
                X_mir[c] = tmp_max
                X_mir[c_opp] = tmp_min
                swapped.add(c)
                swapped.add(c_opp)

        elif "car_asym_max" in c:
            c_opp = c.replace("car_asym_max", "car_asym_min")
            if c_opp in feature_cols:
                tmp_max = -X_mir[c_opp].copy()
                tmp_min = -X_mir[c].copy()
                X_mir[c] = tmp_max
                X_mir[c_opp] = tmp_min
                swapped.add(c)
                swapped.add(c_opp)

        elif "top2_cars_ratio_mean" in c:
            c_opp = c.replace("top2_cars_ratio_mean", "bottom2_cars_ratio_mean")
            if c_opp in feature_cols:
                tmp_top = 1.0 / (X_mir[c_opp] + EPSILON)
                tmp_bot = 1.0 / (X_mir[c] + EPSILON)
                X_mir[c] = tmp_top
                X_mir[c_opp] = tmp_bot
                swapped.add(c)
                swapped.add(c_opp)

        elif "dominant_count" in c:
            X_mir[c] = 8.0 - X_mir[c]
            swapped.add(c)

        # 2. Invert cross-rail differentials and asymmetry indices
        elif "I_minus_II" in c or "asymmetry" in c or "asym" in c or "car_diff_median" in c:
            X_mir[c] = -X_mir[c]
            swapped.add(c)

        # 3. Invert cross-rail ratios
        elif "I_div_II" in c:
            X_mir[c] = 1.0 / (X_mir[c] + EPSILON)
            swapped.add(c)

    # Invert labels: Side I (1) <-> Side II (2), Normal (0) remains Normal
    y_mir = y_in.copy()
    y_mir[y_in == 1] = 2
    y_mir[y_in == 2] = 1

    return X_mir, y_mir


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

def optimize_decision_thresholds(oof_probs, y_true):
    """
    Search for probability scale factors (theta_1, theta_2) on validation
    probabilities that maximize Macro F1 score.
    """
    best_macro = f1_score(y_true, np.argmax(oof_probs, axis=1), average="macro")
    best_t = (1.0, 1.0, 1.0)

    for t1 in np.linspace(0.18, 0.45, 28):
        for t2 in np.linspace(0.20, 0.45, 26):
            adj = np.column_stack([
                oof_probs[:, 0],
                oof_probs[:, 1] / t1,
                oof_probs[:, 2] / t2,
            ])
            preds = np.argmax(adj, axis=1)
            score = f1_score(y_true, preds, average="macro")
            if score > best_macro:
                best_macro = score
                best_t = (1.0, float(t1), float(t2))

    return best_t, best_macro


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_cross_validation_experiments(features_df, output_dir):
    """
    Run 5-split x 3-repeat RepeatedStratifiedKFold cross-validation
    across all proposed experiments.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filenames = features_df["filename"].values
    y = features_df["label"].map(LABEL_TO_INT).values
    X_all = features_df.drop(columns=["filename", "label"]).copy()

    # Identify base features vs enhanced features
    base_cols = [
        c for c in X_all.columns
        if not c.startswith("vib_inv_")
        and not "asymmetry_index" in c
        and not "power_per_speed" in c
        and not "rms_per_speed" in c
    ]
    X_base = X_all[base_cols].copy()

    experiments = [
        {
            "name": "Exp1_XGB_baseline",
            "features": "base",
            "sample_weight": False,
            "augmentation": False,
            "regularized": False,
            "ensemble": False,
        },
        {
            "name": "Exp2_XGB_sample_weighted",
            "features": "base",
            "sample_weight": True,
            "augmentation": False,
            "regularized": False,
            "ensemble": False,
        },
        {
            "name": "Exp3_XGB_enhanced_features",
            "features": "enhanced",
            "sample_weight": True,
            "augmentation": False,
            "regularized": False,
            "ensemble": False,
        },
        {
            "name": "Exp4_XGB_bilateral_augmented",
            "features": "enhanced",
            "sample_weight": True,
            "augmentation": True,
            "regularized": False,
            "ensemble": False,
        },
        {
            "name": "Exp5_XGB_bilateral_regularized",
            "features": "enhanced",
            "sample_weight": True,
            "augmentation": True,
            "regularized": True,
            "ensemble": False,
        },
        {
            "name": "Exp6_Ensemble_XGB_ExtraTrees",
            "features": "enhanced",
            "sample_weight": True,
            "augmentation": True,
            "regularized": True,
            "ensemble": True,
        },
    ]

    rskf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE
    )

    all_fold_records = []
    all_pred_records = []
    summary_records = []
    champion_oof_probs = None
    champion_name = None
    best_overall_macro_f1 = -1.0

    total_evals = len(experiments) * N_SPLITS * N_REPEATS
    print(f"\nStarting {len(experiments)} experiments with {N_SPLITS} folds x {N_REPEATS} repeats ({total_evals} total evaluations)...\n")

    for exp_idx, exp in enumerate(experiments, 1):
        exp_name = exp["name"]
        print(f"[{exp_idx}/{len(experiments)}] Running {exp_name}...")
        t_start = time.time()

        X_curr = X_enhanced = X_all if exp["features"] == "enhanced" else X_base

        fold_macro_scores = []
        fold_normal_scores = []
        fold_sideI_scores = []
        fold_sideII_scores = []

        # Accumulator for out-of-fold probabilities across all repeats
        oof_probs_sum = np.zeros((len(y), 3))

        for fold_i, (train_idx, val_idx) in enumerate(rskf.split(X_curr, y), 1):
            X_tr, y_tr = X_curr.iloc[train_idx].copy(), y[train_idx].copy()
            X_va, y_va = X_curr.iloc[val_idx].copy(), y[val_idx].copy()
            val_fnames = filenames[val_idx]

            # 1. Physics-based Bilateral Symmetry Augmentation
            if exp["augmentation"]:
                fault_mask = (y_tr == 1) | (y_tr == 2)
                X_tr_faults, y_tr_faults = X_tr[fault_mask], y_tr[fault_mask]
                X_tr_mir, y_tr_mir = create_mirrored_batch(X_tr_faults, y_tr_faults)
                X_tr = pd.concat([X_tr, X_tr_mir], axis=0, ignore_index=True)
                y_tr = np.concatenate([y_tr, y_tr_mir])

            # 2. Sample weights
            if exp["sample_weight"]:
                sw_tr = compute_sample_weight("balanced", y_tr)
                # Slight extra boost for minority Side I
                sw_tr[y_tr == 1] *= 1.2
            else:
                sw_tr = None

            # 3. Model construction
            if exp["regularized"]:
                xgb_model = XGBClassifier(
                    n_estimators=160,
                    max_depth=3,
                    learning_rate=0.035,
                    min_child_weight=3,
                    subsample=0.8,
                    colsample_bytree=0.6,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    objective="multi:softprob",
                    num_class=3,
                    random_state=RANDOM_STATE + fold_i,
                    n_jobs=4,
                )
            else:
                xgb_model = XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.03,
                    min_child_weight=1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="multi:softprob",
                    num_class=3,
                    random_state=RANDOM_STATE + fold_i,
                    n_jobs=4,
                )

            xgb_model.fit(X_tr, y_tr, sample_weight=sw_tr)
            val_probs = xgb_model.predict_proba(X_va)

            # 4. Optional ExtraTrees ensemble blend
            if exp["ensemble"]:
                et_model = ExtraTreesClassifier(
                    n_estimators=200,
                    max_depth=6,
                    min_samples_split=3,
                    class_weight="balanced",
                    random_state=RANDOM_STATE + fold_i,
                    n_jobs=4,
                )
                et_model.fit(X_tr, y_tr)
                et_probs = et_model.predict_proba(X_va)
                # Weighted blend: 65% regularized XGBoost + 35% ExtraTrees
                val_probs = 0.65 * val_probs + 0.35 * et_probs

            oof_probs_sum[val_idx] += val_probs
            fold_preds = np.argmax(val_probs, axis=1)

            f1_macro = f1_score(y_va, fold_preds, average="macro")
            f1_per_class = f1_score(y_va, fold_preds, average=None, labels=[0, 1, 2])

            fold_macro_scores.append(f1_macro)
            fold_normal_scores.append(f1_per_class[0])
            fold_sideI_scores.append(f1_per_class[1])
            fold_sideII_scores.append(f1_per_class[2])

            all_fold_records.append({
                "experiment": exp_name,
                "fold": fold_i,
                "macro_f1": f1_macro,
                "normal_f1": f1_per_class[0],
                "side_I_f1": f1_per_class[1],
                "side_II_f1": f1_per_class[2],
            })

            for fn, yt, yp in zip(val_fnames, y_va, fold_preds):
                all_pred_records.append({
                    "experiment": exp_name,
                    "fold": fold_i,
                    "filename": fn,
                    "true_label": INT_TO_LABEL[yt],
                    "prediction": INT_TO_LABEL[yp],
                    "correct": bool(yt == yp),
                })

        # Calculate average OOF probabilities
        avg_oof_probs = oof_probs_sum / N_REPEATS

        macro_mean = np.mean(fold_macro_scores)
        macro_std = np.std(fold_macro_scores)
        normal_mean = np.mean(fold_normal_scores)
        side_I_mean = np.mean(fold_sideI_scores)
        side_II_mean = np.mean(fold_sideII_scores)
        elapsed = time.time() - t_start

        print(f"   -> Macro F1: {macro_mean:.4f} (+/- {macro_std:.4f}) | "
              f"Side I: {side_I_mean:.4f} | Side II: {side_II_mean:.4f} | "
              f"Normal: {normal_mean:.4f} ({elapsed:.1f}s)")

        summary_records.append({
            "experiment": exp_name,
            "macro_f1_mean": macro_mean,
            "macro_f1_std": macro_std,
            "side_I_f1_mean": side_I_mean,
            "side_II_f1_mean": side_II_mean,
            "normal_f1_mean": normal_mean,
            "features_used": X_curr.shape[1],
        })

        if macro_mean > best_overall_macro_f1:
            best_overall_macro_f1 = macro_mean
            champion_name = exp_name
            champion_oof_probs = avg_oof_probs.copy()

    # --------------------------------------------------------
    # Threshold Tuned Variant of Champion
    # --------------------------------------------------------
    print(f"\n[Post-Processing] Optimizing decision thresholds on {champion_name}...")
    best_t, tuned_macro = optimize_decision_thresholds(champion_oof_probs, y)
    adj_probs = np.column_stack([
        champion_oof_probs[:, 0],
        champion_oof_probs[:, 1] / best_t[1],
        champion_oof_probs[:, 2] / best_t[2],
    ])
    tuned_preds = np.argmax(adj_probs, axis=1)
    tuned_per_class = f1_score(y, tuned_preds, average=None, labels=[0, 1, 2])

    print(f"   -> Tuned Thresholds (Normal=1.0, Side I={best_t[1]:.2f}, Side II={best_t[2]:.2f})")
    print(f"   -> Tuned Macro F1: {tuned_macro:.4f} | Side I: {tuned_per_class[1]:.4f} | Side II: {tuned_per_class[2]:.4f} | Normal: {tuned_per_class[0]:.4f}")

    summary_records.append({
        "experiment": f"{champion_name}_threshold_tuned",
        "macro_f1_mean": tuned_macro,
        "macro_f1_std": 0.0,
        "side_I_f1_mean": tuned_per_class[1],
        "side_II_f1_mean": tuned_per_class[2],
        "normal_f1_mean": tuned_per_class[0],
        "features_used": X_all.shape[1],
    })

    # Save summary tables
    summary_df = pd.DataFrame(summary_records).sort_values("macro_f1_mean", ascending=False)
    summary_df.to_csv(output_dir / "step7_experiment_summary.csv", index=False)

    fold_df = pd.DataFrame(all_fold_records)
    fold_df.to_csv(output_dir / "step7_fold_results.csv", index=False)

    pred_df = pd.DataFrame(all_pred_records)
    pred_df.to_csv(output_dir / "step7_predictions_by_file.csv", index=False)

    # File error analysis
    err_grouped = pred_df.groupby(["experiment", "filename", "true_label"]).agg(
        times_tested=("correct", "count"),
        times_correct=("correct", "sum")
    ).reset_index()
    err_grouped["accuracy"] = err_grouped["times_correct"] / err_grouped["times_tested"]
    err_grouped.to_csv(output_dir / "step7_file_error_analysis.csv", index=False)

    # --------------------------------------------------------
    # Generate Visualizations
    # --------------------------------------------------------
    save_summary_charts(summary_df, output_dir)
    save_confusion_matrix_chart(y, np.argmax(champion_oof_probs, axis=1), output_dir)

    # --------------------------------------------------------
    # Train and Save Full Champion Model
    # --------------------------------------------------------
    print(f"\nTraining final Champion Model ({champion_name}) on full augmented dataset...")
    fault_mask = (y == 1) | (y == 2)
    X_faults, y_faults = X_all[fault_mask], y[fault_mask]
    X_mir, y_mir = create_mirrored_batch(X_faults, y_faults)
    X_full = pd.concat([X_all, X_mir], axis=0, ignore_index=True)
    y_full = np.concatenate([y, y_mir])

    sw_full = compute_sample_weight("balanced", y_full)
    sw_full[y_full == 1] *= 1.2

    final_model = XGBClassifier(
        n_estimators=160,
        max_depth=3,
        learning_rate=0.035,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.6,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=3,
        random_state=RANDOM_STATE,
        n_jobs=4,
    )
    final_model.fit(X_full, y_full, sample_weight=sw_full)

    # Save feature importances
    feat_imp = pd.DataFrame({
        "feature": X_all.columns,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    feat_imp.to_csv(output_dir / "step7_feature_importance.csv", index=False)

    # Package model artifact with threshold config and metadata
    champion_package = {
        "model": final_model,
        "champion_name": champion_name,
        "feature_names": list(X_all.columns),
        "optimal_thresholds": best_t,
        "classes": CLASS_NAMES,
    }
    model_path = output_dir / "rail_corrugation_step7_champion.joblib"
    joblib.dump(champion_package, model_path)
    print(f"Serialized champion model package saved to: {model_path}")

    return summary_df


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def save_summary_charts(summary_df, output_dir):
    """Save comparative bar charts for Macro, Side I, and Side II F1."""
    plt.figure(figsize=(12, 6))
    df = summary_df.copy().sort_values("macro_f1_mean", ascending=True)

    y_pos = np.arange(len(df))
    width = 0.25

    plt.barh(y_pos - width, df["side_I_f1_mean"], width, label="Side I F1", color="#d9534f")
    plt.barh(y_pos, df["side_II_f1_mean"], width, label="Side II F1", color="#f0ad4e")
    plt.barh(y_pos + width, df["macro_f1_mean"], width, label="Macro F1", color="#5cb85c")

    plt.yticks(y_pos, df["experiment"])
    plt.xlabel("F1 Score")
    plt.title("Step 7 Performance Comparison across Experiments")
    plt.legend(loc="lower right")
    plt.xlim(0, 1.0)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()

    out_path = output_dir / "step7_side_f1_comparison.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved performance comparison chart to: {out_path}")


def save_confusion_matrix_chart(y_true, y_pred, output_dir):
    """Save normalized confusion matrix for champion model."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(3),
        yticks=np.arange(3),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        title="Champion Model Confusion Matrix (Normalized)",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    for i in range(3):
        for j in range(3):
            val_pct = f"{cm_norm[i, j]*100:.1f}%\n({cm[i, j]})"
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, val_pct, ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    plt.tight_layout()
    out_path = output_dir / "step7_confusion_matrix.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix chart to: {out_path}")


# ============================================================
# PIPELINE STEP ENTRY POINT
# ============================================================

def run_step7_models(enhanced_features_path=None, output_dir=None):
    """Entry point to execute Step 7 model refinement."""
    print()
    print("=" * 60)
    print("STEP 7B: MODEL REFINEMENT & OVERFITTING PREVENTION")
    print("=" * 60)

    if enhanced_features_path is None:
        enhanced_features_path = Path("outputs/step7/rail_features_enhanced.csv")
    else:
        enhanced_features_path = Path(enhanced_features_path)

    if output_dir is None:
        output_dir = Path("outputs/step7")
    else:
        output_dir = Path(output_dir)

    if not enhanced_features_path.exists():
        raise FileNotFoundError(
            f"Enhanced features file not found at: {enhanced_features_path}\n"
            "Please run step 7 feature extraction first."
        )

    print(f"Loading enhanced features from: {enhanced_features_path}")
    df = pd.read_csv(enhanced_features_path)
    print(f"Features shape: {df.shape}")

    summary_df = run_cross_validation_experiments(df, output_dir)
    print("\nStep 7 Model Refinement complete. All artifacts saved to:", output_dir)
    return summary_df


if __name__ == "__main__":
    run_step7_models()
