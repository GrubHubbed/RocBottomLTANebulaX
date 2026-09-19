"""
Rail Corrugation — Step 4: Model Training and Evaluation

Goal:
    Train machine-learning classifiers using the features
    created in Step 3.

Models tested:
    1. Random Forest
    2. Extra Trees

Evaluation:
    Stratified 5-fold cross-validation using Macro F1.

Why Macro F1?
    The dataset is heavily imbalanced:

        Normal  = 234
        Side I  = 14
        Side II = 24

    Macro F1 gives all three classes equal importance.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

N_SPLITS = 5

CLASS_ORDER = [
    "Normal",
    "Side I",
    "Side II",
]


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_ml_data(features):
    """
    Separate our feature table into:

        X = information given to the model
        y = correct answers

    We remove:

        filename
        label

    from X.

    filename is only an identifier and should NOT be used
    by the model.
    """

    feature_columns = [
        column
        for column in features.columns
        if column not in {
            "filename",
            "label",
        }
    ]

    X = features[
        feature_columns
    ].copy()

    y = features[
        "label"
    ].copy()

    return X, y, feature_columns


# ============================================================
# MODELS
# ============================================================

def create_models():
    """
    Create the models we want to compare.

    class_weight="balanced"

    tells the model that mistakes on rare Side I / Side II
    recordings should matter more than they otherwise would.
    """

    models = {

        "Random Forest":
            RandomForestClassifier(
                n_estimators=500,

                class_weight="balanced",

                random_state=RANDOM_STATE,

                n_jobs=-1,
            ),

        "Extra Trees":
            ExtraTreesClassifier(
                n_estimators=500,

                class_weight="balanced",

                random_state=RANDOM_STATE,

                n_jobs=-1,
            ),
    }

    return models


# ============================================================
# CROSS-VALIDATION
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    cv,
):
    """
    Evaluate one model using cross-validation.

    CROSS-VALIDATION means:

        Split 1:
            train on 80%
            predict remaining 20%

        Split 2:
            train on a different 80%
            predict remaining 20%

        ...

    Eventually every recording receives a prediction from
    a model that was NOT trained on that recording.

    This gives us a much more trustworthy estimate of
    performance than evaluating the model on its own
    training data.
    """

    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        n_jobs=1,
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
    )

    return predictions, macro_f1


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    y_true,
    y_pred,
    model_name,
    output_dir,
):
    """
    Save a confusion matrix.

    A confusion matrix tells us WHICH mistakes the model
    is making.

    Example:

                     predicted

                 Normal  Side I  Side II

    Actual Normal   220      5       9
    Actual Side I     3     10       1
    Actual Side II    2      1      21
    """

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_ORDER,
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
    )

    ax.set_title(
        f"{model_name}\n"
        f"5-Fold Cross-Validation"
    )

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    output_path = (
        output_dir
        / f"{safe_name}_confusion_matrix.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close(fig)

    print(
        f"Saved {output_path}"
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def save_feature_importance(
    model,
    feature_columns,
    model_name,
    output_dir,
    top_n=25,
):
    """
    Show which features the trained model relied on most.

    This helps answer questions like:

        Does frequency power matter?

        Does RMS matter?

        Are Side I / Side II comparison features useful?

        Is vibration more useful than shock?
    """

    if not hasattr(
        model,
        "feature_importances_",
    ):
        return

    importance = pd.DataFrame({

        "feature":
            feature_columns,

        "importance":
            model.feature_importances_,
    })

    importance = importance.sort_values(
        "importance",
        ascending=False,
    )

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    # --------------------------------------------------------
    # Save ALL feature importances
    # --------------------------------------------------------

    csv_path = (
        output_dir
        / f"{safe_name}_feature_importance.csv"
    )

    importance.to_csv(
        csv_path,
        index=False,
    )

    # --------------------------------------------------------
    # Plot top features
    # --------------------------------------------------------

    top = importance.head(
        top_n
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        top["feature"][::-1],
        top["importance"][::-1],
    )

    plt.xlabel(
        "Feature importance"
    )

    plt.title(
        f"{model_name}\n"
        f"Top {top_n} Features"
    )

    plt.tight_layout()

    plot_path = (
        output_dir
        / f"{safe_name}_feature_importance.png"
    )

    plt.savefig(
        plot_path,
        dpi=150,
    )

    plt.close()

    print(
        f"Saved {csv_path}"
    )

    print(
        f"Saved {plot_path}"
    )


# ============================================================
# FINAL MODEL
# ============================================================

def train_final_model(
    model,
    X,
    y,
):
    """
    After cross-validation has evaluated the model,
    train it one final time using ALL 272 recordings.

    This final model will eventually be used to predict
    the unseen Test files.
    """

    model.fit(
        X,
        y,
    )

    return model


# ============================================================
# STEP 4
# ============================================================

def run_step4(
    features,
    output_dir,
):
    """
    Run complete Step 4.

    1. Prepare feature matrix.
    2. Create stratified folds.
    3. Compare models.
    4. Calculate Macro F1.
    5. Produce confusion matrices.
    6. Select the model with the highest CV Macro F1.
    7. Train that model on all training data.
    8. Save it.
    """

    print()
    print("=" * 60)
    print(
        "STEP 4 — MODEL TRAINING"
    )
    print("=" * 60)

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # PREPARE DATA
    # ========================================================

    X, y, feature_columns = (
        prepare_ml_data(
            features
        )
    )

    print()
    print(
        f"Recordings: {len(X)}"
    )

    print(
        f"Features: {X.shape[1]}"
    )

    print()
    print(
        "Class distribution:"
    )

    print(
        y.value_counts()
    )

    # ========================================================
    # CROSS-VALIDATION
    # ========================================================

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    models = create_models()

    results = []

    predictions_by_model = {}

    # ========================================================
    # TEST EACH MODEL
    # ========================================================

    for model_name, model in models.items():

        print()
        print("-" * 60)

        print(
            f"Evaluating: {model_name}"
        )

        print("-" * 60)

        predictions, macro_f1 = (
            evaluate_model(
                model,
                X,
                y,
                cv,
            )
        )

        predictions_by_model[
            model_name
        ] = predictions

        print()
        print(
            f"Macro F1: "
            f"{macro_f1:.4f}"
        )

        print()

        print(
            classification_report(
                y,
                predictions,
                labels=CLASS_ORDER,
                digits=4,
                zero_division=0,
            )
        )

        # ----------------------------------------------------
        # Per-class F1
        # ----------------------------------------------------

        per_class_f1 = f1_score(
            y,
            predictions,
            labels=CLASS_ORDER,
            average=None,
        )

        result = {

            "model": model_name,

            "macro_f1":
                macro_f1,

            "normal_f1":
                per_class_f1[0],

            "side_I_f1":
                per_class_f1[1],

            "side_II_f1":
                per_class_f1[2],
        }

        results.append(
            result
        )

        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        save_confusion_matrix(
            y,
            predictions,
            model_name,
            output_dir,
        )

    # ========================================================
    # RESULTS TABLE
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "macro_f1",
        ascending=False,
    )

    results_path = (
        output_dir
        / "model_comparison.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    print()
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False
        )
    )

    # ========================================================
    # SELECT BEST MODEL
    # ========================================================

    best_model_name = (
        results_df.iloc[0]["model"]
    )

    print()
    print(
        f"Best CV model: "
        f"{best_model_name}"
    )

    best_model = models[
        best_model_name
    ]

    # ========================================================
    # TRAIN ON ALL DATA
    # ========================================================

    print()
    print(
        "Training final model "
        "on all 272 recordings..."
    )

    best_model = train_final_model(
        best_model,
        X,
        y,
    )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    save_feature_importance(
        best_model,
        feature_columns,
        best_model_name,
        output_dir,
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = (
        output_dir
        / "rail_corrugation_model.joblib"
    )

    joblib.dump(
        {
            "model": best_model,

            "feature_columns":
                feature_columns,

            "model_name":
                best_model_name,
        },

        model_path,
    )

    print()
    print(
        f"Saved final model to:"
        f"\n{model_path}"
    )

    print()
    print("=" * 60)
    print(
        "STEP 4 COMPLETE"
    )
    print("=" * 60)

    return (
        best_model,
        results_df,
    )