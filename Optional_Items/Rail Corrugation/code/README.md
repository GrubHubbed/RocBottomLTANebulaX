# Rail Corrugation Detection

Problem Statement 3, subsystem 3. The model classifies each 1-second axle-box recording as **Normal**, **Side I** or **Side II** corrugation. It is scored on macro F1.

## Result

| | |
|---|---|
| **Final model** | XGBoost + ExtraTrees blend (0.65 / 0.35), 3-class, averaged over 5 seeds |
| **Features** | 1,214 (998 time/frequency/bilateral + 216 wavelength-domain) |
| **Cross-validated macro F1** | **0.834 ± 0.010** (5-fold × 3 repeats, pooled out-of-fold) |
| Per-class F1 | Normal 0.974 · Side I 0.640 · Side II 0.889 |
| Test predictions (68 files) | 57 Normal · 5 Side I · 6 Side II |

The expected test distribution, from the training ratio, is about 58 Normal / 3.5 Side I / 6 Side II. The predicted counts give no sign of a train/test distribution shift.

---

## 1. Data

- 272 training files: 234 Normal, 14 Side I, 24 Side II.
- Each file is 1 s at 10 kHz. It holds one speed channel and 128 sensor channels: vibration and shock from 64 axle boxes (8 cars × 8 positions).
- Positions 1/3/5/7 are Side I and positions 2/4/6/8 are Side II.
- Speed comes from the 90-tooth speed wheel (wheel diameter 0.85 m). Each tooth gives two transitions, so v = π × 0.85 × transitions / 180 m/s.

Two findings from exploring the data shaped the design:

1. **44 of the Normal files are stationary.** Their speed is about 0, and their vibration RMS is about 0.07 m/s², against about 0.3 m/s² when moving. That is the sensor noise floor, not a failed speed sensor. Every fault file was recorded at 35 km/h or faster.
2. **Fault files cluster in speed.** Most Side I files sit at 45–49 km/h. Side II files are spread more evenly across 42–67 km/h.

## 2. Approach

### 2.1 Features

The features are built in three layers. All of them are computed per file from the raw signals.

| Layer | What | Why |
|---|---|---|
| **Base** (step 3) | Per-sensor RMS, std, kurtosis, crest factor, Welch PSD band powers in 12 fixed-Hz bands, spectral centroid. Aggregated per side (mean/max/std) and per car. | Standard vibration descriptors. |
| **Bilateral / spatial** (step 7) | Side I − Side II differences, ratios and normalised asymmetry indices. Per-car differentials summarised by car-order-invariant statistics (max, min, median, number of cars where Side I dominates). | The task is to decide *which side* is faulty, and the contrast between the two sides carries more information than absolute energy. The car-invariant statistics catch corrugation that shows up on only one or two cars. |
| **Wavelength-domain** (step 9) | The spectrum is re-binned by physical wavelength λ = v / f into 7 bands from 20 to 600 mm. Also a *tonality* measure (peak ÷ median PSD in that range) and the peak wavelength, aggregated per side, as I − II differences, and per car. | Corrugation is a fixed spatial wavelength on the rail. At different speeds it shows up at different frequencies, so fixed-Hz bands blur it. Wavelength bands line it up across speeds. |

Speed is used for two things only: converting frequencies to wavelengths, and the stationary gate below. Speed itself and speed-normalised features are **not** model inputs. Removing them did not change the score (0.826 vs 0.820), and they risked the model learning "fast train → fault" as a shortcut.

### 2.2 Model

- **Classifier:** XGBoost (depth 3, 160 trees, column subsampling 0.6, L1/L2 regularisation) and ExtraTrees (depth 6, 300 trees). Their probabilities are blended 0.65 / 0.35. The two model families make different errors, and the blend beat either one alone.
- **Class imbalance:** balanced sample weights, with an extra ×1.2 on Side I.
- **Bilateral mirror augmentation**, applied only inside each training fold. Every fault file is mirrored: Side I and Side II features are swapped, differences are negated, ratios are inverted, and the Side I and Side II labels are exchanged. This doubles the number of fault examples and encodes the physical symmetry between the two rails.
- **Stationary gate:** files with speed below 1 m/s are predicted Normal and never shown to the model. Corrugation cannot excite vibration without the wheel rolling.
- **Seed averaging:** the final model averages 5 seeds to reduce variance on the small test set.
- **Decision rule:** plain argmax. There is no threshold tuning (see section 4).

### 2.3 Validation protocol

- **Split:** `RepeatedStratifiedKFold`, 5 folds × 3 repeats, split by file. Stratifying keeps about 3 Side I files in every validation fold.
- **Metric:** in each repeat, the out-of-fold predictions from all 5 folds are pooled and macro F1 is computed once. The table reports mean ± std across repeats. Averaging per-fold F1 instead is unstable when a fold contains only 2–3 Side I files.
- **Leakage controls:** augmentation, sample weights and any tuning happen only inside the training folds. Every configuration is compared on identical splits.
- **Noise floor:** rerunning the *same* model with different seeds or file order moves macro F1 by about 0.01–0.02. Differences smaller than about 0.02 are treated as ties.

## 3. Assumptions

These are the design decisions the documentation left open, and why we made each choice.

| Decision | Assumption and evidence |
|---|---|
| **Random stratified CV rather than grouped CV** | If files were consecutive seconds from one run, neighbouring files would share track and a random split would leak. We checked: adjacent file indices are no closer in speed than random pairs (median difference 22.9 vs 21.2 km/h), and only 4% of neighbours are within 1 km/h. We concluded the files are shuffled. Caveat: Train180, 185 and 202 (all Side I, 66.2–67.0 km/h) may come from the same pass. |
| **Stationary files are Normal** | Corrugation needs rolling contact to produce vibration. All 44 stationary training files are Normal and sit at the sensor noise floor. |
| **The two rails are mirror-symmetric** | This justifies the mirror augmentation. The physics is not perfectly symmetric (for example inner vs outer rail on curves), but the augmentation still improved cross-validated macro F1. |
| **Wavelength range 20–600 mm** | Taken from the info kit's range for corrugation, "a few centimetres to dozens of centimetres". |

## 4. What we tried

All scores are pooled out-of-fold macro F1.

| Experiment | Macro F1 | Outcome |
|---|---|---|
| Random Forest, base features | 0.663 | Baseline |
| ExtraTrees, base features | 0.724 | Kept ExtraTrees as a model family |
| XGBoost + sample weights | 0.763 | |
| + bilateral / spatial features | 0.791 | Kept |
| + mirror augmentation | 0.828 | Kept |
| + XGBoost / ExtraTrees blend | 0.837 | Kept |
| Threshold tuning (tuned and scored on the same predictions) | 0.856 | **Rejected.** Optimistic, because the threshold was chosen on the same predictions it was scored on. |
| Threshold tuning, nested (chosen on inner CV only) | 0.816 vs 0.820 untuned | **Rejected.** No real gain, and the chosen thresholds were unstable (0.47 ± 0.10). |
| Fixing a mirror-augmentation bug (12 columns were not being mirrored) | 0.825 vs 0.824 | Kept for correctness; no change in score |
| Removing speed-derived features | 0.826 vs 0.820 | Kept (removed). Same score, less shortcut risk. |
| **Base + bilateral + wavelength features (final)** | **0.834 ± 0.010** vs 0.809 ± 0.025 | **Kept.** Better on all three classes, with lower variance. |
| Full new feature set (step 9) | 0.793 | Rejected |
| Per-side binary reframing (544 side-samples) | 0.754 | Rejected |
| Per-side reframing + nested threshold | 0.769 | Rejected |
| Cross-axle (same-rail) correlation | ±0.001 in ablation | Rejected: no signal |
| Median and second-largest aggregation | −0.029 in ablation (removing it helped) | Rejected |

The table spans several experiment rounds. Rows 1–2 used single 5-fold CV, rows 3–6 averaged per-fold F1 over 5-fold × 3 repeats, and the rest used pooled out-of-fold F1. Compare rows within the same round rather than across rounds.

## 5. Known limitations

- **Six fault files are missed by every model we built:** Train121, 180, 185 and 62 (Side I), and Train216 and 265 (Side II). Four structurally different pipelines all fail on them. Either their signature is too weak in a 1 s window, or their labels are questionable. Tuning further against them would overfit.
- **The wavelength features did not help for the reason we predicted.** We expected them to rescue the Side I files at unusual speeds: Train121 at 35 km/h, and Train180 and 185 at 66 km/h. They did not. The gain came from mid-speed files instead, such as Train106 and Train202.
- **Side I is the weakest class (F1 0.64).** With only 14 training examples, each misclassified file moves Side I F1 by about 0.05–0.07.
- **The test set is small.** The 68 test files hold about 3–4 Side I files, and one error there moves macro F1 by about 0.08. Expect the test score to vary around the cross-validated estimate by more than that estimate's ± 0.010.

## 6. Reproducing

### Requirements

- Python 3.10+
- numpy, pandas, scipy, scikit-learn, xgboost, joblib, matplotlib

Pin the exact versions you trained with (`pip freeze > requirements.txt`). A saved joblib model may fail to load under a different scikit-learn or xgboost version.

### Train and predict (the submission path)

```bash
# 1. Train on all 272 files. Features are rebuilt from the raw CSVs with the same
#    code predict.py uses, then cached. The first run takes about 10 minutes.
python train_final.py --train-dir <data>/Train --labels <data>/Train_Labels.csv --config REF+WL

# 2. Predict
python predict.py --input <data>/Test --output rail_predictions.csv
```

`predict.py` writes `file_id,prediction`, one row per input file. The `--probs` option also writes the class probabilities. It prints the predicted class counts, and warns if the number of predicted faults is far above what the training ratio suggests.

### Reproducing the experiments

```bash
python main_step7.py --step 3          # base features  (edit DATA_ROOT in main_step7.py first)
python main_step7.py --step 7          # bilateral / spatial features + step-7 experiments
python step8_validation.py --features outputs/step7/rail_features_enhanced.csv
python step9_cache_npy.py --data-root <data>
python step9_side_features.py --labels <data>/Train_Labels.csv
python step9_evaluate.py --step9-dir outputs/step9 \
    --old-features outputs/step7/rail_features_enhanced.csv --repeats 3
```

## 7. File index

| File | Role |
|---|---|
| `step1_data_cleaning.py` | Load and clean recordings, column naming, dataset audit |
| `step2_fft_psd.py` | Exploratory PSD analysis and plots |
| `step3_feature_engineering.py` | Base per-sensor, per-side and per-car features; speed estimation |
| `step4_model_training.py` – `step6_xgboost_refinement.py` | Early model comparisons (RF, ExtraTrees, XGBoost) |
| `step7_enhanced_features.py` | Bilateral, asymmetry and car-invariant features |
| `step7_model_refinement.py` | Step-7 experiments: weighting, mirror augmentation, blend |
| `step8_validation.py` | Validation harness: fixed mirror, stationary gate, speed ablation, nested thresholds |
| `step9_cache_npy.py` | Converts the raw CSVs to `.npy` once, for fast reloading |
| `step9_side_features.py` | Wavelength, robust-aggregation and cross-axle features; side-level and file-level tables |
| `step9_evaluate.py` | Final comparison (REF, REF+WL, per-side), ablations, per-file hit rates |
| `rail_model.py` | Feature pipeline and model class shared by training and inference |
| `train_final.py` | Trains and saves the submission model |
| `predict.py` | Inference entry point (`--input`, `--output`) |

`predict.py` needs `rail_model.py`, `step1`, `step3`, `step7_enhanced_features`, `step8_validation`, `step9_side_features` and `outputs/final/rail_model.joblib` in the same folder.
