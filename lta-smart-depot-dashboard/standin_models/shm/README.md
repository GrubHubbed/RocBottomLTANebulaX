# SHM — Structural Health Monitoring

Cumulative fatigue damage regression for PS3. This is the model that produced the submitted
`shm_predictions.csv`.

Separate from `lta-smart-depot-dashboard/standin_models/shm_standin_model.py`, which is a
placeholder baseline used while the dashboard was being wired up.

**Score: 0.9652 held out. 0.978 cross-validated** (leave-one-out over all 64 training files,
MAPE 2.20%, on the official metric `max(0, 1 − MAPE)`). That is out-of-fold, not training-set
performance.

## Running it

```bash
pip install numpy rainflow pandas
python shm_model.py --input ./Test --output shm_predictions.csv
```

No model file to load — the entire trained state is four constants in the source. It can also be
imported directly (`predict_damage`, `predict_batch`, `interpret`).

## Method

Each stress time series is rainflow-counted (ASTM E1049), then Miner's linear damage rule applied:

```
D = Σ nᵢ/Nᵢ ,  Nᵢ = C / σᵢᵐ   ⟹   D = S(m)/C ,  S(m) = Σ nᵢ · σᵢᵐ
```

`m = 5.0` was selected by cross-validation and sits in a sharp, well-defined minimum. `C` is
fitted by minimising MAPE directly, because MAPE is the scoring metric — solved in `u = 1/C`,
where the objective is convex and piecewise-linear so the exact minimiser is one of its kinks.

A third fitted number corrects for damage concentration. Residual analysis found one systematic
effect and only one: records whose damage is carried by a single dominant rainflow cycle were
under-predicted (Spearman −0.43 against that cycle's share of the damage sum). In the worst
training file, one cycle with a count of 0.5 carries 31% of the total damage — exactly where a
single power-law S-N curve is least trustworthy. Scaling by `exp(γ · (top1 − top1_ref))` took
LOOCV MAPE from 2.544% to 2.203%, with γ re-chosen inside every fold.

Stated honestly: that gain is carried by about five files, and shrinks to a third of its size if
the three it helps most are removed. It was the right call on the evidence available, not a
settled result.

Leave-one-out rather than a single train/test split because 64 files is small enough that one
split cannot distinguish a better model from a luckier one.

## What was tested and rejected

| Tried | Result |
|---|---|
| Mean-stress (Goodman-style) correction | Worse at every correction strength |
| Finer exponent search around m = 5 | Within noise; kept the clean m = 5.0 |
| Endurance limit / bilinear two-slope S-N curve | +0.0007, paired t = 1.55 — not significant |
| Cycle-binning conventions (ndigits, binsize, nbins; three bin-edge choices) | No convention reproduces the labels |
| Counting the rainflow residue as full rather than half cycles | ASTM's 0.5 is already optimal |

The log-log slope of damage against `S(m)` is 0.9965, 0.79 standard errors from 1, so Miner's
rule with a power-law S-N curve holds essentially exactly and the remaining ~2% shows no
correlation with any signal statistic tested.

## Limitation worth keeping in mind

All 64 training files are healthy operating conditions, so the model estimates **loading
severity**, not structural damage state. A high `D` means that segment saw a rough ride, not that
the carbody is failing.
