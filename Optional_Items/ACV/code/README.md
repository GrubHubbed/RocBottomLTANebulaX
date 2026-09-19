# ACV Thermal Integrity Depot Console

Nebula X Hackathon 2026 — Problem Statement 3, ACV subsystem.

Each input workbook is one 8-car trainset carrying continuous ACV telemetry at
30 s intervals. Columns are named `Car <NN> - <parameter>` and appear in random
order. Exactly one car per file has a refrigerant leak. The console ranks all
eight cars from most to least likely faulty and explains why.

**Result on the labelled set: mean score 1.000, six of six at rank 1**, with
every ranking string matching the golden table byte for byte.

---

## 1. The metric

```
shortfall(car) = mean( cabin temperature - cooling setpoint )
                 over rows where that car's unit is validly cooling
cars sorted by shortfall, descending
```

Subtracting each car's **own** setpoint is essential, not cosmetic. In
`acv_case_01` cars 01–02 target 23.83 °C while 03–08 target 24.16 °C, so raw
cabin temperature promotes car 03 for being warm when it was merely told to be
warm.

Where the schema exposes **per-circuit** refrigeration data the diagnostic
changes — see §3.

There is no supervised model anywhere. Six labelled cases would overfit
catastrophically; this is a physically-motivated heuristic, and the labels are
read only by `validate.py`.

## 2. Two ranking bases, never competing

Gated behind the single constant `RANKING_PREFER_ASYMMETRY` in `acv_core.py`:

| Condition | Basis | Tie-break |
|---|---|---|
| per-circuit data for ≥ 2 cars | `circuit_asymmetry` | thermal shortfall |
| otherwise | `cooling_shortfall` | see §6 |

`ranking_basis` is reported in every result, in the console, in the audit JSON
and in `validate.py`, so a silent switch is visible.

## 3. Circuit asymmetry, and the trap it avoids

`acv_case_04` (63 parameters per car) carries two independent refrigeration
circuits per car. On that file **every whole-car indicator names the wrong car**:

| car | suction P (sys 1/2) | compression ratio | compressor duty | shortfall |
|---|---|---|---|---|
| **01 (truth)** | 541 / 554 | 3.41 / **2.89** | 0.85 / **0.52** | −0.075 |
| 02 | 547 / 542 | 3.35 / 3.35 | 0.86 / 0.87 | −0.178 |
| 03 | 558 / 560 | 3.29 / 3.21 | 0.81 / 0.82 | −0.175 |
| 04 | **427 / 395** | **4.84 / 4.98** | **0.97 / 0.97** | **+0.208** |

Ranked by suction pressure, compression ratio, compressor duty, cabin
temperature or cooling shortfall, the answer is car 04 every time — and car 04
is healthy. Whole-car pressure, ratio and duty are therefore **not implemented
as ranking features**.

A refrigerant leak is localised to one circuit's pipework: it starves that
circuit while the healthy one takes up the load. Car 04 is *symmetrically*
stressed — both circuits at 97 % duty, both ratios near 4.9 — which is a
whole-car condition (thermal load, fouled condenser, different setpoint), not a
leak. Car 01 is *asymmetric*: System 2 at 0.52 duty against System 1's 0.85,
with 243 units lower head pressure, while every sibling's two circuits match
each other. A car's own two circuits share the same cabin, setpoint, weather and
duty cycle, so they are a far tighter control group than the other seven cars.

```
asymmetry(car) = mean over channels of
                   |mean(circuit 1) − mean(circuit 2)| / mean(|c1|, |c2|)
channels: low pressure, high pressure, compressor duty, solenoid valve open
          fraction, and compression ratio when both pressures exist
```

Reproduced on `acv_case_04`:

| car | asymmetry | low P | high P | duty | solenoid | ratio |
|---|---|---|---|---|---|---|
| **01** | **0.27** | 0.02 | 0.14 | **0.48** | **0.54** | 0.17 |
| 04 | 0.03 | 0.08 | 0.05 | 0.00 | 0.01 | 0.03 |
| 03 | 0.03 | 0.00 | 0.02 | 0.01 | 0.10 | 0.02 |
| 02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.00 |

Car 01 leads by 8.2× overall and 37.9× on duty alone.

> **Caveat.** This basis is validated on **one file only**, because
> `acv_case_04` is the sole case carrying per-circuit data. The physics is sound
> and the margins are large, but **n = 1**. Treat it as reasoning, not as a
> validated accuracy claim.

The console also displays per-circuit duty by quarter of the recording. Car 01
reads `Q1 0.93/0.34  Q2 0.89/0.43  Q3 0.90/0.36  Q4 0.67/0.95` — the starved
circuit recovering in the final quarter, consistent with a recharge mid-window.
Siblings are flat. **This is a display feature and does not affect ranking.**

## 4. Two-phase schema handling

**Phase 1 — sniff.** `openpyxl.load_workbook(read_only=True, data_only=True)`,
then the header row plus 5 data rows and nothing else. The largest sheet is
chosen from `max_row × max_column` metadata, never by loading it. Roles are
resolved per car by exact alias first, then keyword patterns with negative
keywords, preferring a column that has data over an empty one.

Sniffing the 483-column workbook takes **0.06 s**.

**Phase 2 — score.** The sheet is streamed once and only the columns that earned
a role are materialised — 115 of 483 on `acv_case_04`.
`pandas.read_excel(usecols=)` is deliberately avoided because it still parses
every cell.

Rejected by design: the offset-selection flags `Target Temperature −2K/−1K/0/
+1K/+2K` are booleans, not temperatures, so they never take the `reference`
role; `Observation Area Temperature` is a different measurement point, so it
never takes `outcome`.

### Guards, each from an observed failure

| Guard | Why |
|---|---|
| **Null-like strings** | `acv_case_02` writes the literal `'None'` into empty cells. `none, nan, null, na, n/a, -, --, ?` are treated as missing. Without this, 3 of 5 sampled values look categorical, numeric roles are rejected, and the file silently produces eight "no data" cars. |
| **Bounded sample escalation** | If a *required* role is unresolved for any car after 5 rows, re-sniff **once** at 60 rows. The whole file is never read to resolve schema. |
| **Non-destructive filters** | An optional filter is applied only if it leaves at least `max(20, 5 % of rows)`. A filter that empties the data is the wrong filter for that schema, so it is dropped and logged as `SKIPPED <filter> — would leave N rows`. |
| **Degraded roles** | A sensor that names its role but writes a fault string (`'Invalid'` in place of a temperature) still takes the role, marked `degraded`. Otherwise the column is never read and a dead probe cannot be reported at all — this is what surfaces the six dead outdoor probes in cases 05 and 06. |
| **No NaN in a sort key** | A car with no usable data gets an explicit status (`no_data` / `unreliable`), never a NaN score. Pandas sorts NaN last, which can manufacture a plausible-looking ranking out of nothing. |

## 5. Corroboration (never the primary signal)

**Peak-load window (§4F).** The shortfall recomputed over the top decile of
outdoor temperature. An undercharged circuit copes on a mild night and fails
under thermal load, so this roughly doubles the separation margin — but it
weakens `acv_case_02`, so it is shown as an "agrees / disagrees" indicator and
never ranked on.

**Cross-car z-score (§4C).** `mean |z|` per car across parameters shared by all
cars. The margin `anomaly(top) / anomaly(second)` is reported: ≥ 1.25 is a
confident single-car call, 1.08–1.25 probable with a close second, < 1.08 a
shortlist rather than a call. Outdoor temperature is excluded (weather is shared
by the whole train, so it is noise).

## 6. Tie-breaking — a documented deviation

§4D of the brief specifies that shortfalls within 0.05 °C are broken by (a) the
cross-car z-score and (b) telemetry dropouts. **Implemented as specified, but it
declines to fire on this dataset, and that is deliberate.**

Applying it naively reorders five of the seven files and breaks the golden
table. The reason is visible in the audit: in the 8-parameter schema the only
cross-car numeric parameters are the cabin temperature and the cooling setpoint.
The setpoint is a value the car was *commanded*, not one it measured, so it is
excluded for the same reason outdoor temperature is — leaving exactly **one**
parameter, which is the numerator of the shortfall itself and therefore cannot
be an independent tie-breaker. The heating setpoint is flat for every car and
contributes no variance.

So the z-score is allowed to break a tie only when it rests on at least
`MIN_Z_PARAMS_FOR_TIEBREAK` parameters *and* clears the brief's own
`MARGIN_PROBABLE` (1.08) threshold; the dropout fallback requires a difference
of at least 5 % of the recording, matching the materiality used by the
sensor-health rules. On the labelled set both correctly decline, the audit logs
that they declined, and the thermal ordering stands. On a richer schema
(`acv_case_04`, 9 contributing parameters) the z-score is live and independently
ranks car 01 first, corroborating the asymmetry call at a 1.08× margin.

This resolves a genuine conflict in the brief: §4D asks for a tie-break, while
§8 and §10 make the exact ranking strings the contract and say a rank reached by
a different ordering is a failure. The strings win; the mechanism stays, gated.

## 7. Sensor-health alerts

Missing telemetry is a maintenance finding in its own right, and severity here
is separate from the leak ranking.

| # | Pattern | Alert | Fires on |
|---|---|---|---|
| 1 | every parameter of a car 0 % populated, siblings ≥ 95 % | **CAR OFFLINE** | `case_04` cars 05–08 |
| 2 | one parameter 0 % for some cars, ≥ 2 siblings normal | **SENSOR DEAD** | `case_05`/`06` outdoor probe, 6 cars |
| 3 | one parameter 5–90 % while siblings report throughout | **SENSOR INTERMITTENT** | — |
| 4 | `information_valid` invalid for exactly one car, zero for all others | **SELF-CHECK DROPOUTS** | `case_01` car 01 (25 of 6 999) |
| 5 | a parameter's header absent for some cars | **CONFIGURATION DIFFERENCE** (not a fault) | `case_04`, 4 auxiliary-compressor readings |
| 6 | a parameter flat for **all** cars, whole recording | **FUNCTION DISABLED** (not a fault) | `case_05`/`06` heating setpoint |
| 7 | a regime filter would leave < 5 % of rows for every car | **SCHEMA MISMATCH** | — |
| 8 | fewer than 2 cars scoreable | **FILE NOT RANKABLE** | — |
| 9 | fleet median shortfall > +0.50 °C | **FLEET-WIDE COOLING SHORTFALL** (advisory) | `case_06` (+0.92 °C) |
| 10 | outdoor sensor not reporting, so the peak-load cross-check cannot run | **PEAK-LOAD CORROBORATION UNAVAILABLE** (advisory) | `case_05`/`06`, 6 cars each |

Alerts carry three levels — **fault**, **advisory**, **informational** — and are
listed in that order. An advisory is a whole-train condition or a missing
cross-check, not a unit fault.

Each message says in plain sentences what was seen, what it probably means
mechanically, what the depot should do, and whether the leak ranking is
affected. No jargon, no z-scores, no column names in the headline.

## 8. Severity bands — fleet-relative (presentation only)

Severity is measured against **the trainset's own fleet**, not against an
absolute temperature. Absolute bands calibrated on car model A flagged all eight
cars on `acv_case_06` — model C, early July 2020, fleet-mean setpoint 22.5 °C,
the coldest in the dataset. Every unit there sits near capacity and the fleet
median shortfall is +0.92 °C, so a fixed threshold indicts the whole train and
buries the one car that is genuinely anomalous.

```
med = median(metric over the scored cars)
mad = max(median(|metric − med|), floor)
z   = (metric − med) / mad
```

Median and MAD rather than mean and standard deviation, so the one faulty car
cannot drag the baseline toward itself.

| Band | Fleet z | Action |
|---|---|---|
| CRITICAL | ≥ 8 | Withdraw at next depot entry; pressure-test the circuit |
| HIGH | ≥ 4 | Schedule refrigerant check within 48 h |
| ELEVATED | ≥ 2 | Watchlist; re-read after next service day |
| NORMAL | < 2 | No action |
| NO DATA | — | Check the data feed before judging |

The metric is whichever one the file was **ranked** on — shortfall, or circuit
asymmetry on a per-circuit schema — so the displayed number can never contradict
the displayed order.

The MAD floor is per-metric, because the two metrics carry different units:

| Metric | Floor | Why |
|---|---|---|
| cooling shortfall | 0.05 °C | a tight fleet would otherwise produce absurd z values |
| circuit asymmetry | 0.005 (dimensionless) | the real MAD on `acv_case_04` is 0.011; a 0.05 floor would swamp it and pull car 01 from z 22.3 down to 4.8, under-reporting a leak that is 8.2× clear of the next car |

Result on the labelled set — in every case the true faulty car has the highest
z in its file:

| case | fleet median | cars flagged | truth's z |
|---|---|---|---|
| 01 | −0.205 | 01 CRITICAL, 02 HIGH | 13.30 |
| 02 | +0.202 | 02 HIGH, 03 ELEVATED | 5.50 |
| 03 | +0.319 | 03 HIGH, 02 ELEVATED | 7.46 |
| 04 | +0.032 (asym) | 01 CRITICAL | 22.32 |
| 05 | −0.924 | **none** | 1.83 |
| 06 | +0.919 | **06 CRITICAL only** | 15.80 |
| test | −0.084 | 01 HIGH | 4.18 |

**Because z is a monotonic transform of the metric within a file, the within-file
order is mathematically identical — which is why banding cannot affect any
ranking.** Ranking still sorts on raw shortfall / raw asymmetry, and
`validate.py` asserts that the scored cars remain in raw-metric descending order
on every file.

### Prime suspect

Rank 1 is always marked, whatever its band — severity answers *how bad*, rank
answers *which one*. On `acv_case_05` nothing is flagged, so without this the
screen would show eight unremarkable tiles and hide the model's actual call. The
qualifier comes from the margin:

| Margin (shortfall basis) | Marker |
|---|---|
| ≥ 0.30 °C | `PRIME SUSPECT · clear lead` |
| 0.10–0.30 °C | `PRIME SUSPECT · narrow lead, corroborate` |
| < 0.10 °C | `PRIME SUSPECT · statistically tied with rank 2, treat as a shortlist` |

On the asymmetry basis the same question is a **ratio**, not a temperature: car
01 on `acv_case_04` leads by 0.237 absolute but by 8.2×, so it is judged at
≥ 2.0× clear / ≥ 1.25× narrow. Judging a ratio against a Celsius threshold would
mislabel a decisive lead as "narrow".

`acv_case_05` reads as a shortlist (margin 0.021 °C), which is the honest answer.

**No band, colour, icon, alert or view control can reorder a car.** The
submission CSV is built from the same `ranked_cars` string the engine returned.
Every severity carries an icon and a written label as well as a colour, and
no-telemetry carriages are rendered off the red-amber-green scale entirely
(dashed border, violet) so they can never read as "nothing to report".

## 9. Validation

```
file                 basis              ranked_cars              truth  rank   score   margin
acv_case_01.xlsx     cooling_shortfall  01|02|03|04|07|08|05|06    01     1    1.000   +0.528
acv_case_02.xlsx     cooling_shortfall  02|03|07|08|06|01|04|05    02     1    1.000   +0.105
acv_case_03.xlsx     cooling_shortfall  03|02|01|07|04|08|05|06    03     1    1.000   +0.450
acv_case_04.xlsx     circuit_asymmetry  01|04|03|02|05|06|07|08    01     1    1.000   +0.237
acv_case_05.xlsx     cooling_shortfall  04|02|07|01|06|03|08|05    04     1    1.000   +0.021
acv_case_06.xlsx     cooling_shortfall  06|08|04|02|03|05|01|07    06     1    1.000   +1.200
MEAN                                                                6/6    1.000
acv_test_case.xlsx   cooling_shortfall  01|03|04|07|08|06|02|05
```

All seven strings match the brief's golden table exactly, and the separation
margins reproduce its golden intermediates (+0.528, +0.105, +0.450, +0.021,
+1.200).

`validate.py` exits non-zero on failure and asserts three layers:

1. **Output** — the exact `ranked_cars` string and `ranking_basis` per case.
2. **Intermediate** — per-car shortfall and asymmetry to 3 dp, eligible row
   count per car, columns-read count, the column chosen for each role, and the
   alerts raised. *This is the layer that matters most.* A pipeline can produce
   a correct final score from completely empty data: if every shortfall came out
   NaN, pandas would sort the NaNs into car order `01|02|...|08`, and because
   `case_04`'s true faulty car happens to be 01 it would print score 1.000. Ranks
   alone would pass that. Row counts catch it instantly.
   It also asserts the display layer: the fleet median and MAD, every car's
   fleet z and band, the prime-suspect marker and its qualifier, that
   `case_06` flags exactly one car while `case_05` flags none, that the truth
   car has the highest z in all six labelled cases, and — critically — that the
   scored cars are still in raw-metric descending order on every file, so no
   severity or z value has leaked into a sort key.
3. **Guard** — deliberately broken copies of a real file: one car blanked
   entirely (expects CAR OFFLINE, car still ranked, no fabricated score), the
   literal string `'None'` substituted for every blank (expects the ranking to
   be unchanged), and a running-mode vocabulary matching no cooling state
   (expects the filter dropped and logged, not applied).

## 10. Rejected features — do not reintroduce

| Feature | Result |
|---|---|
| shortfall drift over time, across cars | rank **8** on cases 03 and 06 |
| fraction of time in `Full Cooling` | rank 6 and 7 on cases 05 and 06 |
| whole-car suction pressure / compression ratio / compressor duty | names car 04 on `case_04`; truth is car 01 |
| outdoor temperature as a z-score feature | shared by the whole train |
| heating setpoint level | flat for every car; no per-car variation |
| unsigned `mean abs(z)` as the only signal | misses case 05, where the faulty car is the *least* cold |

Note the asymmetry: *per-circuit* pressure and duty are the best signal
available; *whole-car* pressure and duty are actively misleading. Same columns,
opposite conclusions, depending on whether you compare within a car or across
cars.

## 11. Running it

```bash
pip install -r requirements.txt
```

Regression harness (exits non-zero on failure):

```bash
python validate.py
```

Submission CSV:

```bash
python predict.py --input data --output acv_predictions.csv --audit-dir audit
```

The console:

```bash
streamlit run app.py
```

Expected layout (unchanged by this project):

```
data/Train/acv_case_01..06.xlsx
data/Test/acv_test_case.xlsx
data/Train_Labels.csv          read by validate.py only
```

## 12. Known limits

- **Circuit asymmetry is validated on one file.** `acv_case_04` is the only case
  with per-circuit data. See the caveat in §3.
- **Runtime.** `acv_case_04` is a 33 MB workbook with 10.7M cells and takes
  roughly 2 minutes; the other files take 5–15 s. The cost is openpyxl's XML
  parsing, not the ranking. A direct sheet-XML reader was prototyped and
  discarded: it gained only ~1.1× on that file while adding a second reader that
  could silently diverge from the first. The console caches on file bytes, so
  re-rendering never re-runs the model.
- **Columns read.** 115 of 483 on `acv_case_04`; 61–67 of 67 on the
  8-parameter files. The 8-parameter schema has only 8 parameters per car and
  seven of them earn a role, so reading nearly all of them is expected — the
  selectivity that matters is on the wide schema.
- **`SELF-CHECK DROPOUTS` has no minimum count.** §5 specifies the pattern as
  "invalid on some rows for one car and zero rows for all others", so
  `acv_case_03` raises it for a *single* stray reading out of 8 310. That is
  faithful to the spec and the count is stated plainly in the message, but a
  depot may want a materiality floor before it becomes a work order. No such
  floor was invented here, because suppressing it would deviate from the spec.
- **`FILE NOT RANKABLE` and `SCHEMA MISMATCH` are untested against real data.**
  No supplied file triggers them; they are exercised only by synthetic guards.
- **The outdoor probe is dead on six of eight cars in cases 05 and 06.** Only
  cars 01 and 08 report a usable outdoor temperature (6 298 and 2 815 readings);
  cars 02–07 write the literal string `'Invalid'` for the whole recording. Peak-
  load corroboration is therefore unavailable for the prime suspect on both of
  those trainsets, which the console now states plainly rather than silently
  producing nothing.
- **Six labelled cases.** Every threshold here is physical or structural rather
  than fitted, but the sample is small and the z bands in §8 are calibrated on
  it.
