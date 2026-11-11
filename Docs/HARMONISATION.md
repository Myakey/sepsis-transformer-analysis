# Harmonisation: MIMIC-IV cohort ↔ PhysioNet 2019 (sets A and B)

Status: **mapping defined; measured table pending.** The numbers come from `01b_physionet_crosscohort` (Cell 5), which writes `reports/01b_harmonisation.json`. Paste the table and your unit decisions into the last two sections.

## 1 · Feature set

All cross-cohort models use the **intersection** of the MIMIC model inputs and the PhysioNet columns: 33 time-varying features.

| Group | Columns | Handling |
|---|---|---|
| Shared (33) | HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2, BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride, Creatinine, Bilirubin_direct, Glucose, Lactate, Magnesium, Phosphate, Potassium, Bilirubin_total, Hct, Hgb, PTT, WBC, Fibrinogen, Platelets | Same name in both cohorts; used as is, subject to the unit check below |
| MIMIC only (3) | GCS, UrineOut_mL, PF_ratio | Removed. Cross-cohort models are reduced-feature models retrained on MIMIC FIT; their MIMIC VAL score is reported next to the full-feature model |
| PhysioNet only (1) | TroponinI | Dropped (no usable MIMIC counterpart; never a model input) |
| Static (5) | Age, Gender, Unit1, Unit2, HospAdmTime | Direct mapping |
| Time | ICULOS | Never a default input; only the explicit ±ICULOS switch. (The Transformer still knows the hour through its positional encoding.) |
| Never inputs | SepsisLabel, HoursToOnset, every SOFA column | Forbidden-column guard |

Engineered channels that need GCS are dropped automatically for the reduced set.

## 2 · Labels

PhysioNet's `SepsisLabel` is 1 from 6 h before onset (Reyna et al. 2020). The notebook recovers the onset and rebuilds the project's label:

1. onset hour = first positive hour + 6;
2. `HoursToOnset` = onset − ICULOS;
3. label = 1 iff 0 ≤ HoursToOnset ≤ 6; all post-onset rows dropped.

The rebuilt label must equal the Challenge label on every pre-onset row; the notebook prints that agreement (expected 1.0000).

Known differences between the cohorts:

- **Exclusions.** The Challenge excluded patients with onset < 4 h after ICU admission and records shorter than 8 h. The MIMIC cohort has many such stays. Primary comparison = matched stratum (stay ≥ 8 h; non-septic or onset ≥ 4 h; onset not censored); all stays are reported beside it.
- **Censored onset.** A PhysioNet record that is positive on its first row has its true onset somewhere in hours 4–7. Such stays are flagged, kept in the all-stays view and excluded from the matched stratum.
- **Label pipelines.** Both are Sepsis-3, derived by different code from different raw tables. An external drop therefore mixes label shift with covariate shift.

## 3 · Sites

- **Set B = Emory.** The clean external test.
- **Set A = Beth Israel Deaconess**, the hospital behind MIMIC. A temporal / same-site check with possible patient overlap. Always reported separately with this caveat.

## 4 · What is never fitted on external data

Normalisation statistics, calibration and thresholds all come from the training cohort (MIMIC FIT / MIMIC VAL in direction 1). Recalibration on a 10 % external split is reported as a labelled secondary result only.

## 5 · Measured table (fill in from `reports/01b_harmonisation.json`)

Per feature: share of hours with a measurement and median, for MIMIC FIT / set A / set B, plus the flag.

_pending first run of `01b`_

## 6 · Unit decisions (fill in)

For each feature flagged `CHECK UNITS`: unit difference (multiplier entered in `UNIT_FIXES`) or genuine population difference (left as is), with one line of reasoning.

_pending_
