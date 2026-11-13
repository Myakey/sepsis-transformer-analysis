# Harmonisation: MIMIC-IV cohort ↔ PhysioNet 2019 (sets A and B)

Status: **reviewed 2026-11-13. No unit conversions (`UNIT_FIXES = {}`).** The numbers come from `01b_physionet_crosscohort` (Cell 5), which writes `reports/01b_harmonisation.json`. Paste the table and your unit decisions into the last two sections.

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

Source: `01b_physionet_crosscohort` Cell 5 (`reports/01b_harmonisation.json`). M = MIMIC-IV FIT rows; A = PhysioNet set A; B = set B. Measured share = fraction of hours with a value.

| Feature | Measured M | A | B | Median M | A | B | Flag |
|---|---|---|---|---|---|---|---|
| HR | 0.929 | 0.923 | 0.879 | 82 | 84 | 83 | |
| O2Sat | 0.904 | 0.880 | 0.859 | 97 | 98 | 98 | |
| Temp | 0.266 | 0.338 | 0.339 | 36.83 | 37.06 | 36.90 | |
| SBP | 0.869 | 0.848 | 0.861 | 119 | 118.5 | 124 | |
| MAP | 0.882 | 0.898 | 0.852 | 78 | 77 | 84 | |
| DBP | 0.868 | 0.519 | 0.861 | 62 | 58.5 | 65 | CHECK DENSITY (A) |
| Resp | 0.919 | 0.902 | 0.789 | 19 | 18 | 18 | |
| EtCO2 | 0.006 | 0.000 | 0.076 | 36 | n/a | 33 | |
| BaseExcess | 0.048 | 0.104 | 0.002 | 0 | 0 | −2.4 | |
| HCO3 | 0.066 | 0.081 | 0.002 | 24 | 24 | 23.4 | |
| FiO2 | 0.088 | 0.142 | 0.023 | 0.50 | 0.50 | 0.40 | |
| pH | 0.055 | 0.115 | 0.022 | 7.39 | 7.39 | 7.38 | |
| PaCO2 | 0.048 | 0.088 | 0.022 | 40.5 | 40 | 39 | |
| SaO2 | 0.020 | 0.050 | 0.019 | 95 | 97 | 97.5 | |
| AST | 0.014 | 0.015 | 0.018 | 43 | 57 | 32 | |
| BUN | 0.066 | 0.082 | 0.055 | 20 | 18 | 17 | |
| Alkalinephos | 0.013 | 0.015 | 0.018 | 82 | 78 | 70 | |
| Calcium | 0.058 | 0.050 | 0.068 | 8.5 | 8.3 | 8.2 | |
| Chloride | 0.076 | 0.083 | 0.006 | 104 | 106 | 107 | |
| Creatinine | 0.066 | 0.066 | 0.055 | 1.00 | 0.90 | 0.99 | |
| Bilirubin_direct | 0.001 | 0.001 | 0.002 | 1.4 | 1.4 | 0.2 | CHECK UNITS (B) |
| Glucose | 0.086 | 0.122 | 0.222 | 126 | 124 | 128 | |
| Lactate | 0.029 | 0.034 | 0.019 | 1.80 | 1.80 | 1.97 | |
| Magnesium | 0.063 | 0.078 | 0.048 | 2.1 | 2.0 | 2.0 | |
| Phosphate | 0.058 | 0.050 | 0.029 | 3.4 | 3.4 | 3.3 | |
| Potassium | 0.091 | 0.109 | 0.077 | 4.1 | 4.1 | 4.0 | |
| Bilirubin_total | 0.014 | 0.012 | 0.018 | 0.8 | 0.9 | 0.8 | |
| Hct | 0.082 | 0.118 | 0.058 | 29.9 | 30.2 | 30.5 | |
| Hgb | 0.071 | 0.088 | 0.059 | 10.0 | 10.4 | 9.9 | |
| PTT | 0.046 | 0.048 | 0.010 | 32.2 | 32.4 | 32.3 | |
| WBC | 0.062 | 0.075 | 0.053 | 10.0 | 10.8 | 9.6 | |
| Fibrinogen | 0.007 | 0.008 | 0.006 | 224 | 250 | 251 | |
| Platelets | 0.063 | 0.065 | 0.053 | 179 | 181 | 180 | |
| Age | 1.000 | 1.000 | 1.000 | 65 | 65.25 | 63 | |
| HospAdmTime | 1.000 | 1.000 | 1.000 | −3.23 | −2.60 | −8.95 | CHECK UNITS (B) |
| Gender | 1.000 | 1.000 | 1.000 | 1 | 1 | 1 | |
| Unit1 | 0.343 | 0.511 | 0.704 | 0 | 1 | 0 | CHECK DENSITY (B) |
| Unit2 | 0.343 | 0.511 | 0.704 | 1 | 0 | 1 | CHECK UNITS (A); CHECK DENSITY (B) |


## 6 · Unit decisions (fill in)

For each feature flagged `CHECK UNITS`: unit difference (multiplier entered in `UNIT_FIXES`) or genuine population difference (left as is), with one line of reasoning.

Decision: **no unit conversions** (`UNIT_FIXES = {}`), reviewed by Miaki and Claude on 2026-11-13.

| Feature | Flag | Decision and reason |
|---|---|---|
| Bilirubin_direct | CHECK UNITS (B): median 0.2 vs 1.4 | Population / ordering difference, not units. A µmol/L ↔ mg/dL error would be a factor of about 17 in the other direction (≈ 24). Direct bilirubin is measured in 0.1–0.2 % of hours and is mostly ordered when total bilirubin is already raised; ordering habits differ by site. |
| HospAdmTime | CHECK UNITS (B): −8.95 vs −3.23 h | Population difference, not units. A days ↔ hours error would be a factor of 24; set A (−2.60) matches MIMIC. Emory patients spend longer in hospital before ICU admission. |
| Unit2 | CHECK UNITS (A): median 0 vs 1 | Case mix, not units. Unit1 / Unit2 are 0/1 unit-type flags, so the median only shows the majority unit: set A is mostly the unit coded Unit1, MIMIC and set B mostly Unit2. **Assumption to confirm:** the MIMIC cohort build codes Unit1 / Unit2 with the same meaning as PhysioNet (MICU / SICU). |

Checked and consistent across cohorts: FiO2 is a fraction (0.40–0.50) and Temp is °C (36.8–37.1) everywhere.

**Measurement-density shifts** (not fixable; they test whether missingness patterns transfer, see the shortcut diagnostics in `01b` Cell 9):

- DBP is measured in 52 % of set A hours vs 87 % in MIMIC.
- Set B rarely charts Chloride, BaseExcess, HCO3 and PTT (0.2–1 % of hours vs 5–8 % in MIMIC) and measures Glucose far more often (22 % vs 9 %).
- Unit flags are present for 34 % of MIMIC stays, 51 % in set A and 70 % in set B.
- Set A charts more blood gases (pH, PaCO2, BaseExcess, FiO2) than MIMIC.

