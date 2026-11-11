# Decisions log (v4)

Design decisions and judgement calls, with date and who made them. Deviations from pre-registered rules are in `DEVIATIONS.md`.
Items marked **→ PREREG** must be copied into `Docs/PREREGISTRATION.md` before it is frozen (start of Phase 3).

## 2026-11-11 · Phase 0 answers (Miaki)

| # | Decision |
|---|---|
| A1 | **Gate** adopted as in `DEVIATIONS.md` D1. Both arms must pass. |
| A2 | **Shuffled-label control. → PREREG** Primary band member stays the v3 protocol (aux head on). Pre-registered reading of the primary band: *"sepsis labels add nothing detectable beyond physiology learning."* An aux-off shuffled-label control and a budget-matched one are reported beside the band; the aux-off control decides whether the broader S1 claim (nothing beyond architecture and inputs) also holds. The band is never moved. |
| A3 | **Carry-forward. → PREREG** Top-6 heads by \|b1\| on DISCOVERY, re-ranked for each v4 model by the same rule, labelled "carried forward without passing the screen". Layer-1 heads are reported but flagged: they cannot show effects on other hours. |
| A4 | **ICULOS.** Every ±ICULOS result is described as "explicit ICULOS input on/off". The lecturer notebook and slides state that the Transformer always knows the hour (positional encoding of the absolute hour, GRU-D Δ for never-measured features, number of visible keys). One time-blind LightGBM (no `hrs_since` for never-measured features) is added. |
| A5 | **KA-Transformer: no replication of the paper's protocol.** The waterfall and storyline S4 are dropped: the paper makes one prediction per patient with length of stay as an input, a different task. (a) `KATransformer` under our protocol is the lecturer's comparison model: 8 heads × 64 = 512, 4 layers, FFN 2048, dropout 0.2, causal mask, per-hour last-position readout, our labels and horizons, from hour 1, frozen split, ±ICULOS, balanced variant. (b) Kernel attention is an **exploratory** third attention variant inside our trunk: z_tj = −β_h‖q_t − k_j‖², learned β per head, rows normalised (`kernel`), unnormalised as a sensitivity run (`kernel_raw`); same embedding, init, hyperparameters, controls and DISCOVERY/SCREEN stays, 3 seeds, same P1–P4 screen; kept out of the pre-registered primary contrast. (c) The paper's reported numbers are cited only as context, marked "different task, not comparable". |
| A6 | **CONF split. → PREREG** By subject into CONF-A (`04`) and CONF-B (`05`), fixed when the pre-registration is frozen. |
| A7 | **References.** §10 (R1–R31) is in force. Phase 1 needs only R1–R8; all eight are in `Resources/`. |
| A8 | **Drive path.** One constant, `PROJECT_ROOT`, in each notebook's first cell; a clear error if it is not mounted. See open item O2. |
| A9 | **Sigmoid fallback. → PREREG** If the sigmoid arm fails the gate: learning-rate-only tuning (untuned run also reported), plus a secondary matched pair with LayerScale in **both** arms. The unmodified pair stays primary. |
| A10 | **Seed rule. → PREREG** "Pattern present" across 5 seeds is decided by a rule fixed in advance (per-seed count of heads beyond the band, plus a sign test across seeds), because head indices are not aligned across seeds. Exact thresholds are set in the pre-registration. |
| A11 | **Early hours. → PREREG** Δz is the primary cross-arm quantity. Queries are analysed from t ≥ 8 (v3's `CORR_MIN_T`). The 1/(1+T) total-mass difference is reported by query position. |
| A12 | **Heterogeneity. → PREREG** Any I², τ or prediction-interval claim uses cluster-robust stage-1 errors. The v3 estimator stays the primary decision rule for the band. |
| A13 | **PhysioNet. → PREREG** Matched stratum (onset ≥ 4 h, stay ≥ 8 h) is primary; all stays are reported beside it. |
| A14 | **EXT.** One frozen-model table on EXT at the very end, nothing before. |

## 2026-11-11 · Calls made while building the core (Claude; accepted unless you object)

| # | Call | Why |
|---|---|---|
| C1 | `MAX_SEQ_LEN = 336`: one window per stay. | Longest stay in MIMIC and both PhysioNet sets is 336 h. Full coverage with full history, no stitching. The v3 "most recent 168 h" view is a row mask (`metrics.v3_view_mask`). |
| C2 | Evaluation batches shrink for long stays so B·T² never exceeds v3's peak. | T doubled; a fixed batch of 128 would need 4× the attention memory on a T4. Training batches are untouched (they must be identical across arms). |
| C3 | Class cells are **generated**, not retyped. `v3_reference.py` is extracted from the v3 notebook by script with a sha256 per cell; `user_classes.py` is produced from it by two exact text patches (`_make_user_classes.py`). The Preprocessor is extracted the same way (`v3_preprocessor.py`). | Makes "verbatim" checkable: test 4 compares every definition and, in Colab, re-hashes the notebook cells. |
| C4 | The arm is switched **after** construction (`models.set_arm`). | All arms consume the same random numbers and start from the same tensors; `SingleLayer` and `TransformerEncoder` stay verbatim. |
| C5 | Kernel β initialised at 1/(2√head_dim), stored as log β. | ‖q−k‖² = ‖q‖² + ‖k‖² − 2q·k and ‖q_t‖² cancels within a softmax row, so with this β the q·k term **is** v3's scaled dot product at step 0. The only new ingredient is the key-norm penalty. |
| C6 | `KATransformer` adaptations: post-LN layers as drawn in the paper's Fig. 3; inputs are forward-filled z-scores, 4 static scalars and gender; the observed-mask is an extra input, on by default (`use_mask`), because our model sees it and the paper says nothing about it; ICU length of stay becomes the ±ICULOS switch. | Each is a reading of an under-specified paper. Training hyperparameters (the paper's lr 1e-5 / batch 256 / 100 epochs vs our engine) are decided in Phase 2 and logged then. |
| C7 | Weight EMA is not carried into v4. | It was off in the v3 winner (`ema = 0`). |
| C8 | Budget-matched controls run a fixed number of epochs and take the last epoch (`fixed_epochs=True`). | Early stopping on VAL AUROC is meaningless for a model trained on shuffled labels. |
| C9 | The untrained control is the trained model's own step-0 weights, saved as `init.pt` in every run folder. | "Same init", and it permits a per-head trained-minus-init contrast. |
| C10 | Sequence-model balancing = repeat positive-containing FIT windows; SMOTE proper only for LightGBM row features. | Rationale paragraph for the lecturer is the docstring of `sepsis_core/balancing.py`. |
| C11 | v4 run artifacts go to `MyDrive/sepsis_v4` (`Paths.WORK_DIR`). | Same pattern as v3's `sepsis_softmax_pass3`; checkpoints are too large for the project folder. See O3. |
| C12 | Reserved folds are enforced in code (`data.FoldGuard`): training stages cannot load LOC/CONF/EXT; any stage needs `Docs/PREREGISTRATION.md` **and** a `Docs/PREREGISTRATION.frozen` stamp before touching them. | Turns non-negotiable 2 into something that fails loudly. |

## 2026-11-11 · Phase 1 result and repo (Miaki)

- `00_core_tests` is green on synthetic data and on real FIT/VAL (10/10), including the bitwise match to the v3 classes and the notebook-hash check.
- Code lives in `github.com/Myakey/sepsis-transformer-analysis`. Every notebook clones it and imports `sepsis_core` from the clone. **The git commit hash is the core version**; it is written into every checkpoint record, the manifest and MLflow. Pin `REPO_REF` to a commit hash for any run that will be reported.
- MLflow logs to DagsHub only (Colab secrets). There is no local fallback.

## 2026-11-11 · Calls made while building Phase 2 (Claude; accepted unless you object)

| # | Call | Why |
|---|---|---|
| C13 | **KA-Transformer is trained with our recipe** (v3 winner's optimiser, learning rate, batch size, focal loss, schedule, early stopping, soup), without the auxiliary head. The paper's lr 1e-5 / batch 256 / 100 epochs is not used. | "Their architecture, my protocol": with one training recipe the comparison isolates the architecture. The paper's settings were chosen for one sample per patient and would under-train an hourly model in 24 epochs. |
| C14 | **Budget-matched shuffled-label controls** use the primary run's seed (so the same initial weights), the same 24-epoch cosine schedule, and stop after exactly as many epochs as the primary ran; the last epoch is taken. | Matches initial weights, learning-rate trajectory and number of updates. Stopping on VAL AUROC is meaningless for shuffled labels. |
| C15 | Three shuffled-label controls per primary arm: v3 protocol (band member), budget-matched, budget-matched with the aux head off. The exploratory kernel arm gets the v3-protocol control only. | v3 → budget-matched isolates the training budget; budget-matched → aux-off isolates the physiology-forecasting task. |
| C16 | Warmup control: softmax with 2× warmup on 3 seeds. | Enough to show whether warmup length alone moves AUROC by more than the seed SD. |
| C17 | Sigmoid fallback grid: learning rate × {0.5, 2, 4} at the primary seed; LayerScale initialised at 1e-4 in both arms. Runs only if the sigmoid arm fails the gate. | Learning-rate-only tuning as pre-registered; 1e-4 is the LayerScale initialisation Ramapuram et al. report using. |
| C18 | LightGBM SMOTE adds synthetic positives up to 25 % of the negatives (`SMOTE_RATIO`), not 1:1. Binary "was measured" features are interpolated like any other feature. | 1:1 needs several GB more RAM than Colab has for 2.3 M rows. The fractional-flag artefact is the standard SMOTE limitation and is stated in the notebook. |
| C19 | LightGBM gets a **time-blind** variant: time-since-measurement is set to its cap for features never measured so far. | v3's "no ICULOS" trees could still read the hour through that channel. Rolling 6 h windows still shorten in a stay's first hours, so a little time information remains; this is stated, not hidden. |
| C20 | PhysioNet onset = first positive hour + 6. A record positive on its first row has a censored onset (true onset in hours 4–7): flagged, kept in the all-stays view, excluded from the matched stratum. The matched stratum is defined identically for both cohorts on the processed data (length ≥ 8 h after dropping post-onset rows; non-septic or onset ≥ 4 h; not censored). | Follows from the Challenge's label rule and exclusions (Reyna et al. 2020). |
| C21 | Directions 2 (PhysioNet → MIMIC) and 3 (A ↔ B) run without the explicit ICULOS input and with fewer models (softmax + KA + LightGBM; softmax + LightGBM). Direction 1 runs softmax, sigmoid, KA and LightGBM, each ±ICULOS. | Keeps the notebook to a few GPU-hours; direction 1 is the primary one. |
| C22 | A model name means one model across notebooks (`softmax_s42` is trained once and reused by `01`, `01b` and `02`). | No silent duplicates with different weights. |
| C23 | Not built in Phase 2: the PhysioNet-2019 utility score (brief: "if it's cheap"), the optional v3 ladder re-run, and the attention replication on set B (needs the P1–P4 battery, Phase 3). | Each needs its own tested code. |
| C24 | `01b` stops after the harmonisation table until `HARMONISATION_REVIEWED = True`. Unit conversions are entered by you as `UNIT_FIXES`, never guessed by the code. | "Verify units before anything runs." |

## 2026-11-11 · Back to v3's window (Miaki)

| # | Decision |
|---|---|
| C25 | **C1 is reversed.** Default is again v3's view: `MAX_SEQ_LEN = 168`, and each stay contributes its most recent 168 hours for both training and evaluation (`CFG.COVERAGE = "last"`). Reason: the first full run under the 336 h full-coverage setting gave about 0.73 VAL AUROC against v3's 0.807. **The cause is not yet established.** v3's final training also used all 45,685 FIT stays, so the number of stays did not change; what changed was the window (168 → 336 h) and training on every hour of long stays. If `softmax_s42` returns to about 0.807 under this setting, the long window was the cause; if not, the cause is elsewhere. Consequences: 95.6 % of stays are still predicted from ICU hour 1; for the 4.4 % longer stays the hours before the last 168 are not scored. Full coverage stays available as `COVERAGE = "full"` with `MAX_SEQ_LEN = 336`, for a labelled side analysis only. Run folder renamed to `v4_last168` so nothing trained under the old setting is reused. LightGBM is scored on the same rows as the Transformer, as in v3. |
| C26 | The per-epoch attention-mass probe now restores the global random state, so training consumes the same random stream as v3's loop. |
| C27 | `runs.setup` refuses a full run while the configuration is still in smoke mode from an earlier run in the same runtime. |

## Open items

| # | Item | Needed by |
|---|---|---|
| O1 | **The PDF's "placebo set j < k is exactly zero" holds in layer 1 only.** For a query t > k in layers ≥ 2, q_t has already read hour k, so Δz_tj = Δq_t · k_j is not zero even for j < k (test 6 measures it). What is exact: (i) layer 1, all j ≠ k; (ii) any layer, queries t < k. For j < k in deeper layers the key side is unchanged, so that set isolates the **query-side** change; it is a useful comparison set but not a noise floor. The P5–P7 nulls need rewording before the pre-registration is frozen. | Phase 3, before freeze |
| O2 | `PROJECT_ROOT` was left as the placeholder in your reply. The notebooks default to `/content/drive/MyDrive/Sepsis Attention Research`; confirm or correct it. | Before you run `00` |
| O3 | Confirm `MyDrive/sepsis_v4` as the v4 artifact folder (it is outside `Notebooks/v4/` and `Docs/`, but so were v3's checkpoints). | Before you run `02` |
| O4 | Reference conflicts between §10 and the methodology PDF: R20 author list (§10: Makelov, Lange & Nanda; PDF adds Geiger), R22 venue (§10: arXiv; PDF: NeurIPS 2025), R31 pages (2600–2606 vs 2600–2605), R1 authors ("Unknown" in §10; the PDF in `Resources/` gives Zhu, Cheng, Zhang, Zhang, Hong, Wang & Xiao, Discover Applied Sciences 7:218). | Phase 4 |
| O5 | PDFs to add before Phase 2: **R10** (Chawla et al. 2002, SMOTE). Two papers the agreed design relies on are not in §10; proposed for addition: Touvron, H., Cord, M., Sablayrolles, A., Synnaeve, G., & Jégou, H. (2021). Going deeper with Image Transformers. *ICCV 2021* (LayerScale, for A9); Wortsman, M., et al. (2022). Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. *ICML 2022* (checkpoint soup, cited in v3's code). Both `[unverified, PDF not provided]` until added. | Phase 2 |
