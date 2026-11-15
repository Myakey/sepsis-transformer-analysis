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

## 2026-11-15 · Start of Phase 3 (Miaki)

| # | Decision |
|---|---|
| A15 | **Reference substitutions in §10.** R13 (Aitchison 1982) is replaced by Greenacre, M. (2021). *Compositional data analysis — linear algebra, visualization and interpretation.* arXiv:2110.12439 (used for: clr DV). R18 (Geiger, Lu, Icard & Potts 2021) is replaced by Geiger, A., Ibeling, D., Zur, A., Chaudhary, M., Chauhan, S., Huang, J., Arora, A., Wu, Z., Goodman, N., Potts, C., & Icard, T. (2025). *Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability.* JMLR 26, 1–63 (used for: interchange interventions, IIA, P7). Both PDFs are in `Resources/` and were checked: Greenacre defines the centred log-ratio; Geiger et al. define interchange intervention accuracy (their Definition 48). The methodology PDF still cites the old entries; this log is the record of the change. |
| A16 | **Notebooks are not executed by Claude at all**, including smoke tests on synthetic data. Miaki runs every test. |

## 2026-11-15 · Found while reading for Phase 3 (Claude)

| # | Finding | Consequence |
|---|---|---|
| C28 | **The per-epoch mass probe is not representative.** `engine.attention_mass` reads the first two evaluation batches, which the length-sorted sampler fills with the 256 shortest VAL stays, at every query position. The reported sigmoid mass (0.24–0.26) and `kernel_raw` mass (0.039) describe those stays only. The "init ≈ 0.88" beside it in the handoff comes from the small smoke models in `00`, on other data. | The two leads (S3, kernel_raw) stand as leads only. `03` measures total mass on the analysis queries, by layer and position. `PREREGISTRATION.md` 6.1. The probe itself is left as is (changing it would not change any trained model). |
| C29 | **The PDF's four-outcome table is defined by intervention**, so it is decided in `04`. The brief (§5) and the handoff place it in `03`. | `03` reports a labelled correlational analogue. `PREREGISTRATION.md` 8. |
| C30 | **EXT:** A14 says one table at the very end; the PDF (P7) asks for replication on EXT. | One EXT pass at the end holds both. `PREREGISTRATION.md` 3. Needs Miaki's approval. |
| C31 | **PDF P7 leaves IIA undefined for a model that predicts a constant vector** (M0 always; M2 under do(Priority)): Kendall τ against a constant does not exist. | Match rule proposed in `PREREGISTRATION.md` 7.7. Needs Miaki's approval. |
| C32 | **E3 (layer-1 zero) holds for S1 only.** Under S0 the decay features of later hours are recomputed, so their inputs change. | S1 is the primary site for allocation quantities; S0 is reported beside it. `PREREGISTRATION.md` 7.2–7.3. |
| C33 | v3's DISCOVERY/SCREEN stay IDs are in `V3_FINAL_DIR/correlation/correlational_summary.json` (keys `discovery_stays`, `screen_stays`), one folder below where the handoff and the `config.py` comment put it. | `03` reads that path. |
| C34 | The saved output of `00_core_tests` is from commit `77d2dc7` with `MAX_SEQ_LEN = 336`, before C25. | Re-run `00` once on the commit pinned for `03`. |

## Open items, status on 2026-11-15

| # | Status |
|---|---|
| O1 | Rewording drafted in `PREREGISTRATION.md` 7.3; closes when the file is frozen. |
| O2, O3 | Closed in practice: Phase 2 ran from the repo clone and wrote its artifacts to `MyDrive/sepsis_v4`. |
| O4 | Checked against the PDFs in `Resources/`: R20 has three authors (Makelov, Lange, Nanda), as in §10; R22 is Sutter, Minder, Hofmann & Pimentel, arXiv:2507.08802 (the copy in `Resources/` is the arXiv version, so the NeurIPS venue is not confirmed from it); R19 is CLeaR 2024 (PMLR 236). R1 authors and R31 pages still to be settled in Phase 4. |
| O5 | R10 (SMOTE) and Wortsman 2022, Moor 2021, Fleuren 2020, Wong 2021, Kamran 2024, Harutyunyan 2019 and Caruana et al. are now in `Resources/`. Still missing: Touvron et al. 2021 (LayerScale), needed only if the sigmoid fallback is ever run. Every Phase 3 reference (R13–R25, with A15) is present. |
| O6 | New. Sources for Benjamini–Hochberg FDR and for random-effects prediction intervals are not in §10. To be proposed with full citations; `[unverified, PDF not provided]` until added. |
| H1 | Unchanged: Unit1/Unit2 coding in the MIMIC cohort build not yet confirmed. |

## 2026-11-15 · Pre-registration frozen (Miaki)

| # | Decision |
|---|---|
| A17 | **`Docs/PREREGISTRATION.md` is frozen** (stamp `Docs/PREREGISTRATION.frozen`, sha256 `e7f926d38bce508b20396f8f6d02b4b09ae3c92e87552560700f71f2b99b1b9e`). Section 13: items 1, 2, 4–10 and 12 approved as written; item 3 approved with Miaki's addition (if the paired softmax knockout also costs less than 0.01 AUROC, the outcome is "attention contributes little in either arm", pre-specified in 6.2 with its reading of the four-outcome table); item 11 approved with a limitation line in 5.4 (control seed variance is not estimated). This settles C30 and C31 and closes O1. |
| A18 | **Two references added outside §10's numbering**, both in `Resources/` and checked against the PDFs: Benjamini & Hochberg (1995), *JRSS B* 57(1), 289–300 (FDR); Riley, Higgins & Deeks (2011), *BMJ* 342, d549 (prediction intervals in random-effects meta-analysis). This closes O6. |
| A19 | From now on any change to a rule in the frozen file is a deviation: logged in `DEVIATIONS.md` with date, reason and Miaki's approval before the affected analysis runs. `03` and `04` may now be built. LOC, CONF and EXT are unlocked in `data.FoldGuard` by the stamp; the CONF-A/CONF-B halves and the single EXT pass still have to be enforced in code before `04` is handed over. |

## 2026-11-15 · Calls made while building Phase 3–5 code (Claude; accepted unless you object). Nothing was executed.

| # | Call | Why |
|---|---|---|
| C35 | **Arousal pairs use hours where HR and Resp are both measured.** | 7.6 takes the index "from the decayed input". Where both are measured the decayed input is the measured z-score for every model, so the index follows the frozen wording and the same pairs serve all models and seeds. |
| C36 | **S1 donor embedding = the embedding of the donor hour from its own inputs** (its own hour and static features included). | Literal reading of 7.2. The sham donor carries the same disturbance, so it cancels in the severe − sham contrast. |
| C37 | **Exact checks E1–E4 use a tolerance of 1e-5** (`EXACT_TOL` in `04`), and the maxima are printed. | "To float precision" for fp32 logits on a GPU. A larger value stops the notebook. |
| C38 | **Smoke runs never load CONF-A, CONF-B or EXT.** The dry run of the confirmation stage uses a second draw of LOC pairs. | A dry run must not spend the single confirmation pass. Enforced in `data.FoldGuard`. |
| C39 | **Fold enforcement in the core.** `FoldGuard` now knows CONF_A / CONF_B (sha1-parity rule, `data.conf_half`), refuses CONF whole, keeps CONF-A closed until `Docs/LOC_CHOICES.md` is committed, and opens EXT only for stage `final_ext`, only with the completion markers of 04 and 05, and only until the pass is closed in `MyDrive/sepsis_v4/EXT_ACCESS.json`. Test 7 in `00` covers the new rules. | Pre-registration section 3. |
| C40 | **`04` runs in three stages** (`STAGE = "LOC"`, `"CONF_A"`, `"EXT"`). The EXT pass is the last cell of `04` and needs `reports/05_confB_done.json`, which notebook 05 will write. | Keeps the replication code next to the code it replicates. |
| C41 | **Model-level quantities need 5 of 5 seeds** (IIA comparisons, interaction, κ, mass composition). IIA is pooled over the carried-forward heads that lie in layers ≥ 2 (all heads in layers ≥ 2 if the six are all in layer 1). | 5.6 states the 5-of-5 rule for model-level quantities; 7.7 did not say how IIA is aggregated over heads and seeds. |
| C42 | **S3 is patched per layer** (attention-sublayer output vs MLP output at the readout hour), not per head. The DAS "varies with the variable in natural activations" check is not implemented; the illusion check is subspace vs whole-component. | Exploratory site; per-head output patching needs a decomposition of W_O that adds code without changing a decision. |
| C43 | **Equivalence for causal effects** is computed from the bootstrap CI (normal approximation of its standard error), not from a separate TOST bootstrap. | Same decision in practice; one bootstrap per quantity. |
| C44 | **`03` trains two small controls** (`r33_softmax_shuf_v3`, `r33_sigmoid_shuf_v3`, v3 protocol, FIT only) for the exploratory set-B screen. `kernel_raw` has no shuffled-label control, so its band is marked incomplete. | The band needs a shuffled-label member for the reduced-feature models; nothing else is trained in `03`. |
| C45 | **Partial-SOFA proxy** thresholds are those of the SOFA table in Singer et al. 2016 (in `Resources/`): MAP < 70; platelets < 150/100/50/20; bilirubin ≥ 1.2/2.0/6.0/12; creatinine ≥ 1.2/2.0/3.5/5.0. | Exploratory set-B screen. |
| C46 | **Track D tools only** (`sepsis_core/explain.py`): six explanation methods, deletion/insertion, comprehensiveness/sufficiency, seed stability, scorecard. No candidate is chosen or built. Integrated Gradients uses the "nothing measured" hour as its reference input, 32 steps. | Candidate choice waits for the results of 03/04 (pre-registration section 10). |

Open: P1 and P2 in `DEVIATIONS.md` need a decision before the CONF-A stage of `04` is read.
