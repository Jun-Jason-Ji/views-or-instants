**Files.** The archive `view_instant_allocation_v1.1.0.zip` (267.4 MB, SHA-256 fb6b48fba5231e1873bdfd353a8551731702a9e6bbdbafd6c302bb4b4f4dd8ea) is deposited in seven parts (`.part01`–`.part07`) because a single upload of this size did not complete. `REASSEMBLE_v1.1.0.txt` gives the one-line command to join them and the digest of every part.

# v1.1.0 — two confirmed rigs, the noise dependence of the Goldilocks density, and a reference-free predictor

**Added on 2026-09-26, before first publication of this version**

- **Second noise-ladder protocol (`DPJAIT_NOISE_PREREG_v0.2`, frozen 2026-09-25T12:32:42Z).**
  A second, independent observation chain: a YOLO11n detector fine-tuned only on
  DPJAIT development records (weights in `experiments/dpjait_learned_det/`,
  SHA-256 in `train_done.json`), run on the same windows and noise ladder. All
  four predictions pass: the power law (slopes −0.44, −0.63, −0.63, −0.69) and the
  bias crossing replicate, and a correlated-noise correction σ√(1−C), fixed in
  advance, brings 23 of 24 cells within ×1.5 of m* (21 uncorrected). On the real
  rig the correction overshoots. `reports/DPJAIT_NOISE_PREREG_v0.2_RESULT_2026-09-26.md`.
- **Post-hoc analyses (verdicts unchanged).** Window-bootstrap confidence intervals
  (`reports/NOISE_BOOTSTRAP_CI_2026-09-26.md`); a decomposition showing that the
  closed form is right on the real rig partly by cancellation of a 4.5–5.2-fold
  under-estimate of the chord deficit against a 1.5–4.7-fold under-estimate of
  the noise inflation (`reports/MSTAR_UNDERPREDICTION_DIAGNOSIS_2026-09-26.md`).
  That analysis also corrects a statement of earlier manuscripts: a time-varying
  offset between detection and reference DOES add path length.
- **Reference-free predictor (`REFREE_MSTAR_PREREG_v0.1`, frozen 2026-09-26T15:56:53Z).**
  Predicts m* from the observations alone (white noise from second differences,
  exact noncentral-χ segment inflation, deficit coefficient fitted to the data).
  On two records outside its development: 10 of 12 cells within ×1.5 (threshold
  9, pass); non-inferiority to the reference-based formula (10 vs 11) FAILED; a
  split-camera variant passed at threshold (9 of 12) but is unstable. Development
  in `reports/REFREE_MSTAR_DEV_2026-09-26.md`, result in
  `reports/REFREE_MSTAR_PREREG_RESULT_2026-09-26.md`, identity-binding
  sensitivity in `reports/REFREE_IDENTITY_SENSITIVITY_2026-09-26.md`.
- **Pre-registration time stamps.** These protocols were frozen locally (UTC and
  SHA-256 in each `freeze.json`) and first published after their results were
  known, in the public repository (tags `dpjait-prereg-v0.1`,
  `dpjait-noise-prereg-v0.1`, `dpjait-noise-prereg-v0.2`, `refree-mstar-prereg-v0.1`).
  The tags time-stamp publication, not the freeze.
- **Not included:** the detector training set (2.9 GB of DPJAIT frames; rebuilt by
  `src/learned_det_dataset.py`), the last-epoch weights, and the split-camera
  cache of the predictor's development (rebuilt by `src/refree_mstar_m4.py`).
  The manuscript currently under double-anonymized review is not included.

---

## Earlier content of v1.1.0 — the DPJAIT rig, and the mechanism behind the Goldilocks density

This version adds a second **confirmed** rig to the archive and the analysis that
identifies the Goldilocks density with the zero crossing of the signed sampling
bias.

**New in this version**

- **DPJAIT pre-registration (protocol v0.1, frozen 2026-09-20T11:15:23Z).**
  `protocol/DPJAIT_PREREG_v0.1.md`, the frozen runner and its source snapshot,
  the sealed predictions (`experiments/dpjait_prereg_v0.1/predictions/`, with
  `seal.json` giving the SHA-256 and UTC of each file), the scored results, the
  decision, and both errata. Fourteen confirmation records: nine single-target
  and two multi-target real records of the four-camera rig, and three records of
  the eight-camera simulated twin.
- **Outcome.** H1 (real, $m^\*$ median 6.0), H2a (four views win 56 of 56
  equal-budget cells), H3 (reference-free identity cost 3.83 → 1.66 → 1.16 from
  two to four views), H4 (independent-propagation ratio 1.32) and H5 (views do
  not saturate to eight on the simulated twin, median ratio 4.25) pass. **H1
  (simulated) and H2b fail** and are reported as written, not rewritten; see
  `reports/DPJAIT_PREREG_RESULT_2026-09-20.md`.
- **Mechanism.** `src/make_mechanism_figure.py` shows that the $m$ at which the
  signed bias crosses zero predicts the measured $m^\*$ within one grid step in
  all thirteen rig–statistic combinations in which the crossing falls inside the
  sampling grid, on MCalib, DPJAIT, the simulated twin and the Lyon development
  slice. `src/dpjait_sim_obs_noise.py` shows the crossing moves with the
  per-instant observation noise at equal target depth and speed.
- **Controlled noise test (protocol `DPJAIT_NOISE_PREREG_v0.1`, frozen 2026-09-25T11:11:51Z).**
  Our own observation chain built from the DPJAIT raw video; independent noise
  of known size (0–64 px, four seeded replicates) injected into every 2-D
  observation of the same 16 s windows. The log–log slope of $m^\*$ against the
  per-instant error is −0.55 (real) and −0.64 / −0.61 / −0.67 (three simulated
  records never analysed before), inside the pre-registered [−0.70, −0.30]; the
  closed form $m^\*=T\sqrt{\kappa v^2/(7\sigma_a)}$ is within ×1.5 in 22/24 cells
  and the bias crossing in 21/24. One prediction (H5, comparing two observation
  chains on 4 s windows) failed and is reported as written. See
  `reports/DPJAIT_NOISE_PREREG_RESULT_2026-09-25.md`.
- **Uncertainty budget for the new rig** (`src/dpjait_uncertainty_budget.py`):
  U = 198.5 mm = 6.00 % of a 3.310 m measurand at k = 2, against 0.22 % on
  MCalib; the reference contributes 0.2 mm and is not the limiting factor.
- **Current manuscript** (`paper/tim/`), restructured around the two confirmed
  rigs with the mechanism as the headline claim, and its figures.

**Unchanged from v1.0.x:** the MCalib analysis, the Lyon pre-registrations
(v0.2 and v0.3) and their development sweep, and the earlier reports.

**Still not included:** the three third-party datasets (MCalib, LBMC Lyon,
DPJAIT). None is ours to redistribute; the archive carries input hash lists so a
reader who obtains the data can verify it is byte-identical to ours.
