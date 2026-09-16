# Views or instants? Frame-budget allocation for multi-camera path-length measurement

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22784740.svg)](https://doi.org/10.5281/zenodo.22784740)

Reproducibility archive for the article *Allocating camera views and time instants for trajectory-length measurement under a frame-processing budget*, prepared for *Measurement Science and Technology*.

Everything here runs on CPU. No model is trained and none is proposed.

## Current manuscript revision (16 September 2026)

Version 1.0.4 archives the revised manuscript, compiled PDF and derived
exploratory analyses at https://doi.org/10.5281/zenodo.22795247.
The matching source release is https://github.com/Jun-Jason-Ji/views-or-instants/tree/v1.0.4.
Original protocols and sealed predictions remain unchanged. Earlier Zenodo
versions preserve historical snapshots. The manuscript is prepared for submission.

## Sampling-grid and robustness controls (v1.0.4)

The new analysis applies eleven fixed-endpoint grids to three and seven views
at 84 and 168 frames, under DLT, spatial checks and spatial-plus-temporal
checks. It retains all 17820 window/configuration outcomes on five development
and four evaluation records. These are correlated perturbations of previously
examined data, not a new holdout or 17820 independent observations.

At 168 frames the unfiltered three-view MAE spans 8.01--144.66 mm across grids.
Spatial checks narrow this to 7.92--8.36 mm, with three views below seven in
all eleven grids and 60/60 evaluation windows complete throughout. On the
original grid, two internal rejections reduce MAE from 85.66 to 7.92 mm.
Additional temporal gating does not consistently help and loses windows.
Retrospective development-only selection illustrates completion and empirical
tail-error constraints; independent validation is still needed.

The historical geometric subset score is **best-pair**, not worst-pair:
it takes a maximum range/focal scale within each pair, then the minimum
over pairs. Ties use lexicographic camera-ID order. Code outputs and old
selected subsets are unchanged; prior text descriptions were inaccurate.

Run `python src/allocation_revision_v104.py`, then
`python src/allocation_revision_v104_report.py` and
`python src/verify_grid_revision.py`. Exact executed source hashes and snapshots
are under `experiments/allocation_revision_v104_2026-09-16/`.
Version 1.0.3 remains available at https://doi.org/10.5281/zenodo.22794671.

## Added budget and robustness analysis (v1.0.3)

The new grid evaluates two through seven views at budget caps 56, 84, 112
and 168, using one calibration-selected subset per view count and floor(B/k)
uniform instants. All 1680 window configurations are retained, including failures.
At 84 frames, the three-view advantage persists across observation noise of
0.4, 1 and 3 mm, reference decimation and whole-record deletion. At 168 frames,
however, the tested three-view schedule has MAE 85.66 mm against 10.55 mm for
seven views; its largest error is 4110.45 mm. More budget does not require
using all available frames: cheaper schedules remain feasible. This grid is
not a globally optimized lower error envelope and supports no universal view count.

Run `python src/allocation_revision_v103.py` with the original public inputs,
then `python src/verify_budget_revision.py`. New results are under
`experiments/allocation_revision_v103_2026-09-16/`. Original estimates are
replicated for 288 overlapping cases with zero difference. These are exploratory
analyses, not a new independent confirmation set.

## The question and current findings

We evaluate a frame-request budget B = k*m, a processing proxy rather than
measured runtime, energy or camera acquisition cost.

At exactly 84 frames per window, the calibration-selected triple at 28
instants has MAE 9.60 mm, versus 50.77 mm for seven views at 12 instants.
This is an exploratory comparison on the recorded secondary confirmation grid.
At a fixed 33 instants, the selected triple uses 99 rather than 231 frames,
but its MAE is 9.25 rather than 8.63 mm. Statistical equivalence is not established.
Subset medians (8.3 versus 8.6 mm) are different descriptive quantities.

Additional exploratory controls examine overlapping and disjoint camera
sets, independent pixel noise, temporal error increments and a fixed robust
triangulation rule. They do not identify a unique causal mechanism or a
shared-error variance fraction. The reference-assisted curvature rule is
one baseline, not a general optimality bound. The dense-regime mean is
strongly affected by rare extreme errors, and gating trades error against
completion. XFeat, the pre-specified primary single-camera method, did not
meet its full confirmation criterion; the secondary ORB method did.

Error dispersion and reference-decimation sensitivity are conditional
summaries, not a complete instrument uncertainty budget or verified coverage.
The manuscript reports these limitations explicitly.

## Layout

```
src/           analysis code
experiments/   frozen protocols, input digests, sealed predictions, scored results
paper/         manuscript source, figures, novelty statement
reports/       the three internal reports behind the manuscript
SHA256SUMS.txt digest of every file here
```

## How the evidence is structured

Each confirmation experiment runs in four steps, and each leaves a file behind:

1. `freeze` writes the protocol, the candidate list, the decision gates and the
   SHA-256 of every source file it will use.
2. `predict` computes every prediction and writes a SHA-256 seal over the
   prediction file, with a timestamp.
3. `score` reads the reference for the first time and records the timestamp at
   which it did so. That timestamp is always later than the seal.
4. The decision file records the pre-declared gates and whether each passed.

The primary-rig confirmation records had previously been exposed to per-frame
triangulation quality checks. The local timestamps document execution order,
not independently timestamped public preregistration. New analyses are marked
exploratory and do not modify frozen source snapshots or original decisions.
Some original decisions predate corrected failure aggregation; use the current
manuscript and derived summaries rather than treating all historical summaries
as current conclusions.

## Reproducing

```bash
python src/submission_sensitivity_analysis.py  # record-level and sequence-equal summaries
python src/submission_sensitivity_analysis.py --geometry  # also requires original primary data
python src/verify_manuscript_numbers.py   # check tabulated original and revised values
python src/check_mst_compliance.py        # check the manuscript against the journal's rules
```

Main entry points:

| script | produces |
|---|---|
| `run_allocation_sweep.py` | development sweep over all camera subsets |
| `run_allocation_confirm.py` | pre-registered confirmation, in `freeze` / `predict` / `score` steps |
| `score_allocation_v2.py` | corrected aggregation, counting a failed window as failed for every estimator |
| `run_view_saturation_mechanism.py` | geometric prediction, error directions and leave-one-camera-out diagnostics |
| `run_adaptive_moments.py` | non-uniform placement including the oracle arm |
| `run_ctsd_baseline.py` | calibrated continuous-time estimator comparison |
| `uncertainty_budget.py` | historical error-scale calculation; current interpretation is qualified in the manuscript |
| `paired_view_test.py` | historical window/subset test; superseded by exploratory record comparisons |
| `run_tum_dense_cheap_confirm.py` | single-camera chain, pre-registered unseen windows |

Scripts that re-run an experiment write to a new output directory and refuse to
overwrite an existing one, so the sealed records cannot be modified in place.

Requires Python 3 with NumPy, SciPy, OpenCV and Matplotlib. Result files larger
than 5 MB are stored gzip-compressed and can be read directly with `gzip.open`.

## Data

The two datasets are third-party public releases and are **not** redistributed
here. `experiments/*/input_hashes.json` lists every input file with its SHA-256
so you can verify you hold the same inputs. See [NOTICE.md](NOTICE.md) for the
full scope and for the third-party software terms.

## Citing

See [CITATION.cff](CITATION.cff).

- Concept DOI, always the latest version: [10.5281/zenodo.22784740](https://doi.org/10.5281/zenodo.22784740)
- Current budget and robustness revision, v1.0.3: [10.5281/zenodo.22794671](https://doi.org/10.5281/zenodo.22794671)
- Previous manuscript and analysis revision, v1.0.2: [10.5281/zenodo.22793719](https://doi.org/10.5281/zenodo.22793719)
- Original archived results, v1.0.1: [10.5281/zenodo.22791285](https://doi.org/10.5281/zenodo.22791285)
- Previous version, v1.0.0: [10.5281/zenodo.22784741](https://doi.org/10.5281/zenodo.22784741)

## Licence

MIT for the code and the result files in this repository, see
[LICENSE](LICENSE). Third-party components are not included and remain under
their own terms.
