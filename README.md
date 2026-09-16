# Views or instants? Frame-budget allocation for multi-camera path-length measurement

Reproducibility archive for the article *How many views does a path-length
measurement need? Trading camera views for time instants under a fixed frame
budget*, submitted to *Measurement Science and Technology*.

Everything here runs on CPU. No model is trained and none is proposed.

## The question

A camera network that reports how far a target moved spends a finite processing
budget. Every frame that is transferred, decoded, detected and triangulated
costs the same, so a budget of `B` frame reads can buy `k` simultaneous views at
each of `m` time instants, subject to `B = k·m`. The two knobs act on opposite
biases of the same estimate: more instants reduce the corner cutting that makes
a sparsely sampled polyline too short, more views reduce the point noise that
makes a densely sampled one too long. Where should the budget go?

## What we found

**The budget should buy instants.** On a seven-camera rig, three views and seven
views give path-length error that cannot be distinguished at any sampling
density we tested, so the same measurement is obtained with 43 % of the
processed frames. On the pre-registered confirmation set, at 33 instants:

| views | frames per window | path-length MAE |
|---|---|---|
| 3 | 99 | 8.3 mm |
| 7 | 231 | 8.6 mm |

**Why the extra views are wasted.** A geometric model with independent pixel
noise predicts that seven views should be 41 % more accurate than three. The
realised ratio across five records is 0.46 to 1.92, mean 1.08. The reason is
that the reconstruction errors are largely common-mode: the median cosine
between the three-view and seven-view error vectors is 0.87. Extra views can
only average away the independent part, and a path length is invariant to a
common translation of its points, so the independent part is the only one that
survives into the length. On all five records the seven-view solution improves
when a single camera is removed, by 4 % to 45 %.

**Two bounds on further tuning.** Non-uniform placement of the instants gains
only 8 % to 25 % even with oracle access to the reference trajectory, and the
deployable closed-form rule is worse than uniform sampling. Separately, what
looked like noise accumulation at dense sampling is exposure to rare gross
outliers: the median and the ninetieth percentile fall monotonically, and the
mean rises because of one bad window in seventy-five.

**A single-camera chain has the opposite optimum,** so the allocation is a
property of the instrument and must be measured for each one.

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

So for any number in the article the chain from protocol to prediction to score
can be re-walked, and the order of the timestamps shows the reference was not
consulted while the predictions were being made.

## Reproducing

```bash
python src/verify_manuscript_numbers.py   # re-derive every number in the article
python src/check_mst_compliance.py        # check the manuscript against the journal's rules
```

Main entry points:

| script | produces |
|---|---|
| `run_allocation_sweep.py` | development sweep over all camera subsets |
| `run_allocation_confirm.py` | pre-registered confirmation, in `freeze` / `predict` / `score` steps |
| `score_allocation_v2.py` | corrected aggregation, counting a failed window as failed for every estimator |
| `run_view_saturation_mechanism.py` | geometric prediction, common-mode measurement, leave-one-camera-out |
| `run_adaptive_moments.py` | non-uniform placement including the oracle arm |
| `run_ctsd_baseline.py` | calibrated continuous-time estimator comparison |
| `uncertainty_budget.py` | uncertainty budget and its consistency check |
| `paired_view_test.py` | paired three-versus-seven-view test |
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

See [CITATION.cff](CITATION.cff). The archived release has its own DOI, minted
by Zenodo, which is given in the article's data availability statement.

## Licence

MIT for the code and the result files in this repository, see
[LICENSE](LICENSE). Third-party components are not included and remain under
their own terms.
