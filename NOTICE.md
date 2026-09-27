# Release scope and redistribution notice

## What this release contains

Original analysis code, frozen experimental protocols, input digest lists,
sealed predictions, scored results, the manuscript source, and the internal
reports behind it. All original code is under the MIT licence in `LICENSE`.

## What it does not contain, and why

**The two datasets analysed in the manuscript are not redistributed here.**
Both are third-party public releases and neither is ours to pass on. Obtain
them from their publishers under their own terms.

- Primary rig: a seven-camera release with an independent optical tracking
  reference, cited in the manuscript reference list.
- Secondary chain: a public colour-and-depth benchmark, likewise cited there.

What we distribute instead, for every confirmation run, is the list of input
files with their SHA-256 digests. A reader who obtains the data independently
can verify byte for byte that they hold the same inputs we used. **Digests
grant no access to the data and no right to redistribute it.** A dataset's code
or website licence does not automatically license the dataset content.

## Third-party software

`src/xfeat_cpu_adapter.py` is our own code. It does not contain third-party
source: it loads a pinned upstream release from a `third_party/` directory by
SHA-256, and that directory is deliberately excluded from this repository by
`.gitignore`. To run the parts of the pipeline that use it, obtain the upstream
release yourself; it is distributed by its authors under the Apache License
2.0, and its terms govern your use of it. The same applies to the dense matcher
used in the single-camera chain, also Apache 2.0.

No third-party model weights are included. One set of weights of our own is:
`experiments/dpjait_learned_det/train/run/weights/best.pt`, a YOLO11n detector
fine-tuned only on development records of the DPJAIT dataset (below). It is
included because the frozen protocol `REFREE_MSTAR_PREREG_v0.1` records its
SHA-256, so a reader can verify that the confirmation run used exactly these
weights. The base checkpoint and training code are Ultralytics (AGPL-3.0); the
weights are distributed on the same terms as the upstream checkpoint.

## DPJAIT (drone-tracking dataset)

The drone rig of the manuscript is the DPJAIT dataset (Rosner et al., Sci. Data
12:257, 2025), published on Zenodo under CC BY 4.0. We do not redistribute its
videos, calibration or Vicon reference. This release contains only our own
derived results on it: protocols, freeze records, sealed-prediction digests,
decisions and reports. Per-frame detections, dense reconstructions and scored
files are in the Zenodo archive (v1.1.0 and later), with attribution to the
dataset authors as required by CC BY 4.0.

## Pre-registration timestamps: what they do and do not prove

The protocols `DPJAIT_PREREG_v0.1`, `DPJAIT_NOISE_PREREG_v0.1`,
`DPJAIT_NOISE_PREREG_v0.2` and `REFREE_MSTAR_PREREG_v0.1` were frozen locally,
with the UTC time and SHA-256 digests recorded in each `freeze.json`, before
their confirmation records were processed. They were first published together
in this repository on 2026-09-26, after their results were known. The git tags
therefore time-stamp the publication, not the freeze: the freeze times rest on
our own records. Only `lyon-prereg-v0.3` was published before any of its
confirmation data existed.

## Bulk result files

Result files larger than 5 MB are stored gzip-compressed with a `.gz`
extension. They are otherwise unmodified and can be read directly, for example
with `gzip.open` in Python. The uncompressed digests are recorded in
`SHA256SUMS.txt` alongside the compressed ones.

## Authorship and copyright

The manuscript's author list is recorded separately from the software copyright
holders. Adding a manuscript author is not an assertion that they hold
copyright in all of the earlier software.

## A correction that is preserved on purpose

`reports/ALLOCATION_CLOSEOUT_AND_PUBLISHABILITY_2026-09-15.md` retracts an
earlier reading of our own data: a rise in error at dense sampling that we had
reported as the classical noise-accumulation mechanism, and had fitted a model
to, turned out to be exposure to rare gross outliers. The earlier reports
containing the withdrawn model are kept in this release so that a reader
following the record can see where it was retracted and on what evidence.
