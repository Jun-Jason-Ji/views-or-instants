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

No model weights are included in this release.

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
