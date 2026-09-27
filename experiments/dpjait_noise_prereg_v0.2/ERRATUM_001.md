# ERRATUM 001 — dataset metadata written by hand (2026-09-25)

`src/learned_det_dataset.py` (frozen in v0.2, SHA-256 31b7bb9f…) exported every
training and validation image and label, then raised `TypeError: dict() got
multiple values for keyword argument 'train'` while writing `dataset_info.json`,
because its statistics dictionary uses the keys `train`/`val` that clash with the
record-list arguments of the same name.

The frozen source was NOT changed (changing it would break the freeze). The
exported data are complete and were checked by counting: train 6756 images /
21053 boxes from R01_D2, R02_D1, R12_D4, R13_D3, S07_D8; validation 1006 images /
3200 boxes from R03_D1, S05_D8; no confirmation record appears. `dataset_info.json`
was then written by hand from those counts. It is metadata only and is read by
nothing except the queue script's "dataset present?" check.

Consequence: if the dataset were ever deleted, the queue's rebuild step would hit
the same error. It is not deleted; the rebuild path is not used.

# ERRATUM 002 — first training attempt failed on a missing package (2026-09-25)

The queue started training at 15:22 local after the basketball_video_RL chain had
finished. Epoch 1 trained normally (validation mAP50 0.947) and then the job
crashed while saving the checkpoint: ultralytics reads its results CSV with
`polars`, which had not been installed because ultralytics was installed with
`--no-deps` (to keep the shared system Python untouched). Queue log:
`train exit 1`, `QUEUE_EXIT=train:1`.

Fix: `polars 1.44.2` installed into the isolated venv `.venvs/gpu_det` only (the
system Python still has no polars). No frozen source changed. The whole
train -> checkpoint -> validate -> predict path and the inference-cache ->
own_blob_observer.load() hand-off were then smoke-tested on CPU (1 % of the data,
1 epoch, 20 frames of dev record R02_D1). The partial run was archived as
`experiments/dpjait_learned_det/train/run_failed_2026-09-25_no_polars/` and
training restarts from the pretrained weights.

The queue's wait condition was also widened from `a100.exp_seeds` to any
`a100.` module of basketball_video_RL, after a new `a100.eval_controls` GPU job
appeared; the queue script is not a frozen source.
