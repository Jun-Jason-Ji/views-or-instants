#!/usr/bin/env bash
# GPU queue for protocol DPJAIT_NOISE_PREREG_v0.2, queued BEHIND the GPU chain of
# D:/PythonProject/basketball_video_RL (multi-seed sweep -> tools/gpu_queue_m1.sh).
#
# Same convention as that project's own queues: never share the 8 GB card.
#   1. wait until the basketball chain is gone: no gpu_queue_m1.sh process and no
#      train_qlora / run_m1_mixunit / analyze_m1_mixunit / a100 seed-sweep python
#      process, on two checks 120 s apart;
#   2. wait until the card is actually free: < 2500 MiB used, two checks 60 s apart
#      (also protects any other project's GPU job);
#   3. GPU: train the detector (dev records only), then infer the 11 records;
#   4. CPU: predict (sealed) -> score -> decide.
# Every step is resumable: after a crash or reboot just run this script again.
# To skip the wait (card known to be free):  V02_SKIP_WAIT=1 bash <this script>
set -uo pipefail
cd /e/research/VLLM/paper3
OUT=experiments/dpjait_noise_prereg_v0.2
LOG=$OUT/logs/gpu_queue.log
VPY=.venvs/gpu_det/Scripts/python.exe
RECS="R09_D1 R08_D1 R06_D1 R07_D1 R10_D1 R16_D1_A R11_D1 R05_D1 S11_D4 S13_D3 S12_D3"
mkdir -p $OUT/logs
stamp() { date '+%m-%d %H:%M:%S'; }
say() { echo "[v02-queue $(stamp)] $*" >> "$LOG"; }

basketball_alive() {
  # only python.exe / bash.exe: the powershell process running this query must not count itself
  powershell -NoProfile -Command "@(Get-CimInstance Win32_Process | Where-Object { (\$_.Name -eq 'python.exe' -or \$_.Name -eq 'bash.exe') -and \$_.CommandLine -match 'gpu_queue_m1\.sh|train_qlora|run_m1_mixunit|analyze_m1_mixunit|a100\.' }).Count" 2>/dev/null | tr -d '\r'
}
gpu_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' \r'
}

if [ "${V02_SKIP_WAIT:-0}" != "1" ]; then
  say "waiting for the basketball_video_RL GPU chain (sweep -> M1) to finish"
  until [ "$(basketball_alive)" = "0" ] && sleep 120 && [ "$(basketball_alive)" = "0" ]; do sleep 60; done
  say "basketball chain no longer running"
  say "waiting for the card to be free (< 2500 MiB)"
  until u=$(gpu_used_mib); [ -n "$u" ] && [ "$u" -lt 2500 ] && sleep 60 && u=$(gpu_used_mib) && [ -n "$u" ] && [ "$u" -lt 2500 ]; do sleep 60; done
  say "card free ($(gpu_used_mib) MiB used)"
fi

# --- 3a. dataset (CPU; normally already built) ----------------------------------------------
if [ ! -f experiments/dpjait_learned_det/dataset/dataset_info.json ]; then
  say "dataset missing -> building"
  rm -rf experiments/dpjait_learned_det/dataset
  python src/learned_det_dataset.py --out experiments/dpjait_learned_det/dataset >> "$LOG" 2>&1
  rc=$?; say "dataset exit $rc"; [ $rc -ne 0 ] && { echo "QUEUE_EXIT=dataset:$rc" >> "$LOG"; exit $rc; }
fi

# --- 3b. GPU: train ------------------------------------------------------------------------------------
say "train start"
$VPY src/learned_det_train.py >> $OUT/logs/train.log 2>&1
rc=$?; say "train exit $rc"
[ $rc -ne 0 ] && { echo "QUEUE_EXIT=train:$rc" >> "$LOG"; exit $rc; }

# --- 3c. GPU: inference per record (frozen protocol checked by predict later) ----------------------------
for r in $RECS; do
  say "infer $r start"
  $VPY src/learned_det_infer.py --record $r --out $OUT/observations >> $OUT/logs/infer_$r.log 2>&1
  rc=$?; say "infer $r exit $rc"
  [ $rc -ne 0 ] && { echo "QUEUE_EXIT=infer:$r:$rc" >> "$LOG"; exit $rc; }
done
say "GPU work done ($(gpu_used_mib) MiB used)"

# --- 4. CPU: predict (sealed) -> score -> decide -------------------------------------------------------------
for r in $RECS; do
  if [ ! -f $OUT/predictions/$r.json ]; then
    python src/dpjait_noise_runner_v02.py predict --out $OUT --record $r >> $OUT/logs/predict_$r.log 2>&1
    rc=$?; say "predict $r exit $rc"; [ $rc -ne 0 ] && { echo "QUEUE_EXIT=predict:$r:$rc" >> "$LOG"; exit $rc; }
  fi
  if [ ! -f $OUT/scored/$r.json ]; then
    python src/dpjait_noise_runner_v02.py score --out $OUT --record $r >> $OUT/logs/score_$r.log 2>&1
    rc=$?; say "score $r exit $rc"; [ $rc -ne 0 ] && { echo "QUEUE_EXIT=score:$r:$rc" >> "$LOG"; exit $rc; }
  fi
done
if [ ! -f $OUT/decision.json ]; then
  python src/dpjait_noise_runner_v02.py decide --out $OUT >> $OUT/logs/decide.log 2>&1
  rc=$?; say "decide exit $rc"; [ $rc -ne 0 ] && { echo "QUEUE_EXIT=decide:$rc" >> "$LOG"; exit $rc; }
fi
echo "QUEUE_EXIT=0" >> "$LOG"
say "all done"
