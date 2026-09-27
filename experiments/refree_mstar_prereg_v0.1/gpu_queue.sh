#!/usr/bin/env bash
# Queue for protocol REFREE_MSTAR_PREREG_v0.1 (same GPU etiquette as dpjait_noise_prereg_v0.2):
#   wait until no basketball_video_RL GPU job runs (two checks 120 s apart) and the card
#   uses < 2500 MiB (two checks 60 s apart); then GPU inference on the two confirmation
#   records; then CPU predict (sealed, incl. M3/M4) -> score -> decide.
# Resumable: rerun after a crash or reboot.  Skip the wait with RF_SKIP_WAIT=1.
set -uo pipefail
cd /e/research/VLLM/paper3
unset REFREE_DRY_RECORDS
OUT=experiments/refree_mstar_prereg_v0.1
LOG=$OUT/logs/gpu_queue.log
VPY=.venvs/gpu_det/Scripts/python.exe
RECS="R04_D2 R14_D3"
stamp() { date '+%m-%d %H:%M:%S'; }
say() { echo "[refree-queue $(stamp)] $*" >> "$LOG"; }
basketball_alive() {
  powershell -NoProfile -Command "@(Get-CimInstance Win32_Process | Where-Object { (\$_.Name -eq 'python.exe' -or \$_.Name -eq 'bash.exe') -and \$_.CommandLine -match 'gpu_queue_m1\.sh|train_qlora|run_m1_mixunit|analyze_m1_mixunit|a100\.' }).Count" 2>/dev/null | tr -d '\r'
}
gpu_used_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' \r'; }

if [ "${RF_SKIP_WAIT:-0}" != "1" ]; then
  say "waiting for basketball_video_RL GPU jobs to finish"
  until [ "$(basketball_alive)" = "0" ] && sleep 120 && [ "$(basketball_alive)" = "0" ]; do sleep 60; done
  say "basketball jobs gone; waiting for card < 2500 MiB"
  until u=$(gpu_used_mib); [ -n "$u" ] && [ "$u" -lt 2500 ] && sleep 60 && u=$(gpu_used_mib) && [ -n "$u" ] && [ "$u" -lt 2500 ]; do sleep 60; done
  say "card free ($(gpu_used_mib) MiB used)"
fi
for r in $RECS; do
  if [ ! -f $OUT/observations/$r.npz ]; then
    say "infer $r start"
    $VPY src/learned_det_infer.py --record $r --out $OUT/observations >> $OUT/logs/infer_$r.log 2>&1
    rc=$?; say "infer $r exit $rc"; [ $rc -ne 0 ] && { echo "QUEUE_EXIT=infer:$r:$rc" >> "$LOG"; exit $rc; }
  fi
done
say "GPU work done"
for r in $RECS; do
  if [ ! -f $OUT/predictions/$r.json ]; then
    python src/refree_prereg_runner.py predict --out $OUT --record $r >> $OUT/logs/predict_$r.log 2>&1
    rc=$?; say "predict $r exit $rc"; [ $rc -ne 0 ] && { echo "QUEUE_EXIT=predict:$r:$rc" >> "$LOG"; exit $rc; }
  fi
done
for r in $RECS; do
  if [ ! -f $OUT/scored/$r.json ]; then
    python src/refree_prereg_runner.py score --out $OUT --record $r >> $OUT/logs/score_$r.log 2>&1
    rc=$?; say "score $r exit $rc"; [ $rc -ne 0 ] && { echo "QUEUE_EXIT=score:$r:$rc" >> "$LOG"; exit $rc; }
  fi
done
if [ ! -f $OUT/decision.json ]; then
  python src/refree_prereg_runner.py decide --out $OUT >> $OUT/logs/decide.log 2>&1
  rc=$?; say "decide exit $rc"; [ $rc -ne 0 ] && { echo "QUEUE_EXIT=decide:$rc" >> "$LOG"; exit $rc; }
fi
echo "QUEUE_EXIT=0" >> "$LOG"; say "all done"
