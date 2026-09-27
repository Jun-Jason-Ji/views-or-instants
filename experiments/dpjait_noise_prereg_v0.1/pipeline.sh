#!/usr/bin/env bash
# predict -> score each record as soon as its observe step has exited 0; decide at the end.
cd /e/research/VLLM/paper3
OUT=experiments/dpjait_noise_prereg_v0.1
RECS="R05_D1 R06_D1 R07_D1 R08_D1 R09_D1 R10_D1 R11_D1 R16_D1_A S11_D4 S12_D3 S13_D3"
run_one() {
  r=$1; log=$OUT/logs/observe_$r.log
  until grep -q "EXIT=" "$log" 2>/dev/null; do sleep 15; done
  if ! grep -q "EXIT=0" "$log"; then echo "$r OBSERVE_FAILED" >> $OUT/logs/pipeline_status.txt; return; fi
  python src/dpjait_noise_runner.py predict --out $OUT --record $r > $OUT/logs/predict_$r.log 2>&1; pe=$?
  echo "EXIT=$pe" >> $OUT/logs/predict_$r.log
  if [ $pe -ne 0 ]; then echo "$r PREDICT_FAILED" >> $OUT/logs/pipeline_status.txt; return; fi
  python src/dpjait_noise_runner.py score --out $OUT --record $r > $OUT/logs/score_$r.log 2>&1; se=$?
  echo "EXIT=$se" >> $OUT/logs/score_$r.log
  if [ $se -ne 0 ]; then echo "$r SCORE_FAILED" >> $OUT/logs/pipeline_status.txt; return; fi
  echo "$r OK" >> $OUT/logs/pipeline_status.txt
}
for r in $RECS; do run_one $r & done
wait
python src/dpjait_noise_runner.py decide --out $OUT > $OUT/logs/decide.log 2>&1
echo "EXIT=$?" >> $OUT/logs/decide.log
echo "PIPELINE_DONE"
cat $OUT/logs/pipeline_status.txt
cat $OUT/logs/decide.log
