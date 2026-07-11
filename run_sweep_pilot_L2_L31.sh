#!/bin/bash
# Pilot-90 depth sweep, remaining arms: depths 2-9 and 20-31 (L10-19 done in run 021).
# Llama-3.1-8B single-layer magn steering, fv4 pilot directions, versions 1+6.
set -e
cd /Users/hadi/Source/BlackBoxNLP26/universal_steering
PY=/Users/hadi/Source/BlackBoxNLP26/.venv/bin/python
LOG=/private/tmp/claude-501/-Users-hadi-Source-BlackBoxNLP26-universal-steering/b9aa12f2-5555-45ec-808d-b1b9f50c2c31/scratchpad/sweep_pilot_L2_L31.log
CFG=steering_benchmark/run_config.py

for D in 2 3 4 5 6 7 8 9 20 21 22 23 24 25 26 27 28 29 30 31; do
  KEY=$((32 - D))
  sed -E -i '' "s/TARGET_KEYS = \{-[0-9]+\}.*/TARGET_KEYS = {-$KEY}  # single layer, depth $D/" $CFG
  echo "=== SWEEP pilot llama depth $D (key -$KEY) gen start $(date) ===" >> $LOG
  $PY -m steering_benchmark.eval_generations --model_set llama --model_size 8B >> $LOG 2>&1
  echo "=== SWEEP pilot llama depth $D judge start $(date) ===" >> $LOG
  $PY -m steering_benchmark.parse_results --model_set llama --model_size 8B >> $LOG 2>&1
  echo "=== SWEEP pilot llama depth $D DONE $(date) ===" >> $LOG
done
echo "=== SWEEP PILOT L2-L31 ALL DONE $(date) ===" >> $LOG
