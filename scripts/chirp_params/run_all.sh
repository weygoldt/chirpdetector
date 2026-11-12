#!/usr/bin/env bash
# Train the decoder variants in three guarded chains (shared machine: small
# memory and VRAM caps), then evaluate and run the real-data check.
#   run_all.sh CACHE RUNS OUT
set -euo pipefail
cache=$1 runs=$2 out=$3
here=$(cd "$(dirname "$0")" && pwd)
guard=/home/weygoldt/wrk/analyses/chirp_competition_upgrade/scripts/guarded
py=.venv/bin/python
ep=${EPOCHS:-20}
mkdir -p "$runs" "$out"
chain1() {
  $guard --name params-train-B --mem 6G --vram 0.1 -- $py "$here/train.py" "$cache" "$runs/B_sum" --input B --epochs "$ep" --test-baseline-sd 2.4 5
  $guard --name params-train-C --mem 6G --vram 0.1 -- $py "$here/train.py" "$cache" "$runs/C_top3" --input C --epochs "$ep" --test-baseline-sd 2.4
}
chain2() {
  $guard --name params-train-Bj --mem 6G --vram 0.1 -- $py "$here/train.py" "$cache" "$runs/B_sum_bljit" --input B --epochs "$ep" --baseline-sd-hz 2.4 --test-baseline-sd 2.4 5
  $guard --name params-train-A --mem 6G --vram 0.1 -- $py "$here/train.py" "$cache" "$runs/A_detector" --input A --epochs "$ep"
}
chain3() {
  $guard --name params-train-K --mem 6G --vram 0.1 -- $py "$here/train.py" "$cache" "$runs/K_clean" --input K --epochs "$ep" --test-baseline-sd 2.4
  $guard --name params-train-Blong --mem 6G --vram 0.1 -- $py "$here/train.py" "$cache" "$runs/B_long_windows" --input B --channels 0 1 --epochs "$ep" --test-baseline-sd 2.4
  $guard --name params-train-Bshort --mem 6G --vram 0.1 -- $py "$here/train.py" "$cache" "$runs/B_short_windows" --input B --channels 2 3 --epochs "$ep" --test-baseline-sd 2.4
}
chain1 > "$out/chain1.log" 2>&1 &
p1=$!
chain2 > "$out/chain2.log" 2>&1 &
p2=$!
chain3 > "$out/chain3.log" 2>&1 &
p3=$!
wait $p1 $p2 $p3
$guard --name params-eval --mem 6G --vram 0.0 -- $py "$here/evaluate.py" "$cache" "$runs" "$out" > "$out/evaluate.log" 2>&1
$guard --name params-real --mem 5G --vram 0.1 -- $py "$here/real_check.py" "$runs" "$out/real" > "$out/real.log" 2>&1
$py "$here/analyze_real.py" "$out/real" > "$out/real_analysis.log" 2>&1
echo "all done"
