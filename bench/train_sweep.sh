#!/usr/bin/env bash
# M3 sweep: train every config in configs/m3/gen in `order` on the GPU with LANES concurrent runs
# (2 lanes measured 1.47x the throughput of 1 on the M0 host; DECISIONS D15). Runs whose
# dependencies (KD teacher, curriculum init) are not trained yet wait. Re-entrant: finished runs
# (summary.json) are skipped; claims live in logs/claims/.
#   LANES=2 bash bench/train_sweep.sh        ONLY="b16_s0 b10_s0" bash bench/train_sweep.sh
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=results/m3/runs
LANES=${LANES:-2}
mkdir -p "$OUT" logs/claims
# claims of runs still alive from a previous invocation are kept (EXTRA_CLAIMS)
rm -rf logs/claims/*
for n in ${EXTRA_CLAIMS:-}; do mkdir -p "logs/claims/$n"; done
names=${ONLY:-$(uv run python -c "
import yaml, pathlib
cs = [yaml.safe_load(p.read_text()) for p in pathlib.Path('configs/m3/gen').glob('*.yaml')]
print(' '.join(c['name'] for c in sorted(cs, key=lambda c: c.get('order', 99))))")}

dep_ok() {  # every checkpoint dependency (teacher / init) has *finished* training (summary.json);
  # the checkpoint file alone is not enough: it is rewritten at every best-validation step
  local cfg=$1 d n
  for d in $(grep -E '^init_from:|^\s+teacher:' "$cfg" | awk '{print $2}'); do
    [ "$d" = "null" ] && continue
    n=$(basename "$d" .pt)
    [ -f "$OUT/$n/summary.json" ] && [ -f "$d" ] || return 1
  done
}

lane() {
  local id=$1 name progressed
  while true; do
    progressed=0; pending=0
    for name in $names; do
      [ -f "$OUT/$name/summary.json" ] && continue
      [ -e "logs/claims/$name.failed" ] && continue
      pending=1
      dep_ok "configs/m3/gen/$name.yaml" || continue
      mkdir "logs/claims/$name" 2>/dev/null || continue   # atomic claim
      echo "[lane $id] train $name $(date +%H:%M)"
      if uv run python -W ignore -m bench.train --config "configs/m3/gen/$name.yaml" --out "$OUT" \
           > "logs/train_$name.log" 2>&1; then
        echo "[lane $id] done $name $(date +%H:%M)"
      else
        echo "[lane $id] FAILED $name (logs/train_$name.log)"; touch "logs/claims/$name.failed"
      fi
      rmdir "logs/claims/$name"
      progressed=1
      break
    done
    [ $pending -eq 0 ] && break
    [ $progressed -eq 0 ] && sleep 60   # waiting on a dependency being trained by another lane
    if [ $progressed -eq 0 ] && [ -z "$(find logs/claims -mindepth 1 -maxdepth 1 -type d)" ]; then
      echo "[lane $id] nothing runnable and nothing in progress (missing dependency?)"; break
    fi
  done
}

for i in $(seq 1 "$LANES"); do lane "$i" & sleep 20; done
wait
echo "sweep finished $(date)"
