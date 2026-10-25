#!/usr/bin/env bash
# M8 on-device energy session (needs a phone on USB with the app installed; see HUMAN_TODO.md).
#   bash android/scripts/energy_session.sh <minutes> <tag>
# Resets batterystats, records a Perfetto trace (power rails + CPU freq + thermal) for the session,
# then dumps batterystats. Unplug USB power during the run if your phone supports adb over Wi-Fi,
# otherwise charge-state confounds %battery (power rails are still valid).
set -euo pipefail
MIN=${1:-30}; TAG=${2:-b16}
OUT=results/m8/device/$TAG-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
adb shell dumpsys batterystats --reset
adb shell dumpsys battery unplug || true
cat > "$OUT/perfetto.cfg" <<EOF
buffers { size_kb: 65536 fill_policy: RING_BUFFER }
data_sources { config { name: "android.power" android_power_config { battery_poll_ms: 1000 collect_power_rails: true } } }
data_sources { config { name: "linux.sys_stats" sys_stats_config { stat_period_ms: 1000 stat_counters: STAT_CPU_TIMES } } }
data_sources { config { name: "linux.ftrace" ftrace_config { ftrace_events: "power/cpu_frequency" ftrace_events: "thermal/thermal_temperature" } } }
duration_ms: $((MIN * 60 * 1000))
EOF
adb push "$OUT/perfetto.cfg" /data/local/tmp/clarity_perfetto.cfg >/dev/null
adb shell am start -n org.clarity.app/.MainActivity >/dev/null
echo "Start processing in the app now (wired headset, target model $TAG). Recording $MIN min..."
adb shell perfetto --txt -c /data/local/tmp/clarity_perfetto.cfg -o /data/misc/perfetto-traces/clarity.pftrace
adb pull /data/misc/perfetto-traces/clarity.pftrace "$OUT/" >/dev/null
adb shell dumpsys batterystats --charged org.clarity.app > "$OUT/batterystats.txt"
adb shell dumpsys battery reset || true
adb shell getprop ro.product.model > "$OUT/device.txt"
adb shell getprop ro.soc.model >> "$OUT/device.txt" || true
uv run python -m bench.parse_energy "$OUT" --minutes "$MIN" --tag "$TAG"
