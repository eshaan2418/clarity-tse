#!/usr/bin/env bash
# Download licence-open data + tools into data/external (git-ignored). Idempotent.
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/external; mkdir -p $D/src $D/dnsmos
for f in dev-clean test-clean train-clean-100; do
  [ -d $D/LibriSpeech/$f ] || (curl -sSL https://www.openslr.org/resources/12/$f.tar.gz | tar xz -C $D)
done
[ -d $D/ESC-50-master ] || (curl -sSL -o $D/esc50.zip https://github.com/karoldvl/ESC-50/archive/master.zip && unzip -q $D/esc50.zip -d $D && rm $D/esc50.zip)
[ -f $D/dnsmos/sig_bak_ovr.onnx ] || curl -sSLf -o $D/dnsmos/sig_bak_ovr.onnx https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx
bash scripts/build_rnnoise.sh
echo "data ready"
