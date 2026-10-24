#!/usr/bin/env bash
# Build xiph/rnnoise (BSD-3) as a shared library for the RNNoise baseline.
set -euo pipefail
cd "$(dirname "$0")/.."
S=data/external/src/rnnoise
[ -d $S ] || git clone -q --depth 1 https://github.com/xiph/rnnoise.git $S
cd $S
h=$(cat model_version)
[ -f src/rnnoise_data.c ] || (curl -sSLf -o rnnoise_data-$h.tar.gz https://media.xiph.org/rnnoise/models/rnnoise_data-$h.tar.gz && tar xzf rnnoise_data-$h.tar.gz)
EXT=$([ "$(uname)" = Darwin ] && echo dylib || echo so)
cd src && cc -O2 -shared -fPIC -I../include -I. -o ../librnnoise.$EXT denoise.c rnn.c pitch.c kiss_fft.c celt_lpc.c nnet.c nnet_default.c parse_lpcnet_weights.c rnnoise_data.c rnnoise_tables.c -lm
echo "built $S/librnnoise.$EXT"
