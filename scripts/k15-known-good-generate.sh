#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$DIR/vllm-xpu-shell" -lc '
export VLLM_XPU_USE_SAMPLER_KERNEL=0
export VLLM_USE_FLASHINFER_SAMPLER=0
python3 /workspace/examples/k15-tiny-generate.py
'
