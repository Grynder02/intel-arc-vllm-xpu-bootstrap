#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$DIR/vllm-xpu-shell" -lc 'python3 - <<PY
import torch, vllm
print("torch", torch.__version__)
print("xpu", torch.xpu.is_available(), torch.xpu.device_count())
print("device", torch.xpu.get_device_name(0))
print("vllm", vllm.__version__)
PY'
