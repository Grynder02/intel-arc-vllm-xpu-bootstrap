#!/usr/bin/env bash
set -euo pipefail

mkdir -p "$HOME/.local/bin"
cp "$(dirname "$0")/vllm-xpu-shell" "$HOME/.local/bin/vllm-xpu-shell"
chmod +x "$HOME/.local/bin/vllm-xpu-shell"

echo "Installed: $HOME/.local/bin/vllm-xpu-shell"
echo "Test with: vllm-xpu-shell -lc 'python3 -c \"import torch; print(torch.xpu.is_available())\"'"
