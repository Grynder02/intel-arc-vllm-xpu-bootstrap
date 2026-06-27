#!/usr/bin/env bash
set -euo pipefail

NAME="${VLLM_CONTAINER_NAME:-vllm-xpu-opt125m}"

docker rm -f "$NAME" 2>/dev/null || true
echo "Stopped: $NAME"
