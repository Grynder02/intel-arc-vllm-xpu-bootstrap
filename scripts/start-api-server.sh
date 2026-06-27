#!/usr/bin/env bash
set -euo pipefail

NAME="${VLLM_CONTAINER_NAME:-vllm-xpu-opt125m}"
IMAGE="${VLLM_XPU_IMAGE:-intel/vllm:0.21.0-ubuntu24.04-20260625}"
MODEL="${VLLM_MODEL:-facebook/opt-125m}"
PORT="${VLLM_PORT:-8001}"

docker rm -f "$NAME" 2>/dev/null || true

docker run -d \
  --name "$NAME" \
  --network=host \
  --device /dev/dri:/dev/dri \
  -v /dev/dri/by-path:/dev/dri/by-path \
  -v "$PWD":/workspace \
  -w /workspace \
  --ipc=host \
  -e VLLM_XPU_USE_SAMPLER_KERNEL=0 \
  -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  -e LD_LIBRARY_PATH=/opt/intel/oneapi/ccl/2021.15/lib:/opt/intel/oneapi/mpi/2021.15/lib:/opt/intel/oneapi/compiler/2025.3/lib:/opt/intel/oneapi/2025.3/lib:/usr/local/lib \
  --entrypoint bash \
  "$IMAGE" \
  -lc "python3 -m vllm.entrypoints.openai.api_server --host 127.0.0.1 --port $PORT --model $MODEL --dtype float32 --max-model-len 512 --gpu-memory-utilization 0.5 --enforce-eager"

echo "Started: $NAME"
echo "API: http://127.0.0.1:$PORT/v1"
