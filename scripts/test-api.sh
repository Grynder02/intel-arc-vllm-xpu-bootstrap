#!/usr/bin/env bash
set -euo pipefail

PORT="${VLLM_PORT:-8001}"
MODEL="${VLLM_MODEL:-facebook/opt-125m}"

echo "== models =="
curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -m json.tool

echo
echo "== completion =="
curl -s "http://127.0.0.1:$PORT/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"The capital of Texas is\",\"max_tokens\":24,\"temperature\":0}" \
  | python3 -m json.tool
