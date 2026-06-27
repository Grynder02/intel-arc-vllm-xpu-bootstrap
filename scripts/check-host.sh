#!/usr/bin/env bash
set -euo pipefail

echo "== Intel vLLM XPU host check =="

echo
echo "== GPU device =="
ls -l /dev/dri /dev/dri/renderD* 2>/dev/null || echo "missing /dev/dri render device"

echo
echo "== user groups =="
id | tr ' ' '\n' | grep -E 'render|video' || echo "missing render/video group"

echo
echo "== Docker =="
docker --version
docker images intel/vllm --format 'table {{.Repository}}\t{{.Tag}}\t{{.Size}}' || true

echo
echo "== power profile =="
command -v powerprofilesctl >/dev/null && powerprofilesctl get || echo "powerprofilesctl not found"
