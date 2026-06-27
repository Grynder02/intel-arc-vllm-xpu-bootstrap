# Intel Arc vLLM XPU Bootstrap

Bootstrap scripts for running Intel's prebuilt vLLM XPU container on Intel Arc / Core Ultra style Linux machines.

Tested first on Ubuntu 24.04 with Intel Graphics exposed through `/dev/dri`.

Current proof:
- Docker image: `intel/vllm:0.21.0-ubuntu24.04-20260625`
- Torch: `2.11.0+xpu`
- `torch.xpu.is_available()` = `True`
- vLLM imports successfully
- Required fix: pass oneAPI library paths through `LD_LIBRARY_PATH`

This repo avoids host-building vLLM first and keeps the Intel runtime stack inside Docker.
