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

## Why float32 (and what to know before trying bf16)

The known-good scripts run `--dtype float32` with `--enforce-eager` on purpose. On
Meteor Lake–class iGPUs the Intel graphics compiler (IGC) **rejects vLLM's prebuilt
bf16 SPIR-V kernels** at load time — the failure shows up as cryptic kernel/JIT
compile errors, not as a clean "dtype unsupported" message. float32 avoids those
prebuilt kernels entirely via the Triton fallback path.

If you want bf16/half on this class of hardware, force the Triton attention backend
so kernels are JIT-compiled instead of loaded as prebuilt SPIR-V:

```bash
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
```

Expect to validate this per driver/IGC version — it is the single most common reason
a "working" setup breaks after switching dtype.

## Known limitations

- **Host driver fixes do not reach into the container.** The intel/vllm image ships
  its own Level Zero loader and compute runtime. If you have patched the host driver
  stack (e.g. for the Level Zero loader's fork-after-init deadlock, where a
  `std::call_once`-cached init makes child processes hang on first GPU call after
  `fork()`), the container still runs the **unpatched** stack. These scripts set
  `VLLM_WORKER_MULTIPROC_METHOD=spawn` to sidestep that class of bug — do not remove
  it unless the image's compute runtime is known to be fork-safe.
- The smoke test (`facebook/opt-125m`, fp32, 512 ctx) proves the plumbing, not
  performance. Real models on an iGPU will be memory-bound; fp32 doubles the
  footprint vs bf16.
