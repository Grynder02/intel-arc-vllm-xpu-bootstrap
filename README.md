# Intel Arc vLLM XPU Bootstrap

Bootstrap scripts for running Intel's prebuilt vLLM XPU container on Intel Arc / Core Ultra style Linux machines.

Tested first on Ubuntu 24.04 with Intel Graphics exposed through `/dev/dri`.

Current proof:
- Docker image: `intel/vllm:0.21.0-ubuntu24.04-20260625`
- Torch: `2.11.0+xpu`
- vLLM inside the image: `0.21.1.dev17+g0a4756bb5` (the tag says 0.21.0, but the image ships a dev build — trust `vllm --version` inside the container, not the tag)
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
  its own Level Zero loader and compute runtime, so a patched host driver stack does
  not change what runs inside the container. Concretely: Level Zero has a
  fork-after-init bug on Meteor Lake-class hardware (the loader's
  `std::call_once`-cached init makes forked children hang or see 0 drivers). A fix
  exists — see the
  [`fix/meteor-lake-fork-safety` branch](https://github.com/Grynder02/compute-runtime/tree/fix/meteor-lake-fork-safety)
  and the upstream submission
  [intel/compute-runtime#954](https://github.com/intel/compute-runtime/pull/954) —
  and if you build/install that fixed driver on the host, you can run vLLM bare-metal
  (host venv, no container) instead of using this repo. This repo remains the
  **zero-host-modification** path for users who haven't touched their driver stack.
  Either way these scripts set `VLLM_WORKER_MULTIPROC_METHOD=spawn`: the container's
  runtime is unpatched, and even on a fixed host, PyTorch itself refuses XPU re-init
  in forked children — spawn is the supported worker mode everywhere. Do not remove it.
- The smoke test (`facebook/opt-125m`, fp32, 512 ctx) proves the plumbing, not
  performance. Real models on an iGPU will be memory-bound; fp32 doubles the
  footprint vs bf16.
