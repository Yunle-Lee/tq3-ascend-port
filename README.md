# TurboQuant TQ3 × Ascend CANN Backend Port

Port [TurboQuant (TQ3)](https://github.com/YTan2000/turboquant) 3-bit quantized inference to [llama.cpp](https://github.com/ggml-org/llama.cpp)'s Ascend CANN backend, enabling TQ3 models (e.g. Qwen3.6-35B-A3B-MTP-TQ3\_4S) to run on **Ascend 910 NPUs** without NVIDIA GPUs.

## Background

### TurboQuant TQ3

TurboQuant is a 3-bit weight quantization scheme that achieves ~4 bpw (bits per weight) with competitive quality. It uses:

- **3-bit packed indices** (12 bytes → 32 indices) with a centroid LUT of 8 values
- **Per-8 scaling**: 4 E3M5 mini-floats per block (TQ3\_4S) or 2 FP16 half-block scales (TQ3\_1S)
- **Walsh-Hadamard Transform (WHT)**: butterfly network to spread quantization error
- **Golden-ratio sign projection**: deterministic sign hashing for zero-mean output
- **Block size**: 32 elements, 16 bytes per block

### The Problem

llama.cpp already has a CANN backend (`ggml/src/ggml-cann/`) supporting F32/F16/BF16/Q4\_0/Q8\_0 matmul via ACLNN operators, plus a CUDA backend with hand-optimized TQ3 kernels (`tq3-native.cu`/`.cuh`). However:

- The TQ3 kernels are written in CUDA and require NVIDIA GPUs
- Ascend 910 NPUs cannot run CUDA code
- No TQ3 support existed in the CANN backend

### Solution

Replace the CUDA TQ3 kernel path with a **CPU dequant → ACL Cast → ACL BatchMatMul** pipeline integrated into the existing CANN backend. While slower than a native Ascend C kernel, this gets TQ3 models running on Ascend hardware immediately, with a clear optimization path to a custom Ascend C dequant kernel.

## What We Did

| Step | Status |
|------|--------|
| Read and understood the CUDA TQ3 kernel (`tq3-native.cu`/`.cuh`, `tq3-prefill.cuh`, `turbo-wht.cu`) | ✅ |
| Understood TQ3\_4S block format (E3M5, 3-bit unpacking, WHT, sign projection) | ✅ |
| Wrote a CPU TQ3\_4S dequant implementation and verified against known test vectors | ✅ |
| Integrated into `ggml_cann_mul_mat` dispatch — TQ3\_4S and TQ3\_1S now accepted | ✅ |
| Compiled `ggml-cann` library with zero errors | ✅ |
| Verified multi-NPU model loading with llama.cpp CANN backend | ✅ |
| Wrote Ascend C kernel template via `msopgen` (compilation pending device kernel toolchain fix) | 🟡 |

### Source Files Created/Modified

#### New files in `ggml/src/ggml-cann/`

| File | Lines | Purpose |
|------|-------|---------|
| `tq3_cann.h` | 3 | Declaration of `ggml_cann_mul_mat_tq3` |
| `tq3_cann.cpp` | ~210 | CPU dequant (E3M5 decode, 3-bit unpack, WHT butterfly, sign) + ACL Cast + ACL BatchMatMul |

#### Modified files in `ggml/src/ggml-cann/`

| File | Change | Purpose |
|------|--------|---------|
| `aclnn_ops.cpp` | +3 lines in `ggml_cann_mul_mat` switch | Dispatch `GGML_TYPE_TQ3_4S` / `GGML_TYPE_TQ3_1S` to the new function |

No CMake changes needed — the build uses `file(GLOB ... "*.cpp")` and picks up new files automatically.

## Architecture

```
User: TQ3 GGUFF model
           │
           ▼
    llama.cpp inference loop
           │
           ▼
    ggml_cann_mul_mat(dst)
      │ type = TQ3_4S / TQ3_1S
      │
      ▼
    ggml_cann_mul_mat_tq3(ctx, dst)
      │
      ├── 1. CPU dequant TQ3 weights → FP32
      │      ├── decode 4x E3M5 per-8 scales
      │      ├── unpack 32 x 3-bit indices
      │      ├── centroid LUT lookup + per-8 scaling
      │      ├── Walsh-Hadamard Transform (32 elements)
      │      └── golden-ratio sign projection + 1/sqrt(32) normalize
      │
      ├── 2. ACL Cast FP32 → FP16 (on device)
      │
      ├── 3. ACL Cast input → FP16 (if needed)
      │
      ├── 4. ACL BatchMatMul [MxK] x [KxN] → [MxN] FP16
      │
      └── 5. ACL Cast output → dst type (if needed)
```

### Key Design Decisions

1. **CPU dequant (not Ascend C kernel)** — The CANN toolchain (ccec) for Ascend C kernel compilation is complex. CPU dequant gets the system working immediately; an Ascend C kernel can be swapped in later without changing the matmul pipeline.

2. **FP16 matmul (not F32)** — Ascend 910's Cube units are optimized for FP16. The dequant output is FP32 but immediately cast to FP16 before matmul.

3. **ACL BatchMatMul (not MatMulV2)** — BatchMatMul is the standard CANN API for general matrix multiply. It handles arbitrary batch dimensions correctly.

4. **Dual NPU support** — The existing CANN backend already splits model layers across multiple NPUs. No changes needed for multi-device support.

## Quick Start

### Prerequisites

- **Hardware**: Ascend 910 (or 910B) NPU
- **Software**:
  - CANN toolkit 8.5.0+ (with `set_env.sh`)
  - CMake 3.16+
  - GCC 9.4+
  - Python 3.9+

### Build

```bash
# 1. Source CANN environment
source /path/to/Ascend/cann-8.5.0/set_env.sh

# 2. Clone llama.cpp-tq3 fork (includes TQ3 types)
git clone https://github.com/ggml-org/llama.cpp.git
# OR use the TQ3 fork:
git clone https://github.com/YTan2000/llama.cpp-tq3.git

# 3. Apply our patch
cd llama.cpp-tq3
git am /path/to/tq3-ascend-port/patches/0001-add-tq3-cann-backend.patch

# 4. Build with CANN backend
mkdir build && cd build
cmake .. -DGGML_CANN=ON \
         -DCANN_INSTALL_DIR=${ASCEND_TOOLKIT_HOME} \
         -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

# 5. Run inference
./bin/llama-perplexity \
    -m /path/to/qwen3.6-35b-a3b-mtp-tq3_4s.gguf \
    -ngl 99 \
    --no-warmup
```

### Test CPU Dequant (Standalone)

```bash
cd tests/
g++ -std=c++17 -O2 -o test_tq3_dequant test_tq3_dequant.cpp ../ggml-cann-integration/tq3_cann.cpp -lm
./test_tq3_dequant
```

## Environment

This port was developed and tested on:

| Component | Version |
|-----------|---------|
| NPU | 2× Ascend 910 (64GB HBM each) |
| Architecture | aarch64 (ARMv8.2) |
| CANN | 8.5.0 |
| CCE compiler | clang 15.0.5 (ccec) |
| OS | Linux (aarch64) |
| llama.cpp fork | YTan2000/llama.cpp-tq3 (TQ3 types: 44/46/200) |

## File Reference

```
tq3-ascend-port/
├── README.md                                 # This file
├── patches/
│   └── 0001-add-tq3-cann-backend.patch       # Git patch for llama.cpp-tq3
├── ggml-cann-integration/
│   ├── tq3_cann.h                            # Declares ggml_cann_mul_mat_tq3
│   └── tq3_cann.cpp                          # CPU dequant + ACL Cast + BatchMatMul
├── tests/
│   ├── test_tq3_dequant.cpp                  # CPU dequant validation
│   └── test_tq3_cann_integration.cpp         # CANN backend integration test
├── reference/
│   ├── ggml_cann_tq3_integration.cpp         # Original integration stub/design
│   └── TQ3_ALGORITHM.md                      # TQ3 format specification
├── ascend-c-kernel/
│   ├── README.md                             # Status of Ascend C kernel effort
│   ├── op_desc.json                          # msopgen operator description
│   └── src/
│       ├── tq3_dequant_host.cpp              # Host-side operator code (msopgen)
│       └── tq3_dequant_kernel.cpp            # Device kernel template (empty body)
└── examples/
    └── compile_and_test.sh                   # Build + test script
```

## Key Files Referenced

### TQ3 CUDA Kernel (Original)
- `ggml/src/ggml-cuda/tq3-native.cu` — TQ3 dot product kernel
- `ggml/src/ggml-cuda/tq3-native.cuh` — TQ3 header with block structs and helpers
- `ggml/src/ggml-cuda/tq3-prefill.cuh` — TQ3 prefill (dequant + matmul) kernel
- `ggml/src/ggml-cuda/turbo-wht.cu` — Generic Walsh-Hadamard Transform kernel
- `ggml/src/ggml-cuda/template-instances/mmq-instance-tq3_4s.cu` — MMQ template instantiation

### CANN Backend (Target)
- `ggml/src/ggml-cann/aclnn_ops.cpp` — ACLNN operations including `ggml_cann_mul_mat`
- `ggml/src/ggml-cann/aclnn_ops.h` — Operation declarations and `GGML_CANN_CALL_ACLNN_OP` macro
- `ggml/src/ggml-cann/common.h` — `ggml_backend_cann_context`, `ggml_cann_pool_alloc`, helpers
- `ggml/src/ggml-cann/acl_tensor.h` — `ggml_cann_create_tensor`, `ggml_cann_type_mapping`
- `ggml/src/ggml-cann/ggml-cann.cpp` — Backend init, compute dispatch

### TQ3 Type Definitions
- `ggml/include/ggml.h` — `GGML_TYPE_TQ3_1S=44`, `GGML_TYPE_TQ3_4S=46`, `GGML_TYPE_TQ3_0=200`
- `ggml/src/ggml-common.h` — `block_tq3_4s` / `block_tq3_1s` struct definitions

### External
- [TurboQuant paper & code](https://github.com/YTan2000/turboquant)
- [Qwen3.6-35B-A3B-MTP-TQ3_4S model](https://huggingface.co/YTan2000/Qwen3.6-35B-A3B-MTP-TQ3_4S)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [CANN documentation](https://www.hiascend.com/en/software/cann)

## Limitations

1. **CPU dequant bottleneck** — Weight dequant runs on CPU, not NPU. For a 35B parameter model, dequantizing all weights on every forward pass will be significantly slower than native GPU kernels. Estimated overhead: ~10-50 ms per token for dequant alone.

2. **No Ascend C kernel yet** — The toolchain (ccec + msopgen) for compiling custom Ascend C kernels has unresolved issues. The `op_kernel/` directory contains a template that compiles on the host side but doesn't produce device `.o` files because the kernel body is empty. A working Ascend C kernel would eliminate the CPU dequant bottleneck.

3. **TQ3_1S not tested** — The TQ3_1S dequant code is implemented (FP16 half-block scales) but untested due to lack of a TQ3_1S model file.

4. **TQ3_0 (KV-cache type, type=200) not supported** — TQ3_0 uses a different quantization scheme for KV-cache and requires `tq3_rotate_act` which is not implemented.

5. **No test model available in this repo** — TQ3 quantized GGUF models are large (~17 GB for the 35B-3B MoE variant) and must be downloaded separately from HuggingFace.

6. **Standalone integration test fails** — The CANN integration test (`test_tq3_cann_integration.cpp`) cannot execute outside of the llama.cpp build tree due to library path issues with `libascend_hal.so`.

7. **WHT activation transform** — The CUDA kernel implements `tq3_rotate_act` to apply WHT to activation vectors for certain inference paths (especially TQ3_0 KV-cache). This is not yet ported.

8. **MMQ / dequant-only paths** — The CUDA backend has two TQ3 paths: a fused dequant+dot kernel (for batch size 1) and a prefill kernel (for larger batches). The current CANN port only implements the prefill-style dequant+matmul path.

## Future Work

| Priority | Task | Benefit |
|----------|------|---------|
| P0 | **Ascend C dequant kernel** — Fill the kernel body in `ascend-c-kernel/` with proper Ascend C code and get it compiling via ccec | ~10-50x speedup for dequant, removes CPU bottleneck |
| P1 | **WHT activation kernel** — Port `tq3_rotate_act` for TQ3_0 KV-cache support | Enables TQ3_0 weight type |
| P2 | **Dequant-only path** — Add option to keep weights dequantized in FP16 device memory across inference steps | Avoids repeated CPU dequant on every token |
| P3 | **Quantized matmul fusion** — Study whether TQ3 dequant can be fused with Ascend C's `Matmul` API to avoid the FP16 intermediate buffer | Further reduces memory bandwidth |

## License

This project is a port/adaptation of the TQ3 kernels from llama.cpp-tq3, which itself builds on [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT License). The port code follows the same license as the original project.
