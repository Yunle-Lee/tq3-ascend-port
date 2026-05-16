# TurboQuant TQ3 × Ascend CANN 后端适配

> 将 TurboQuant 3-bit 量化推理移植到 llama.cpp 的 Ascend CANN 后端，使 TQ3 模型（如 Qwen3.6-35B-A3B-MTP-TQ3_4S）无需 NVIDIA GPU 即可在昇腾 910 NPU 上运行。

---

## 目录（中文）

- [背景](#背景)
- [我们做了什么](#我们做了什么)
- [架构](#架构)
- [快速开始](#快速开始)
- [文件说明](#文件说明)
- [参考资料](#参考资料)
- [局限性](#局限性)
- [后续工作](#后续工作)

---

## 背景

### TurboQuant TQ3

TurboQuant 是一种 3-bit 权重量化方案，约 4 bpw。其核心流程：

- **3-bit 打包索引**：12 字节存储 32 个索引，查 8 值质心表
- **分 8 缩放**：每块 4 个 E3M5 微型浮点数（TQ3_4S）或 2 个 FP16 半块缩放（TQ3_1S）
- **Walsh-Hadamard 变换（WHT）**：蝶形网络扩散量化误差
- **黄金比例符号投影**：确定性符号哈希，使输出零均值化
- **块大小**：32 元素，16 字节每块

### 解决的问题

llama.cpp 已有支持 F32/F16/BF16/Q4_0/Q8_0 的 CANN 后端（`ggml/src/ggml-cann/`），以及含有手写 TQ3 CUDA 内核（`tq3-native.cu/.cuh`）的 CUDA 后端。问题是：

- TQ3 内核是用 CUDA 写的，需要 NVIDIA GPU
- 昇腾 910 NPU 无法运行 CUDA
- CANN 后端完全没有 TQ3 支持

### 方案

将 CUDA TQ3 内核路径替换为 **CPU 反量化 → ACL Cast → ACL BatchMatMul** 管线，集成到现有 CANN 后端中。虽不及原生 Ascend C 内核快，但能让 TQ3 模型立即在昇腾硬件上跑起来。

---

## 我们做了什么

| 步骤 | 状态 |
|------|------|
| 阅读并理解 CUDA TQ3 内核代码 | ✅ |
| 理解 TQ3_4S 块格式（E3M5、3-bit 拆包、WHT、符号投影） | ✅ |
| 编写 CPU TQ3_4S 反量化实现并通过测试 | ✅ |
| 集成到 `ggml_cann_mul_mat` 分发中 | ✅ |
| 编译 `ggml-cann` 库零错误 | ✅ |
| 多 NPU 模型加载验证通过 | ✅ |
| 通过 msopgen 生成 Ascend C 内核模板（设备内核编译待修复） | 🟡 |

### 新增文件

| 文件 | 行数 | 用途 |
|------|------|------|
| `ggml-cann-integration/tq3_cann.h` | 3 | 函数声明 |
| `ggml-cann-integration/tq3_cann.cpp` | ~210 | CPU 反量化 + ACL Cast + BatchMatMul |

### 修改文件

| 文件 | 改动 | 用途 |
|------|------|------|
| `aclnn_ops.cpp` | switch 中 +3 行 | 分发 TQ3_4S / TQ3_1S |

---

## 架构

```
TQ3 GGUF 模型
       │
       ▼
llama.cpp 推理循环
       │
       ▼
ggml_cann_mul_mat(dst)
  │ type = TQ3_4S / TQ3_1S
  │
  ▼
ggml_cann_mul_mat_tq3(ctx, dst)
  │
  ├── 1. CPU 反量化 TQ3 权重 → FP32
  │      ├── 解码 4× E3M5 缩放因子
  │      ├── 拆包 32× 3-bit 索引
  │      ├── 质心表查表 + 分 8 缩放
  │      ├── Walsh-Hadamard 变换（32 元素）
  │      └── 黄金比例符号投影 + 1/sqrt(32) 归一化
  │
  ├── 2. ACL Cast FP32 → FP16（设备端）
  │
  ├── 3. ACL Cast 输入 → FP16（如需要）
  │
  ├── 4. ACL BatchMatMul [M×K] × [K×N] → [M×N] FP16
  │
  └── 5. ACL Cast 输出 → 目标类型（如需要）
```

### 关键设计决策

1. **CPU 反量化而非 Ascend C 内核** — CANN 工具链（ccec）编译复杂，CPU 方案先让系统跑起来
2. **FP16 矩阵乘** — 昇腾 910 的 Cube 单元对 FP16 优化最佳
3. **ACL BatchMatMul** — CANN 通用矩阵乘标准 API
4. **双 NPU 支持** — 继承现有 CANN 后端的多设备分发，无需改动

---

## 快速开始

### 前置条件

- **硬件**：昇腾 910（或 910B）NPU
- **软件**：CANN 8.5.0+、CMake 3.16+、GCC 9.4+

### 编译

```bash
# 1. 加载 CANN 环境
source /path/to/Ascend/cann-8.5.0/set_env.sh

# 2. 克隆 llama.cpp-tq3（含 TQ3 类型定义）
git clone https://github.com/YTan2000/llama.cpp-tq3.git

# 3. 应用补丁
cd llama.cpp-tq3
git am /path/to/patches/0001-add-tq3-cann-backend.patch

# 4. 编译
mkdir build && cd build
cmake .. -DGGML_CANN=ON \
         -DCANN_INSTALL_DIR=${ASCEND_TOOLKIT_HOME} \
         -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

# 5. 推理测试
./bin/llama-perplexity \
    -m /path/to/qwen3.6-35b-a3b-mtp-tq3_4s.gguf \
    -ngl 99 --no-warmup
```

### CPU 反量化测试（独立运行）

```bash
g++ -std=c++17 -O2 -o test_tq3_dequant \
    tests/test_tq3_dequant.cpp \
    ggml-cann-integration/tq3_cann.cpp -lm
./test_tq3_dequant
```

---

## 文件说明

```
tq3-ascend-port/
├── README.md                                    # 本文档
├── patches/
│   ├── 0001-add-tq3-cann-backend.patch          # Git 补丁
│   └── apply.sh                                 # 一键打补丁
├── ggml-cann-integration/
│   ├── tq3_cann.h                               # 函数声明
│   └── tq3_cann.cpp                             # CPU 反量化 + ACL 算子
├── tests/
│   ├── test_tq3_dequant.cpp                     # CPU 反量化测试
│   └── test_tq3_cann_integration.cpp            # CANN 集成测试
├── reference/
│   ├── ggml_cann_tq3_integration.cpp            # 集成方案设计
│   └── TQ3_ALGORITHM.md                         # TQ3 算法参考
├── ascend-c-kernel/
│   ├── README.md                                # Ascend C 内核状态
│   ├── op_desc.json                             # 算子描述
│   └── src/
│       ├── tq3_dequant_host.cpp                 # 宿主端代码
│       └── tq3_dequant_kernel.cpp               # 设备内核（待实现）
└── examples/
    └── compile_and_test.sh                      # 编译测试脚本
```

---

## 测试环境

| 组件 | 版本 |
|------|------|
| NPU | 2× 昇腾 910（各 64GB HBM） |
| 架构 | aarch64 |
| CANN | 8.5.0 |
| CCE 编译器 | clang 15.0.5（ccec） |
| llama.cpp 分支 | YTan2000/llama.cpp-tq3（TQ3 类型: 44/46/200） |

---

## 参考资料

### TQ3 CUDA 内核（原始代码）
- `tq3-native.cu/.cuh` — TQ3 点积内核
- `tq3-prefill.cuh` — TQ3 prefill 内核
- `turbo-wht.cu` — 通用 WHT 内核

### CANN 后端（适配目标）
- `aclnn_ops.cpp` — ACLNN 操作实现
- `common.h` — CANN 后端上下文与工具
- `ggml-cann.cpp` — 后端初始化与计算分发

### TQ3 类型定义
- `ggml.h` — `GGML_TYPE_TQ3_1S=44`, `TQ3_4S=46`, `TQ3_0=200`
- `ggml-common.h` — `block_tq3_4s` / `block_tq3_1s` 结构体

### 外部链接
- [TurboQuant 论文与代码](https://github.com/YTan2000/turboquant)
- [Qwen3.6-35B-A3B-MTP-TQ3_4S 模型](https://huggingface.co/YTan2000/Qwen3.6-35B-A3B-MTP-TQ3_4S)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [CANN 文档](https://www.hiascend.com/en/software/cann)

---

## 局限性

1. **CPU 反量化性能瓶颈** — 权重反量化在 CPU 上运行，35B 模型每 token 预估额外开销 10-50ms
2. **尚无 Ascend C 内核** — ccec 工具链存在编译问题，设备内核骨架已生成但主体为空
3. **TQ3_1S 未测试** — 代码已实现但缺少 TQ3_1S 模型文件
4. **TQ3_0（KV-cache 类型）不支持** — 需要 `tq3_rotate_act` 尚未移植
5. **无测试模型** — TQ3 GGUF 模型需自行从 HuggingFace 下载
6. **独立集成测试无法运行** — `test_tq3_cann_integration.cpp` 因库路径问题无法脱离构建树执行
7. **WHT 激活变换未移植** — CUDA 内核的 `tq3_rotate_act` 尚未实现
8. **仅实现 dequant+matmul 路径** — 缺乏针对 batch size 1 优化的融合点积路径

---

## 后续工作

| 优先级 | 任务 | 收益 |
|--------|------|------|
| P0 | **Ascend C 反量化内核** — 填充 `ascend-c-kernel/` 中的内核主体并解决 ccec 编译问题 | 加速 10-50×，消除 CPU 瓶颈 |
| P1 | **WHT 激活内核** — 移植 `tq3_rotate_act` 支持 TQ3_0 | 启用 KV-cache 量化 |
| P2 | **反量化缓存** — 在设备 FP16 内存中缓存反量化后的权重 | 避免每 token 重复反量化 |
| P3 | **融合量化矩阵乘** — 研究是否能融合进 Ascend C 的 Matmul API | 减少中间缓冲区 |

---

## 许可

本项目是对 llama.cpp-tq3 中 TQ3 内核的移植改编。llama.cpp 遵循 MIT 许可。移植代码继承原项目许可。

---

# TurboQuant TQ3 × Ascend CANN Backend Port

> Port [TurboQuant (TQ3)](https://github.com/YTan2000/turboquant) 3-bit quantized inference to [llama.cpp](https://github.com/ggml-org/llama.cpp)'s Ascend CANN backend, enabling TQ3 models (e.g. Qwen3.6-35B-A3B-MTP-TQ3\_4S) to run on **Ascend 910 NPUs** without NVIDIA GPUs.

---

## Table of Contents

- [Background](#background)
- [What We Did](#what-we-did)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [File Reference](#file-reference)
- [Environment](#environment)
- [References](#references)
- [Limitations](#limitations)
- [Future Work](#future-work)

---

## Background

### TurboQuant TQ3

TurboQuant is a 3-bit weight quantization scheme achieving ~4 bpw (bits per weight). It uses:

- **3-bit packed indices** (12 bytes → 32 indices) with a centroid LUT of 8 values
- **Per-8 scaling**: 4 E3M5 mini-floats per block (TQ3\_4S) or 2 FP16 half-block scales (TQ3\_1S)
- **Walsh-Hadamard Transform (WHT)**: butterfly network to spread quantization error
- **Golden-ratio sign projection**: deterministic sign hashing for zero-mean output
- **Block size**: 32 elements, 16 bytes per block

### The Problem

llama.cpp already has a CANN backend (`ggml/src/ggml-cann/`) supporting F32/F16/BF16/Q4\_0/Q8\_0 matmul via ACLNN operators, plus a CUDA backend with hand-optimized TQ3 kernels (`tq3-native.cu`/`.cuh`). However:

- TQ3 kernels are written in CUDA and require NVIDIA GPUs
- Ascend 910 NPUs cannot run CUDA code
- No TQ3 support existed in the CANN backend

### Solution

Replace the CUDA TQ3 kernel path with a **CPU dequant → ACL Cast → ACL BatchMatMul** pipeline integrated into the existing CANN backend. While slower than a native Ascend C kernel, this gets TQ3 models running on Ascend hardware immediately, with a clear optimization path to a custom Ascend C dequant kernel.

---

## What We Did

| Step | Status |
|------|--------|
| Read and understood the CUDA TQ3 kernel (`tq3-native.cu`/`.cuh`, `tq3-prefill.cuh`, `turbo-wht.cu`) | ✅ |
| Understood TQ3\_4S block format (E3M5, 3-bit unpacking, WHT, sign projection) | ✅ |
| Wrote a CPU TQ3\_4S dequant implementation and verified against known test vectors | ✅ |
| Integrated into `ggml_cann_mul_mat` dispatch — TQ3\_4S and TQ3\_1S now accepted | ✅ |
| Compiled `ggml-cann` library with zero errors | ✅ |
| Verified multi-NPU model loading with llama.cpp CANN backend | ✅ |
| Wrote Ascend C kernel template via `msopgen` (device kernel compilation pending) | 🟡 |

### Source Files Created/Modified

#### New files

| File | Lines | Purpose |
|------|-------|---------|
| `ggml-cann-integration/tq3_cann.h` | 3 | Function declaration |
| `ggml-cann-integration/tq3_cann.cpp` | ~210 | CPU dequant + ACL Cast + ACL BatchMatMul |

#### Modified files (in llama.cpp's `ggml/src/ggml-cann/`)

| File | Change | Purpose |
|------|--------|---------|
| `aclnn_ops.cpp` | +3 lines in switch | Dispatch `GGML_TYPE_TQ3_4S` / `TQ3_1S` |

No CMake changes needed — `file(GLOB ... "*.cpp")` picks up new files automatically.

---

## Architecture

```
TQ3 GGUF model
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
  │      ├── decode 4× E3M5 per-8 scales
  │      ├── unpack 32× 3-bit indices
  │      ├── centroid LUT lookup + per-8 scaling
  │      ├── Walsh-Hadamard Transform (32 elements)
  │      └── golden-ratio sign projection + 1/sqrt(32) normalize
  │
  ├── 2. ACL Cast FP32 → FP16 (on device)
  │
  ├── 3. ACL Cast input → FP16 (if needed)
  │
  ├── 4. ACL BatchMatMul [M×K] × [K×N] → [M×N] FP16
  │
  └── 5. ACL Cast output → dst type (if needed)
```

### Key Design Decisions

1. **CPU dequant (not Ascend C kernel)** — The CANN toolchain (ccec) for Ascend C kernel compilation is complex. CPU dequant gets the system working immediately; an Ascend C kernel can be swapped in later without changing the matmul pipeline.

2. **FP16 matmul (not F32)** — Ascend 910's Cube units are optimized for FP16. The dequant output is FP32 but immediately cast to FP16 before matmul.

3. **ACL BatchMatMul (not MatMulV2)** — BatchMatMul is the standard CANN API for general matrix multiply. It handles arbitrary batch dimensions correctly.

4. **Dual NPU support** — The existing CANN backend already splits model layers across multiple NPUs. No changes needed for multi-device support.

---

## Quick Start

### Prerequisites

- **Hardware**: Ascend 910 (or 910B) NPU
- **Software**: CANN 8.5.0+, CMake 3.16+, GCC 9.4+

### Build

```bash
# 1. Source CANN environment
source /path/to/Ascend/cann-8.5.0/set_env.sh

# 2. Clone llama.cpp-tq3 (includes TQ3 type definitions)
git clone https://github.com/YTan2000/llama.cpp-tq3.git

# 3. Apply our patch
cd llama.cpp-tq3
git am /path/to/patches/0001-add-tq3-cann-backend.patch

# 4. Build with CANN backend
mkdir build && cd build
cmake .. -DGGML_CANN=ON \
         -DCANN_INSTALL_DIR=${ASCEND_TOOLKIT_HOME} \
         -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

# 5. Run inference
./bin/llama-perplexity \
    -m /path/to/qwen3.6-35b-a3b-mtp-tq3_4s.gguf \
    -ngl 99 --no-warmup
```

### Test CPU Dequant (Standalone)

```bash
g++ -std=c++17 -O2 -o test_tq3_dequant \
    tests/test_tq3_dequant.cpp \
    ggml-cann-integration/tq3_cann.cpp -lm
./test_tq3_dequant
```

---

## File Reference

```
tq3-ascend-port/
├── README.md                                    # This file
├── patches/
│   ├── 0001-add-tq3-cann-backend.patch          # Git patch for llama.cpp-tq3
│   └── apply.sh                                 # One-click patch script
├── ggml-cann-integration/
│   ├── tq3_cann.h                               # Function declaration
│   └── tq3_cann.cpp                             # Dequant + ACL ops
├── tests/
│   ├── test_tq3_dequant.cpp                     # CPU dequant test
│   └── test_tq3_cann_integration.cpp            # CANN integration test
├── reference/
│   ├── ggml_cann_tq3_integration.cpp            # Integration design stub
│   └── TQ3_ALGORITHM.md                         # TQ3 algorithm reference
├── ascend-c-kernel/
│   ├── README.md                                # Ascend C kernel status
│   ├── op_desc.json                             # Operator descriptor
│   └── src/
│       ├── tq3_dequant_host.cpp                 # Host-side code
│       └── tq3_dequant_kernel.cpp               # Device kernel (WIP)
└── examples/
    └── compile_and_test.sh                      # Build & test script
```

---

## Environment

This port was developed and tested on:

| Component | Version |
|-----------|---------|
| NPU | 2× Ascend 910 (64GB HBM each) |
| Architecture | aarch64 |
| CANN | 8.5.0 |
| CCE compiler | clang 15.0.5 (ccec) |
| OS | Linux (aarch64) |
| llama.cpp fork | YTan2000/llama.cpp-tq3 (TQ3 types: 44/46/200) |

---

## References

### TQ3 CUDA Kernel (Original)
- `tq3-native.cu/.cuh` — TQ3 dot product kernel
- `tq3-prefill.cuh` — TQ3 prefill kernel
- `turbo-wht.cu` — Generic WHT kernel
- `mmq-instance-tq3_4s.cu` — MMQ template instantiation

### CANN Backend (Target)
- `aclnn_ops.cpp` — ACLNN operations including `ggml_cann_mul_mat`
- `aclnn_ops.h` — Operation declarations and `GGML_CANN_CALL_ACLNN_OP` macro
- `common.h` — `ggml_backend_cann_context`, helpers
- `acl_tensor.h` — `ggml_cann_create_tensor`, `ggml_cann_type_mapping`
- `ggml-cann.cpp` — Backend init and compute dispatch

### TQ3 Type Definitions
- `ggml.h` — `GGML_TYPE_TQ3_1S=44`, `TQ3_4S=46`, `TQ3_0=200`
- `ggml-common.h` — `block_tq3_4s` / `block_tq3_1s` structs

### External Links
- [TurboQuant paper & code](https://github.com/YTan2000/turboquant)
- [Qwen3.6-35B-A3B-MTP-TQ3_4S model](https://huggingface.co/YTan2000/Qwen3.6-35B-A3B-MTP-TQ3_4S)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [CANN documentation](https://www.hiascend.com/en/software/cann)

---

## Limitations

1. **CPU dequant bottleneck** — Weight dequant runs on CPU, not NPU. Estimated overhead ~10-50 ms per token for a 35B model.
2. **No Ascend C kernel yet** — The toolchain (ccec + msopgen) has unresolved issues; the kernel template body is empty.
3. **TQ3_1S not tested** — Code is implemented but no TQ3_1S model file is available.
4. **TQ3_0 (KV-cache type) not supported** — Requires `tq3_rotate_act` which is not yet ported.
5. **No test model in this repo** — TQ3 GGUF models must be downloaded separately from HuggingFace.
6. **Standalone integration test fails** — Cannot execute outside the llama.cpp build tree due to `libascend_hal.so` path issues.
7. **WHT activation transform not ported** — CUDA's `tq3_rotate_act` is not yet implemented.
8. **Only dequant+matmul path implemented** — Missing the fused dot-product path optimized for batch size 1.

---

## Future Work

| Priority | Task | Benefit |
|----------|------|---------|
| P0 | **Ascend C dequant kernel** — Fill the kernel body in `ascend-c-kernel/` and fix ccec compilation | 10-50× speedup, removes CPU bottleneck |
| P1 | **WHT activation kernel** — Port `tq3_rotate_act` for TQ3_0 | Enables KV-cache quantization |
| P2 | **Dequant cache** — Keep dequantized weights in FP16 device memory across inference steps | Avoids repeated CPU dequant per token |
| P3 | **Fused quantized matmul** — Explore fusion with Ascend C's Matmul API | Reduces intermediate buffer overhead |

---

## License

This project is a port/adaptation of the TQ3 kernels from llama.cpp-tq3, which builds on [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT License). The port code follows the same license as the original project.
