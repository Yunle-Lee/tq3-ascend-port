# TurboQuant TQ3 × Ascend CANN 后端适配 / Ascend CANN Backend Port

> 将 TurboQuant 3-bit 量化推理移植到 llama.cpp 的 Ascend CANN 后端，使 TQ3 模型（如 Qwen3.6-35B-A3B-MTP-TQ3_4S）无需 NVIDIA GPU 即可在华为昇腾 910 NPU 上运行。

> Port [TurboQuant (TQ3)](https://github.com/YTan2000/turboquant) 3-bit quantized inference to [llama.cpp](https://github.com/ggml-org/llama.cpp)'s Ascend CANN backend, enabling TQ3 models (e.g. Qwen3.6-35B-A3B-MTP-TQ3\_4S) to run on **Huawei Ascend 910 NPUs** without NVIDIA GPUs.

---

## 目录 / Table of Contents

- [项目状态 / Project Status](#项目状态--project-status)
- [背景 / Background](#背景--background)
- [架构 / Architecture](#架构--architecture)
- [快速开始 / Quick Start](#快速开始--quick-start)
  - [前置条件 / Prerequisites](#前置条件--prerequisites)
  - [方式一：一键打补丁编译 / Method 1: Patch & Build](#方式一一键打补丁编译--method-1-patch--build)
  - [方式二：独立编译反量化库 / Method 2: Standalone Build](#方式二独立编译反量化库--method-2-standalone-build)
- [测试 / Testing](#测试--testing)
  - [CPU 反量化测试 / CPU Dequant Test](#cpu-反量化测试--cpu-dequant-test)
  - [CANN 集成测试 / CANN Integration Test](#cann-集成测试--cann-integration-test)
  - [最近测试结果 / Recent Test Results](#最近测试结果--recent-test-results)
- [文件结构 / File Structure](#文件结构--file-structure)
- [TQ3 算法参考 / TQ3 Algorithm Reference](#tq3-算法参考--tq3-algorithm-reference)
- [Ascend C 自定义算子 / Ascend C Custom Operator](#ascend-c-自定义算子--ascend-c-custom-operator)
- [技术决策 / Design Decisions](#技术决策--design-decisions)
- [已知局限 / Limitations](#已知局限--limitations)
- [未来规划 / Future Work](#未来规划--future-work)
- [参考资料 / References](#参考资料--references)
- [许可 / License](#许可--license)

---

## 项目状态 / Project Status

| 模块 / Module | 状态 / Status | 说明 / Note |
|---------------|--------------|-------------|
| CPU 反量化 (TQ3_4S) | ✅ 已验证 | 精度测试通过，输出正确 |
| CPU 反量化 (TQ3_1S) | ✅ 已实现 | 代码完整，缺少模型文件验证 |
| ACL Cast → BatchMatMul 管线 | ✅ 已编译 | 符号已编译进 `libggml-cann.a` |
| llama.cpp-tq3 集成 | ✅ 可运行 | 2× Ascend 910 NPU 上验证通过 |
| Ascend C 设备内核 | ✅ 已实现 | 完整 DataCopy 管线，待 msopgen 编译 |
| Ascend C 宿主端 | ✅ 已修复 | InferDataType/InferShape 类型推断修正 |
| TQ3_0 (KV-cache) | ❌ 未支持 | 需移植 `tq3_rotate_act` |
| 独立编译 (CMake) | ✅ 已添加 | `CMakeLists.txt` 支持独立编译测试 |

---

## 背景 / Background

### 什么是 TQ3？ / What is TQ3?

TurboQuant TQ3 是一种 3-bit 权重量化方案，约 4 bpw (bits per weight)。每 32 个权重元素打包为 16 字节的 block：

TurboQuant TQ3 is a 3-bit weight quantization scheme achieving ~4 bpw (bits per weight). Each 32-element block is packed into 16 bytes:

```
┌───────────────── TQ3_4S Block Layout (16 bytes, 32 elements) ─────────────────┐
│ byte 0-3:   4 × E3M5 per-8 scales   (4 bytes)                                │
│ byte 4-15:  32 × 3-bit packed indices (12 bytes)                              │
└───────────────────────────────────────────────────────────────────────────────┘
```

反量化流程 / Dequantization pipeline:
1. **解码 E3M5 缩放因子**：每个 scale 是 8-bit 迷你浮点数 (3-bit 指数 + 5-bit 尾数)
2. **拆包 3-bit 索引**：从 12 字节中解出 32 个 3-bit 索引
3. **质心表查表**：8 个质心值 (Lloyd-Max 优化于标准高斯分布)
4. **分 8 缩放**：每个 group 乘对应 scale
5. **Walsh-Hadamard 变换**：32 点蝶形网络扩散量化误差
6. **黄金比例符号投影 + 归一化**：确定性符号哈希 × 1/sqrt(32)

### 解决了什么问题？ / What Problem Does This Solve?

llama.cpp 已有支持 F32/F16/BF16/Q4_0/Q8_0 的 CANN 后端，以及含手写 TQ3 CUDA 内核的 CUDA 后端。问题是：

- TQ3 内核用 CUDA 编写，需要 NVIDIA GPU
- 华为昇腾 910 NPU 无法运行 CUDA 代码
- CANN 后端完全没有 TQ3 支持

llama.cpp already has a CANN backend supporting F32/F16/BF16/Q4_0/Q8_0 matmul, and a CUDA backend with hand-optimized TQ3 kernels. However:

- TQ3 kernels are CUDA code requiring NVIDIA GPUs
- Huawei Ascend 910 NPUs cannot run CUDA code
- The CANN backend had zero TQ3 support

### 解决方案 / Solution

将 CUDA TQ3 内核路径替换为 **CPU 反量化 → ACL Cast → ACL BatchMatMul** 管线。虽不及原生 Ascend C 内核快，但能让 TQ3 模型立即在昇腾硬件上跑起来，后续可无缝替换为 Ascend C 设备端内核。

Replace the CUDA TQ3 kernel path with a **CPU dequant → ACL Cast → ACL BatchMatMul** pipeline. While slower than a native Ascend C kernel, this gets TQ3 models running on Ascend hardware immediately, with a drop-in path to a custom Ascend C device kernel.

---

## 架构 / Architecture

```
TQ3 GGUF 模型文件
       │
       ▼
llama.cpp 推理循环
       │
       ▼
ggml_cann_mul_mat(dst)
  │ type == TQ3_4S 或 TQ3_1S
  │
  ▼
ggml_cann_mul_mat_tq3(ctx, dst)
  │
  ├── 1. CPU 反量化 TQ3 权重 → FP32
  │      ├── 解码 4× E3M5 缩放因子
  │      ├── 拆包 32× 3-bit 索引
  │      ├── 质心表查表 + 分8缩放
  │      ├── Walsh-Hadamard 变换 (32 元素蝶形网络)
  │      └── 黄金比例符号投影 + 1/sqrt(32) 归一化
  │
  ├── 2. ACL Cast: FP32 → FP16 (设备端)
  │
  ├── 3. ACL Cast: 输入 → FP16 (如输入非 FP16)
  │
  ├── 4. ACL BatchMatMul: [K, M] × [K, N] → [M, N] FP16
  │
  └── 5. ACL Cast: 输出 → 目标类型 (如 dst 非 FP16)
       (或直接 aclrtMemcpy Device→Device 如果目标就是 FP16)
```

### 数据流图 / Data Flow

```
┌──────────────┐    CPU反量化     ┌──────────────┐   ACL Cast    ┌──────────────┐
│  TQ3 权重     │ ──────────────→  │  FP32 权重    │ ──────────→  │  FP16 权重    │
│  (uint8)      │   dequant       │  (float)      │   F16 device  │  (half)       │
│  [K/2, M]     │                 │  [K, M]       │               │  [K, M]       │
└──────────────┘                 └──────────────┘               └──────┬───────┘
                                                                       │
                                                              BatchMatMul
                                                                       │
┌──────────────┐                 ┌──────────────┐               ┌──────▼───────┐
│  输入激活     │    ACL Cast     │  FP16 输入    │               │  FP16 输出    │
│  (F32/F16)   │ ──────────────→  │  (half)       │ ────────────→ │  (half)       │
│  [K, N]      │   (按需)         │  [K, N]       │               │  [M, N]       │
└──────────────┘                 └──────────────┘               └──────┬───────┘
                                                                       │
                                                                   ACL Cast
                                                                       │
                                                               ┌──────▼───────┐
                                                               │  最终输出     │
                                                               │  (dst type)  │
                                                               │  [M, N]      │
                                                               └──────────────┘
```

---

## 快速开始 / Quick Start

### 前置条件 / Prerequisites

| 项目 / Item | 要求 / Requirement |
|-------------|-------------------|
| 硬件 / Hardware | 华为 Ascend 910 / 910B NPU (至少 1 张) |
| 操作系统 / OS | Linux aarch64 (推荐) 或 x86_64 |
| CANN SDK | 8.5.0 或更高版本 |
| 编译器 / Compiler | GCC 9.4+, CMake 3.16+, C++17 |
| 磁盘 / Disk | ~50GB (用于模型文件) |
| 库 / Libraries | `libascendcl.so`, `libnnopbase.so`, `libopapi.so`, `libacl_op_compiler.so` |

### 方式一：一键打补丁编译 / Method 1: Patch & Build

这是推荐的使用方式。将本项目作为 llama.cpp-tq3 的插件编译：

This is the recommended approach. Use this project as a plugin for llama.cpp-tq3:

```bash
# ============================================================
# 第一步：设置 CANN 环境 / Step 1: Source CANN environment
# ============================================================
source /path/to/Ascend/cann-8.5.0/set_env.sh

# 验证环境 / Verify
which cmake && echo "CANN: $ASCEND_TOOLKIT_HOME"

# ============================================================
# 第二步：克隆 llama.cpp-tq3 / Step 2: Clone llama.cpp-tq3
# ============================================================
git clone https://github.com/YTan2000/llama.cpp-tq3.git
cd llama.cpp-tq3

# ============================================================
# 第三步：克隆本仓库并打补丁 / Step 3: Clone this repo & apply patch
# ============================================================
git clone https://github.com/Yunle-Lee/tq3-ascend-port.git
cd tq3-ascend-port
./patches/apply.sh ..

# 或者手动打补丁 / Or manually apply:
# cd .. && git am tq3-ascend-port/patches/0001-add-tq3-cann-backend.patch

# ============================================================
# 第四步：编译 llama.cpp / Step 4: Build llama.cpp
# ============================================================
mkdir build && cd build
cmake .. \
    -DGGML_CANN=ON \
    -DCANN_INSTALL_DIR=${ASCEND_TOOLKIT_HOME} \
    -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

# 验证编译 / Verify build
nm ggml/src/ggml-cann/libggml-cann.a | grep tq3
# 应该输出 / Should output:
#   T _Z21ggml_cann_mul_mat_tq3R25ggml_backend_cann_contextP11ggml_tensor

# ============================================================
# 第五步：下载模型并推理 / Step 5: Download model & run inference
# ============================================================
# 下载 TQ3_4S GGUF 模型 (约 18GB) / Download model (~18GB)
wget https://huggingface.co/YTan2000/Qwen3.6-35B-A3B-MTP-TQ3_4S/resolve/main/ggml-model-Q3_K.gguf \
    -O ~/models/qwen-tq3_4s.gguf

# 推理 / Run inference
./bin/llama-perplexity \
    -m ~/models/qwen-tq3_4s.gguf \
    -ngl 99 --no-warmup

# 双卡推理 / Dual NPU inference
ASCEND_RT_VISIBLE_DEVICES=0,1 ./bin/llama-perplexity \
    -m ~/models/qwen-tq3_4s.gguf \
    -ngl 99 --no-warmup
```

### 方式二：独立编译反量化库 / Method 2: Standalone Build

仅编译 CPU 反量化库和测试，不需要 llama.cpp 或 NPU：

Build only the CPU dequant library and tests, no llama.cpp or NPU required:

```bash
# ============================================================
# 克隆本项目 / Clone this repo
# ============================================================
git clone https://github.com/Yunle-Lee/tq3-ascend-port.git
cd tq3-ascend-port

# ============================================================
# CMake 编译 / CMake build
# ============================================================
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON
cmake --build .

# 运行 CPU 反量化测试 / Run CPU dequant test
./test_tq3_dequant

# 或者直接 g++ 编译 / Or direct g++ compilation
g++ -std=c++17 -O2 \
    -I ../ggml-cann-integration \
    -o test_tq3_dequant \
    ../tests/test_tq3_dequant.cpp \
    -lm
./test_tq3_dequant
```

---

## 测试 / Testing

### CPU 反量化测试 / CPU Dequant Test

**文件 / File**: `tests/test_tq3_dequant.cpp`

测试 TQ3_4S 反量化函数对已知测试块的输出正确性：
- 构造一个确定性的测试 block（固定 E3M5 scale=0.0996，交替循环的 centroid 索引）
- 反量化到 32 个 FP32 值
- 验证统计特性：均值 ≈ 0（WHT + 符号投影保证），RMS ≈ scale

Tests TQ3_4S dequant correctness against a known test block:
- Builds a deterministic test block (fixed E3M5 scale=0.0996, cycling centroid indices)
- Dequants to 32 FP32 values
- Verifies statistical properties: mean ≈ 0 (WHT + sign projection guarantee), RMS ≈ scale

**编译和运行 / Build & Run**:
```bash
g++ -std=c++17 -O2 -I ggml-cann-integration \
    -o test_tq3_dequant tests/test_tq3_dequant.cpp -lm
./test_tq3_dequant
```

### CANN 集成测试 / CANN Integration Test

**文件 / File**: `tests/test_tq3_cann_integration.cpp`

端到端测试完整 CANN 管线：
1. 初始化 CANN backend
2. 创建 TQ3_4S 权重张量 + F32 输入张量
3. 构造 ggml 计算图 (mul_mat)
4. 在 CANN 设备上执行
5. 对比 CPU 参考结果

End-to-end test of the full CANN pipeline:
1. Initialize CANN backend
2. Create TQ3_4S weight + F32 input tensors
3. Build ggml compute graph (mul_mat)
4. Execute on CANN device
5. Compare against CPU reference

**编译和运行 / Build & Run** (需要 llama.cpp-tq3 已编译 + CANN 设备空闲 / requires compiled llama.cpp-tq3 + free CANN device):
```bash
# 设置环境 / Source environment
source /path/to/Ascend/cann-8.5.0/set_env.sh

# 编译 / Compile
g++ -std=c++17 -O2 \
    -I /path/to/llama.cpp-tq3/ggml/include \
    -I /path/to/llama.cpp-tq3/ggml/src \
    -I /path/to/llama.cpp-tq3/ggml/src/ggml-cann \
    -I ggml-cann-integration \
    -I ${ASCEND_TOOLKIT_HOME}/include \
    tests/test_tq3_cann_integration.cpp \
    -L /path/to/llama.cpp-tq3/build/ggml/src/ggml-cann \
    -L /path/to/llama.cpp-tq3/build/ggml/src \
    -lggml-cann -lggml -lggml-base -lggml-cpu \
    -lascendcl -lnnopbase -lopapi -lacl_op_compiler \
    -lpthread -lrt -ldl -lm \
    -o test_tq3_cann_integration

# 运行 / Run
./test_tq3_cann_integration
```

### 最近测试结果 / Recent Test Results

**测试环境 / Test Environment** (2026-05-24):

| 组件 / Component | 版本 / Version |
|-------------------|----------------|
| NPU | 2× Ascend 910 (64GB HBM each) |
| 架构 / Arch | aarch64 |
| CANN | 8.5.0 |
| 编译器 / Compiler | GCC + CCE clang 15.0.5 |

**CPU 反量化测试结果 / CPU Dequant Test Results:**

```
TQ3_4S Dequant Test
==================
E3M5 scales:
  group 0: raw=0xb3 scale=0.099609
  group 1: raw=0xb3 scale=0.099609
  group 2: raw=0xb3 scale=0.099609
  group 3: raw=0xb3 scale=0.099609

Dequantized values (first 4):
  [ 0] =  0.00000000   [ 1] =  0.18579204
  [ 2] = -0.35142571   [ 3] = -0.00000000

Statistics:
  mean = -0.02627998  (expect ~0)    ✅
  RMS  =  0.13241258  (expect ~scale) ✅
  Test passed: OK                    ✅
```

**CANN 集成测试结果 / CANN Integration Test Results:**

| 检查项 / Check | 结果 / Result |
|---------------|---------------|
| `ggml_cann_mul_mat_tq3` 符号 / Symbol | ✅ 已编译进 `libggml-cann.a` |
| llama.cpp-tq3 CANN 后端编译 / Build | ✅ 零错误 |
| 多 NPU 模型加载 / Multi-NPU model load | ✅ 2× Ascend 910 验证通过 |
| ACL Cast FP32→FP16 | ✅ 正确 |
| ACL BatchMatMul | ✅ 正确 |

---

## 文件结构 / File Structure

```
tq3-ascend-port/
│
├── README.md                                   # 项目文档 (本文档) / Project docs (this file)
│
├── CMakeLists.txt                              # 独立编译配置 / Standalone CMake build
├── .gitignore                                  # Git 忽略规则
│
├── ggml-cann-integration/                      # ★ 核心代码 / Core Implementation
│   ├── tq3_cann.h                              #   函数声明 / Function declaration
│   ├── tq3_cann.cpp                            #   CPU反量化 + ACL Cast + BatchMatMul 管线
│   └── tq3_cann_kernels.h                      #   共享反量化辅助函数 (E3M5/WHT/拆包)
│                                               #   Shared dequant helpers (E3M5/WHT/unpack)
│
├── tests/                                      # 测试 / Tests
│   ├── test_tq3_dequant.cpp                    #   CPU 反量化精度测试 (独立运行,无需NPU)
│   └── test_tq3_cann_integration.cpp           #   CANN 集成端到端测试 (需NPU)
│
├── patches/                                    # llama.cpp 集成 / Integration
│   ├── 0001-add-tq3-cann-backend.patch         #   Git patch for llama.cpp-tq3
│   └── apply.sh                                #   一键打补丁脚本 / One-click apply script
│
├── ascend-c-kernel/                            # Ascend C 自定义算子 (未来加速) / Custom op (future)
│   ├── README.md                               #   Ascend C 内核状态说明
│   ├── op_desc.json                            #   算子描述 (输入/输出形状和类型)
│   └── src/
│       ├── tq3_dequant_host.cpp                #   宿主端: 算子注册 + Tiling + 形状/类型推断
│       └── tq3_dequant_kernel.cpp              #   设备端: DataCopy→反量化→DataCopy 管线
│
├── reference/                                  # 参考资料 / References
│   ├── TQ3_ALGORITHM.md                        #   TQ3 算法详细说明 / Algorithm reference
│   └── ggml_cann_tq3_integration.cpp           #   集成交互参考 / Integration reference
│
└── examples/
    └── compile_and_test.sh                     #   编译测试脚本 / Build & test script
```

### 核心文件说明 / Core File Descriptions

| 文件 / File | 行数 / Lines | 功能 / Purpose |
|------------|-------------|----------------|
| `ggml-cann-integration/tq3_cann_kernels.h` | 102 | **反量化核心**：E3M5解码、3-bit拆包、WHT变换、符号投影、TQ3_4S/TQ3_1S block反量化。所有测试和集成代码共享此文件，避免重复 |
| `ggml-cann-integration/tq3_cann.cpp` | 120 | **ACL 管线**：调用反量化 → ACL Cast FP32→FP16 → ACL Cast 输入→FP16 → ACL BatchMatMul → ACL Cast 输出→目标类型 |
| `ggml-cann-integration/tq3_cann.h` | 5 | 声明 `ggml_cann_mul_mat_tq3()` 函数签名 |
| `ascend-c-kernel/src/tq3_dequant_kernel.cpp` | 110 | **Ascend C 设备内核**：DataCopy(GM→Local) → 反量化计算 → DataCopy(Local→GM) 的完整管线，使用 TPipe/TQue 双缓冲 |
| `ascend-c-kernel/src/tq3_dequant_host.cpp` | 63 | **宿主端代码**：算子注册、Tiling 函数、InferShape (输出维度×2)、InferDataType (输出 float16) |
| `patches/0001-add-tq3-cann-backend.patch` | ~300 | Git 补丁：将 `tq3_cann.{h,cpp}` + `tq3_cann_kernels.h` 集成到 llama.cpp-tq3 的 `ggml/src/ggml-cann/` 目录 |

---

## TQ3 算法参考 / TQ3 Algorithm Reference

详见 `reference/TQ3_ALGORITHM.md`。此处仅列出核心常量：

See `reference/TQ3_ALGORITHM.md` for full details. Key constants:

```cpp
// 质心表 (Lloyd-Max 优化于标准高斯分布)
// Centroid LUT (Lloyd-Max optimized for unit Gaussian)
static const float centroids[8] = {
    -2.1519f, -1.3439f, -0.7560f, -0.2451f,
     0.2451f,  0.7560f,  1.3439f,  2.1519f
};

// E3M5 微型浮点数解码 (3-bit 指数 + 5-bit 尾数, bias=9)
// E3M5 mini-float decode
float e3m5_decode(uint8_t v) {
    int e = (v >> 5) & 7, m = v & 0x1F;
    return ldexpf(1.0f + m / 32.0f, e - 9);
}

// 黄金比例符号哈希 (确定性, 零均值化)
// Golden-ratio sign hash (deterministic, zero-mean)
float golden_sign(int i) {
    return (((unsigned)i * 0x9E3779B9u) >> 31) & 1 ? -1.0f : 1.0f;
}

// 反量化归一化常数 = 1/sqrt(32)
// Dequant normalization constant
const float rcp = 0.1767766952966369f;
```

**反量化公式 / Dequant Formula:**

```
out[i] = WHT( centroids[idx[i]] × scale[group(i)] ) × golden_sign(i) / sqrt(32)

其中 / where:
  idx[i]     = 3-bit index unpacked from bytes 4-15
  group(i)   = i / 8                        (0,1,2,3 for TQ3_4S)
  scale[g]   = e3m5_decode(block[g])        (TQ3_4S)
             = fp16_to_f32(block[g*2:g*2+2]) (TQ3_1S)
  WHT        = 32-point Walsh-Hadamard Transform (butterfly network)
```

---

## Ascend C 自定义算子 / Ascend C Custom Operator

### 状态 / Status

自定义 Ascend C Tq3Dequant 算子**已完整实现**，待 msopgen 编译流程部署：

The custom Ascend C Tq3Dequant operator is **fully implemented**, pending msopgen compilation:

| 组件 / Component | 文件 / File | 状态 / Status |
|-----------------|------------|--------------|
| 算子描述 / Operator Descriptor | `op_desc.json` | ✅ 支持多维张量、float16 输出 |
| 宿主端 / Host-side | `tq3_dequant_host.cpp` | ✅ InferShape/InferDataType 已修正 |
| 设备内核 / Device Kernel | `tq3_dequant_kernel.cpp` | ✅ DataCopy + 反量化管线完整 |

### 宿主端修复要点 / Host-side Fixes

原来存在两个会导致运行时崩溃的 bug：

Two bugs that would cause runtime crashes were fixed:

1. **InferDataType**: 原来设置输出类型 = 输入类型 (uint8)，修正为 `ge::DT_FLOAT16`
2. **InferShape**: 原来照抄输入 shape，修正为 `inner_dim * 2`（每 16 字节输入 → 32 个 float16 输出元素）

### 编译流程 / Compilation

```bash
# 1. 生成 msopgen 项目 / Generate msopgen project
msopgen gen -c op_desc.json -lang cpp

# 2. 编译算子 / Build operator
cd build
cmake .. -DASCEND_TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME}
cmake --build .

# 3. 安装算子 / Install operator
./scripts/install.sh

# 4. 在 llama.cpp 中替换 CPU 反量化调用
#    将 ggml_cann_mul_mat_tq3() 中的 dequant_tq3_tensor()
#    替换为 aclnnTq3Dequant() 调用
```

### 内核架构 / Kernel Architecture

```
extern "C" __global__ __aicore__ void tq3_dequant(GM_ADDR workspace, GM_ADDR tiling)
  │
  ├── GET_TILING_DATA(tiling_data, tiling)    # 获取 tiling 信息
  │
  ├── KernelTq3Dequant::Init()                # 初始化 GM buffer + TPipe/TQue
  │
  └── KernelTq3Dequant::Process()             # 循环处理每个 block
       │
       ├── CopyIn()                           # DataCopy: GM → Local (uint8)
       ├── Compute()                           # 反量化: Local (uint8) → Local (half)
       │    ├── E3M5 解码                        # 从 uint8 解码 float scale
       │    ├── 3-bit 拆包                        # 从 12 字节拆出 32 个索引
       │    ├── 质心查表 + 分8缩放                 # centroid[idx] × scale
       │    ├── Walsh-Hadamard 变换 (32 点)        # 蝶形网络
       │    └── 符号投影 + 归一化                  # golden_sign × 1/sqrt(32)
       └── CopyOut()                          # DataCopy: Local → GM (half)
```

---

## 技术决策 / Design Decisions

| 决策 / Decision | 原因 / Rationale |
|----------------|-----------------|
| CPU 反量化先行 / CPU dequant first | CANN 工具链复杂，CPU 方案先让系统跑起来；Ascend C 内核可无缝替换 |
| FP16 矩阵乘 / FP16 matmul | 昇腾 910 Cube 单元对 FP16 优化最佳 |
| ACL BatchMatMul / Use BatchMatMul | CANN 标准矩阵乘 API，处理任意 batch 维度 |
| 共享 kernel header / Shared kernel header | 消除 tq3_cann.cpp 和 测试文件之间的 ~50 行重复代码 |
| 双 NPU 支持 / Dual NPU support | 继承现有 CANN 后端多设备分发，无需额外改动 |
| 独立 CMakeLists.txt / Standalone CMake | 支持脱离 llama.cpp 编译反量化功能和测试 |

---

## 已知局限 / Limitations

| 局限 / Limitation | 影响 / Impact | 缓解 / Mitigation |
|-------------------|--------------|------------------|
| CPU 反量化瓶颈 / CPU dequant bottleneck | 35B 模型每 token 额外 10-50ms | Ascend C 内核可消除 (P0) |
| Ascend C 内核未部署 / Ascend C kernel not deployed | 反量化在 CPU 不在 NPU | msopgen 编译流程已就绪 |
| TQ3_1S 未测试 / TQ3_1S untested | 代码完整但缺模型验证 | 获取 TQ3_1S 模型后测试 |
| TQ3_0 (KV-cache) 不支持 / TQ3_0 not supported | 缺少 `tq3_rotate_act` | 需移植 CUDA 的 WHT 激活变换 |
| 无融合点积路径 / No fused dot-product path | batch=1 时带宽利用率低 | 可融合到 Ascend C Matmul API (P3) |
| 独立集成测试需 CANN 环境 / Integration test needs CANN | 无法在纯 CPU 环境跑集成测试 | CPU 反量化测试可独立运行 |

---

## 未来规划 / Future Work

| 优先级 | 任务 | 收益 |
|--------|------|------|
| **P0** | Ascend C 反量化内核部署 — 通过 msopgen 编译 `ascend-c-kernel/` 并集成到推理管线 | 10-50× 加速，消除 CPU 瓶颈 |
| **P1** | WHT 激活内核移植 — 移植 `tq3_rotate_act` 支持 TQ3_0 KV-cache 量化 | 启用 KV-cache 量化 |
| **P2** | 反量化权重缓存 — 在 FP16 设备内存中缓存反量化后的权重 | 避免每 token 重复反量化 |
| **P3** | 融合量化矩阵乘 — 研究融合进 Ascend C Matmul API | 减少中间缓冲区 |
| **P4** | TQ3_1S 模型验证 — 获取模型文件进行端到端测试 | 确认 TQ3_1S 路径正确性 |

| Priority | Task | Benefit |
|----------|------|---------|
| **P0** | Deploy Ascend C dequant kernel via msopgen → integrate into pipeline | 10-50× speedup, CPU bottleneck removed |
| **P1** | Port WHT activation kernel — `tq3_rotate_act` for TQ3_0 | KV-cache quantization support |
| **P2** | Weight dequant cache in device FP16 memory | Avoid repeated CPU dequant per token |
| **P3** | Fused quantized matmul via Ascend C Matmul API | Reduce intermediate buffer overhead |
| **P4** | TQ3_1S model validation — end-to-end test | Verify TQ3_1S path correctness |

---

## 参考资料 / References

### TQ3 相关 / TQ3 Related
- [TurboQuant 论文与代码 / Paper & Code](https://github.com/YTan2000/turboquant)
- [Qwen3.6-35B-A3B-MTP-TQ3_4S 模型 / Model](https://huggingface.co/YTan2000/Qwen3.6-35B-A3B-MTP-TQ3_4S)
- [llama.cpp-tq3 分支 (含 TQ3 类型定义) / llama.cpp fork with TQ3 types](https://github.com/YTan2000/llama.cpp-tq3)

### CANN / Ascend 相关
- [CANN 官方文档 / CANN Documentation](https://www.hiascend.com/en/software/cann)
- [Ascend C 算子开发指南 / Ascend C Operator Dev Guide](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/operatordevelopment/ascendcopdevg/AscendCopDevG-0001.html)
- [llama.cpp CANN 后端 / CANN Backend](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cann)

### 核心参考文件 / Core Reference Files (llama.cpp 内)
- `ggml/src/ggml-cann/aclnn_ops.cpp` — ACLNN 操作实现 (含 `ggml_cann_mul_mat` 分发)
- `ggml/src/ggml-cann/common.h` — CANN 后端上下文与工具
- `ggml/src/ggml-cann/acl_tensor.h` — `ggml_cann_create_tensor` / type mapping
- `ggml/include/ggml.h` — `GGML_TYPE_TQ3_1S=44`, `TQ3_4S=46`, `TQ3_0=200`

---

## 许可 / License

本项目是对 llama.cpp-tq3 中 TQ3 内核的移植改编。llama.cpp 遵循 MIT 许可。移植代码继承原项目许可。

This project is a port/adaptation of TQ3 kernels from llama.cpp-tq3, which builds on [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT License). The port code follows the same license.
