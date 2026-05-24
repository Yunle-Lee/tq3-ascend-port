// ─── Integration reference: TQ3 matmul for ggml-cann backend ──────────────
// This file documents how TQ3 support plugs into ggml_cann_mul_mat.
// Actual implementation: ggml-cann-integration/tq3_cann.cpp
//
// Flow:
//   1. Detect TQ3_4S / TQ3_1S weight type in ggml_cann_mul_mat
//   2. CPU dequant weights → FP32 via tq3_4s_dequant_block / tq3_1s_dequant_block
//   3. ACL Cast FP32 → FP16 on device
//   4. ACL Cast activations → FP16 (if needed)
//   5. ACL BatchMatMul FP16 [M×K] × [K×N] → [M×N]
//   6. ACL Cast output → destination type (if needed)
//
// Functional status (as of 2026-05):
//   ✅ TQ3_4S dequant + matmul: verified on 2× Ascend 910 NPU
//   ✅ TQ3_1S path: implemented, untested (no model file yet)
//   ❌ TQ3_0 (KV-cache): requires tq3_rotate_act, not yet ported
//
// All helper functions (e3m5_decode, unpack_3bit, wht_32, dequant blocks)
// are consolidated in tq3_cann_kernels.h to avoid duplication between
// tq3_cann.cpp and test files.
//
// Future: replace CPU dequant with Ascend C Tq3Dequant custom operator.

#ifndef GGML_CANN_TQ3_REFERENCE_H
#define GGML_CANN_TQ3_REFERENCE_H

#include "aclnn_ops.h"
#include "common.h"

#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_batch_matmul.h>

void ggml_cann_mul_mat_tq3(ggml_backend_cann_context & ctx, ggml_tensor * dst);

#endif // GGML_CANN_TQ3_REFERENCE_H
