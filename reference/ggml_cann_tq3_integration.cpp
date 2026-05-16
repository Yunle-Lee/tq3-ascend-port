// ─── Integration stub: TQ3 matmul for ggml-cann backend ──────────────────────
// This file shows how to plug TQ3 support into ggml_cann_mul_mat.
// Expected location: llama.cpp-tq3/ggml/src/ggml-cann/
//
// Flow:
//   1. Detect TQ3_4S / TQ3_1S weight type in ggml_cann_mul_mat
//   2. Allocate FP16 device buffer
//   3. CPU-dequant weights → FP32 → cast to FP16 on device (or use ACL cast)
//   4. Call ggml_cann_mat_mul_fp (FP16 matmul)
//
// Future optimization: replace step 3 with Ascend C dequant kernel.

#include "aclnn_ops.h"
#include "common.h"

#include <aclnnop/aclnn_cast.h>

// ─── CPU dequant (declared in tq3_cann_kernels.h) ──────────────────────────
void tq3_4s_dequant_tensor(const uint8_t * src_data, int64_t rows, int64_t cols, float * dst_f32);

// ─── TQ3 matmul for CANN backend ──────────────────────────────────────────
// Called from ggml_cann_mul_mat when src0->type is TQ3_4S or TQ3_1S
void ggml_cann_mul_mat_tq3(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];   // weight (TQ3 quantized)
    ggml_tensor * src1 = dst->src[1];   // activation (FP16/FP32)

    int64_t M = src0->ne[1];            // output rows = weight rows
    int64_t K = src0->ne[0];            // inner dim
    int64_t N = src1->ne[1];            // output cols = tokens

    int64_t num_weights = ggml_nelements(src0);

    // Step 1: CPU dequant weights → FP32
    size_t f32_size = num_weights * sizeof(float);
    ggml_cann_pool_alloc f32_buf(ctx.pool());
    float * f32_weights = (float *)f32_buf.alloc(f32_size);

    tq3_4s_dequant_tensor(src0, f32_weights);

    // Step 2: Upload FP32 → device, cast to FP16
    size_t f16_size = num_weights * sizeof(uint16_t);
    ggml_cann_pool_alloc f16_buf(ctx.pool());
    void * f16_weights = f16_buf.alloc(f16_size);
    {
        int64_t ne[]     = { K, M };
        size_t  nb_f32[] = { sizeof(float), K * sizeof(float) };
        size_t  nb_f16[] = { sizeof(uint16_t), K * sizeof(uint16_t) };
        acl_tensor_ptr acl_f32 = ggml_cann_create_tensor(
            f32_weights, ACL_FLOAT, sizeof(float), ne, nb_f32, 2);
        acl_tensor_ptr acl_f16 = ggml_cann_create_tensor(
            f16_weights, ACL_FLOAT16, sizeof(uint16_t), ne, nb_f16, 2);
        aclnn_cast(ctx, acl_f32.get(), acl_f16.get(), ACL_FLOAT16);
    }

    // Step 3: Build fake FP16 weight tensor for matmul
    // We need to replace src0->data temporarily for the FP16 matmul call.
    // In production, we'd call ggml_cann_mat_mul_fp directly.
    // For now, the simplest approach is to modify src0 in-place:
    void * orig_data = src0->data;
    ggml_type orig_type = src0->type;
    src0->data = f16_weights;
    src0->type = GGML_TYPE_F16;

    // Call existing FP16 matmul (from aclnn_ops.cpp)
    // ggml_cann_mat_mul_fp(ctx, dst);

    // Restore
    src0->data = orig_data;
    src0->type = orig_type;

    // NOTE: The ggml_cann_mat_mul_fp function expects src1 as activations
    // and produces dst. We need to ensure src1 is FP16 on device too.
    // See aclnn_ops.cpp for ggml_cann_mat_mul_fp implementation.
}
