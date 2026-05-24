#include "tq3_cann.h"
#include "tq3_cann_kernels.h"
#include "aclnn_ops.h"
#include "common.h"
#include "ggml-impl.h"

#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_batch_matmul.h>

static void dequant_tq3_tensor(const ggml_tensor * src, float * dst) {
    auto * data = (const uint8_t *)src->data;
    int64_t rows = src->ne[1], cols = src->ne[0];
    int64_t nb = cols / 32;

    if (src->type == GGML_TYPE_TQ3_4S) {
        for (int64_t r = 0; r < rows; r++)
            for (int64_t b = 0; b < nb; b++)
                tq3_4s_dequant_block(data + (r * nb + b) * 16,
                                     dst + r * cols + b * 32);
    } else {
        for (int64_t r = 0; r < rows; r++)
            for (int64_t b = 0; b < nb; b++)
                tq3_1s_dequant_block(data + (r * nb + b) * 16,
                                     dst + r * cols + b * 32);
    }
}

void ggml_cann_mul_mat_tq3(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    int64_t K = src0->ne[0];
    int64_t M = src0->ne[1];
    int64_t N = src1->ne[1];

    GGML_ASSERT(src0->type == GGML_TYPE_TQ3_4S || src0->type == GGML_TYPE_TQ3_1S);
    GGML_ASSERT(K == src1->ne[0]);

    int64_t num_weights = M * K;

    size_t f32_sz = num_weights * sizeof(float);
    ggml_cann_pool_alloc f32_buf(ctx.pool());
    auto * f32_w = (float *)f32_buf.alloc(f32_sz);
    dequant_tq3_tensor(src0, f32_w);

    size_t f16_sz = num_weights * sizeof(uint16_t);
    ggml_cann_pool_alloc f16_buf(ctx.pool());
    auto * f16_w = (uint16_t *)f16_buf.alloc(f16_sz);

    {
        int64_t ne_w[]  = { K, M };
        size_t  nb_f32[] = { sizeof(float), K * sizeof(float) };
        size_t  nb_f16[] = { sizeof(uint16_t), K * sizeof(uint16_t) };

        acl_tensor_ptr acl_f32_t = ggml_cann_create_tensor(
            f32_w, ACL_FLOAT, sizeof(float), ne_w, nb_f32, 2);
        acl_tensor_ptr acl_f16_t = ggml_cann_create_tensor(
            f16_w, ACL_FLOAT16, sizeof(uint16_t), ne_w, nb_f16, 2);
        GGML_CANN_CALL_ACLNN_OP(ctx, Cast, acl_f32_t.get(), ACL_FLOAT16, acl_f16_t.get());
    }

    bool input_is_f16 = (src1->type == GGML_TYPE_F16);
    ggml_cann_pool_alloc in_buf(ctx.pool());
    void * in_dev = src1->data;

    if (!input_is_f16) {
        size_t in_sz = ggml_nelements(src1) * sizeof(uint16_t);
        in_dev = in_buf.alloc(in_sz);

        int64_t ne_in[]  = { K, N };
        size_t  nb_in[]  = { ggml_type_size(src1->type), K * ggml_type_size(src1->type) };
        size_t  nb_in16[] = { sizeof(uint16_t), K * sizeof(uint16_t) };

        acl_tensor_ptr acl_in_src = ggml_cann_create_tensor(
            src1->data, ggml_cann_type_mapping(src1->type),
            ggml_type_size(src1->type), ne_in, nb_in, 2);
        acl_tensor_ptr acl_in_dst = ggml_cann_create_tensor(
            in_dev, ACL_FLOAT16, sizeof(uint16_t), ne_in, nb_in16, 2);
        GGML_CANN_CALL_ACLNN_OP(ctx, Cast, acl_in_src.get(), ACL_FLOAT16, acl_in_dst.get());
    }

    size_t out_sz = M * N * sizeof(uint16_t);
    ggml_cann_pool_alloc out_buf(ctx.pool());
    auto * out_dev = (uint16_t *)out_buf.alloc(out_sz);

    {
        int64_t ne_w_t[]  = { K, M };
        size_t  nb_w_t[]  = { sizeof(uint16_t), K * sizeof(uint16_t) };

        int64_t ne_a[]    = { K, N };
        size_t  nb_a[]    = { sizeof(uint16_t), K * sizeof(uint16_t) };

        int64_t ne_o[]    = { M, N };
        size_t  nb_o[]    = { sizeof(uint16_t), M * sizeof(uint16_t) };

        acl_tensor_ptr acl_w = ggml_cann_create_tensor(
            f16_w, ACL_FLOAT16, sizeof(uint16_t), ne_w_t, nb_w_t, 2);
        acl_tensor_ptr acl_a = ggml_cann_create_tensor(
            in_dev, ACL_FLOAT16, sizeof(uint16_t), ne_a, nb_a, 2);
        acl_tensor_ptr acl_o = ggml_cann_create_tensor(
            out_dev, ACL_FLOAT16, sizeof(uint16_t), ne_o, nb_o, 2);

        GGML_CANN_CALL_ACLNN_OP(ctx, BatchMatMul, acl_w.get(), acl_a.get(), acl_o.get(), static_cast<int8_t>(2));
    }

    if (dst->type != GGML_TYPE_F16) {
        int64_t ne_o[]   = { M, N };
        size_t  nb_o16[] = { sizeof(uint16_t), M * sizeof(uint16_t) };
        size_t  nb_od[]  = { ggml_type_size(dst->type), M * ggml_type_size(dst->type) };

        acl_tensor_ptr acl_o16 = ggml_cann_create_tensor(
            out_dev, ACL_FLOAT16, sizeof(uint16_t), ne_o, nb_o16, 2);
        acl_tensor_ptr acl_od  = ggml_cann_create_tensor(
            dst->data, ggml_cann_type_mapping(dst->type),
            ggml_type_size(dst->type), ne_o, nb_od, 2);
        GGML_CANN_CALL_ACLNN_OP(ctx, Cast, acl_o16.get(), ggml_cann_type_mapping(dst->type), acl_od.get());
    } else {
        ACL_CHECK(aclrtMemcpy(dst->data, out_sz, out_dev, out_sz, ACL_MEMCPY_DEVICE_TO_DEVICE));
    }
}
