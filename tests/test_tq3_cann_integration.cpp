// TQ3_4S CANN backend integration test
// Verifies that ggml_cann_mul_mat correctly dispatches TQ3_4S weights
// through CPU dequant → ACL Cast → ACL BatchMatMul pipeline.
//
// Build:
//   source /home/developer/Ascend/cann-8.5.0/set_env.sh
//   export LD_LIBRARY_PATH=/home/developer/Ascend/cann-8.5.0/lib64:$LD_LIBRARY_PATH
//   export LIBRARY_PATH=/home/developer/Ascend/cann-8.5.0/lib64:$LIBRARY_PATH
//   g++ -std=c++17 -O2 -I /mnt/workspace/llama.cpp-tq3/ggml/include \
//       -I /mnt/workspace/llama.cpp-tq3/ggml/src \
//       -I /mnt/workspace/llama.cpp-tq3/ggml/src/ggml-cann \
//       -I /home/developer/Ascend/cann-8.5.0/include \
//       -I /home/developer/Ascend/cann-8.5.0/include/aclnn \
//       -I /home/developer/Ascend/cann-8.5.0/acllib/include \
//       test_tq3_cann_integration.cpp \
//       -L /mnt/workspace/llama.cpp-tq3/build-cann/ggml/src/ggml-cann \
//       -L /mnt/workspace/llama.cpp-tq3/build-cann/ggml/src \
//       -lggml-cann -lggml -lggml-base -lggml-cpu \
//       -lascendcl -lnnopbase -lopapi -lacl_op_compiler \
//       -lpthread -lrt -ldl -lm \
//       -o test_tq3_cann_integration

#include "ggml-cann.h"
#include "ggml-alloc.h"

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <numeric>
#include <random>

// TQ3_4S block layout (same as ggml-common.h)
struct block_tq3_4s {
    uint8_t d[4];    // 4 × E3M5 per-8 scales
    uint8_t qs[12];  // 32 × 3-bit indices packed into 12 bytes
};
static_assert(sizeof(block_tq3_4s) == 16, "block_tq3_4s must be 16 bytes");

// ─── helpers (duplicated from tq3_cann.cpp to avoid linker dependency) ────
static float golden_sign(int i) {
    return ((((unsigned)i * 0x9E3779B9u) >> 31) & 1) ? -1.0f : 1.0f;
}
static const float centroids[8] = {
    -2.1519f, -1.3439f, -0.7560f, -0.2451f,
     0.2451f,  0.7560f,  1.3439f,  2.1519f
};
static float e3m5_decode(uint8_t v) {
    int e = (v >> 5) & 7, m = v & 0x1F;
    return ldexpf(1.0f + m / 32.0f, e - 9);
}
static void unpack_3bit(const uint8_t qs[12], uint8_t idx[32]) {
    for (int i = 0; i < 32; i++) {
        int bp = (i * 3) / 8, bo = (i * 3) % 8;
        uint32_t w = qs[bp] | ((uint32_t)(bp + 1 < 12 ? qs[bp+1] : 0) << 8);
        idx[i] = (w >> bo) & 7;
    }
}
static void wht_32(float v[32]) {
    for (int step = 1; step < 32; step <<= 1)
        for (int i = 0; i < 32; i++)
            if (int j = i ^ step; j > i) {
                float a = v[i], b = v[j];
                v[i] = a + b; v[j] = a - b;
            }
}
static void dequant_one_block(const uint8_t blk[16], float * out) {
    float sc[4];
    for (int g = 0; g < 4; g++) sc[g] = e3m5_decode(blk[g]);
    uint8_t idx[32];
    unpack_3bit(blk + 4, idx);
    float buf[32];
    for (int i = 0; i < 32; i++) buf[i] = centroids[idx[i]] * sc[i / 8];
    wht_32(buf);
    const float rcp = 0.1767766952966369f;
    for (int i = 0; i < 32; i++) out[i] = buf[i] * golden_sign(i) * rcp;
}

// ─── manually quantize float data to TQ3_4S ────────────────────────────
// This is a simplified quantizer: we use the CPU dequant path in reverse.
// In practice, model quantizers use a more careful optimization.
static void quantize_tq3_4s_ref(const float * src, int64_t n, uint8_t * dst) {
    int64_t nb = n / 32;
    for (int64_t b = 0; b < nb; b++) {
        const float * block_src = src + b * 32;
        block_tq3_4s * blk = (block_tq3_4s *)(dst + b * sizeof(block_tq3_4s));

        // Find best centroid index for each element (pre-WHT)
        // We need to INVERT the dequant process, which requires solving:
        //   out = WHT(centroids[idx] * scale) * golden_sign / sqrt(32)
        // This is complex, so we use a simplified approach:
        // set all indices to 0 and scales to a fixed value
        for (int g = 0; g < 4; g++) {
            blk->d[g] = (5 << 5) | 19; // scale ≈ 0.0996
        }
        memset(blk->qs, 0, 12);
    }
}

// ─── CPU reference matmul ──────────────────────────────────────────────
static void cpu_matmul_f32(const float * A, const float * B,
                            float * C, int M, int N, int K) {
    // C[M×N] = A^T[M×K] * B[K×N]
    // But wait - in ggml, mul_mat treats A as weights and transposes internally
    // Actual: C = A^T * B  where A is [K, M], B is [K, N]
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[n * K + k];
            }
            C[m * N + n] = sum;
        }
    }
}

// ─── main test ─────────────────────────────────────────────────────────
int main() {
    printf("TQ3_4S CANN Integration Test\n");
    printf("============================\n\n");

    // Check CANN devices
    int n_dev = ggml_backend_cann_get_device_count();
    printf("CANN devices found: %d\n", n_dev);
    if (n_dev < 1) {
        printf("FAIL: No CANN devices available\n");
        return 1;
    }

    // Small test dimensions: K=32, M=4, N=2
    // 1 TQ3 block per row × 4 rows = 4 blocks
    const int64_t K = 32, M = 4, N = 2;
    const int64_t n_blocks = M * (K / 32); // = 4

    printf("Test dimensions: K=%ld, M=%ld, N=%ld\n", K, M, N);
    printf("TQ3 blocks: %ld\n\n", n_blocks);

    // ─── 1. Initialize CANN backend ──────────────────────────────────
    ggml_backend_t cann_backend = ggml_backend_cann_init(0);
    if (!cann_backend) {
        printf("FAIL: ggml_backend_cann_init failed\n");
        return 1;
    }
    printf("CANN backend initialized\n");

    // ─── 2. Create ggml tensors ─────────────────────────────────────
    ggml_init_params params = {
        /*.mem_size  =*/ ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
        /*.mem_buffer=*/ nullptr,
        /*.no_alloc  =*/ true,
    };
    ggml_context * ctx = ggml_init(params);

    // Weight: TQ3_4S quantized, shape [K, M]
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_TQ3_4S, K, M);
    // Input: F32, shape [K, N]
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    // Output: mul_mat result
    ggml_tensor * out = ggml_mul_mat(ctx, w, x);

    // ─── 3. Allocate device memory ─────────────────────────────────
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cann_backend);
    if (!buf) {
        printf("FAIL: buffer allocation failed\n");
        ggml_free(ctx);
        ggml_backend_free(cann_backend);
        return 1;
    }
    printf("Device memory allocated\n");

    // ─── 4. Fill TQ3_4S weight data ─────────────────────────────────
    // We'll use the known test block from dequant test
    std::vector<uint8_t> w_data(n_blocks * sizeof(block_tq3_4s));

    // Initialize weight blocks manually
    float known_scale = 0.099609f;
    for (int b = 0; b < n_blocks; b++) {
        block_tq3_4s * blk = (block_tq3_4s *)(w_data.data() + b * sizeof(block_tq3_4s));
        for (int g = 0; g < 4; g++) blk->d[g] = (5 << 5) | 19; // scale ≈ 0.0996
        memset(blk->qs, 0, 12);
        // Fill with alternating centroid indices
        for (int i = 0; i < 32; i++) {
            int idx = i & 7;
            int bp = (i * 3) / 8, bo = (i * 3) % 8;
            blk->qs[bp] |= idx << bo;
            if (bo > 5) blk->qs[bp + 1] |= idx >> (8 - bo);
        }
    }

    ggml_backend_tensor_set(w, w_data.data(), 0, w_data.size());
    printf("TQ3 weight data uploaded\n");

    // ─── 5. Fill input data ─────────────────────────────────────────
    std::vector<float> x_data(K * N);
    // Simple input: first column = all 1s, second column = all 0s
    for (int i = 0; i < K; i++) {
        x_data[i] = 1.0f;          // first token
        x_data[K + i] = 0.0f;      // second token
    }
    ggml_backend_tensor_set(x, x_data.data(), 0, x_data.size() * sizeof(float));
    printf("Input data uploaded\n");

    // ─── 6. Compute reference on CPU ────────────────────────────────
    // Dequant weight to FP32
    std::vector<float> w_f32(M * K);
    for (int b = 0; b < n_blocks; b++) {
        dequant_one_block(w_data.data() + b * sizeof(block_tq3_4s),
                          w_f32.data() + b * 32);
    }

    // CPU matmul: C[M×N] = W_dequant^T * X
    // W_dequant layout is [M, K] (row-major)
    // X layout is [K, N] (col-major = N rows of K)
    // Result: C[m][n] = sum_k W_dequant[m][k] * X[k][n]
    std::vector<float> ref_out(M * N);
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += w_f32[m * K + k] * x_data[n * K + k];
            }
            ref_out[m * N + n] = sum;
        }
    }
    printf("CPU reference computed\n");

    // ─── 7. Run on CANN ─────────────────────────────────────────────
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_status status = ggml_backend_graph_compute(cann_backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        printf("FAIL: graph compute returned %d\n", (int)status);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(cann_backend);
        return 1;
    }

    // Synchronize
    ggml_backend_synchronize(cann_backend);

    printf("CANN graph compute succeeded\n");

    // ─── 8. Read output ─────────────────────────────────────────────
    std::vector<float> cannn_out(M * N);
    ggml_backend_tensor_get(out, cannn_out.data(), 0, M * N * sizeof(float));

    // ─── 9. Compare ─────────────────────────────────────────────────
    printf("\nResults:\n");
    int n_err = 0;
    float max_err = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(cannn_out[i] - ref_out[i]);
        if (err > max_err) max_err = err;
        if (err > 1.0f) n_err++;
        printf("  [%d] CANN=%8.4f  REF=%8.4f  err=%8.6f%s\n",
               i, cannn_out[i], ref_out[i], err,
               (err > 1.0f) ? " ***" : "");
    }

    printf("\nMax error: %f\n", max_err);

    bool pass = true;
    if (n_err > 0) {
        printf("FAIL: %d elements with error > 1.0\n", n_err);
        pass = false;
    }
    for (int i = 0; i < M * N; i++) {
        if (std::isnan(cannn_out[i]) || std::isinf(cannn_out[i])) {
            printf("FAIL: output contains NaN or Inf at [%d]\n", i);
            pass = false;
            break;
        }
    }

    if (pass) {
        printf("\nPASSED: TQ3_4S CANN integration test OK\n");
    } else {
        printf("\nFAILED\n");
    }

    // ─── 10. Cleanup ────────────────────────────────────────────────
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(cann_backend);

    return pass ? 0 : 1;
}
