#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BLOCK_SIZE = 16;
constexpr int32_t OUTPUT_SIZE = 32;
constexpr int32_t BUFFER_NUM = 2;

class KernelTq3Dequant {
public:
    __aicore__ inline KernelTq3Dequant() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling,
                                uint32_t block_num, uint32_t block_offset) {
        xGm.SetGlobalBuffer((__gm__ uint8_t*)x);
        yGm.SetGlobalBuffer((__gm__ half*)y);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, BLOCK_SIZE * sizeof(uint8_t));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, OUTPUT_SIZE * sizeof(half));
    }

    __aicore__ inline void Process() {
        constexpr uint8_t centroids[8] = {0};
        for (int32_t i = 0; i < block_num; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t idx) {
        LocalTensor<uint8_t> xLocal = inQueueX.AllocTensor<uint8_t>();
        DataCopy(xLocal, xGm[idx * BLOCK_SIZE], BLOCK_SIZE);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t idx) {
        LocalTensor<uint8_t> xLocal = inQueueX.DeQue<uint8_t>();
        LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();

        float scales[4];
        for (int g = 0; g < 4; g++) {
            uint8_t raw = xLocal.GetValue(g);
            int e = (raw >> 5) & 7;
            int m = raw & 0x1F;
            scales[g] = (1.0f + (float)m / 32.0f) * ((float)(1 << e) / 512.0f);
        }

        static const float centroid_vals[8] = {
            -2.1519f, -1.3439f, -0.7560f, -0.2451f,
             0.2451f,  0.7560f,  1.3439f,  2.1519f
        };

        float buf[32];
        for (int i = 0; i < 32; i++) {
            int bp = (i * 3) / 8;
            int bo = (i * 3) % 8;
            uint32_t w = xLocal.GetValue(4 + bp) | ((uint32_t)xLocal.GetValue(4 + bp + 1) << 8);
            uint8_t idx = (w >> bo) & 7;
            buf[i] = centroid_vals[idx] * scales[i / 8];
        }

        for (int step = 1; step < 32; step <<= 1) {
            for (int i = 0; i < 32; i++) {
                int j = i ^ step;
                if (j > i) {
                    float a = buf[i], b = buf[j];
                    buf[i] = a + b;
                    buf[j] = a - b;
                }
            }
        }

        static const float rcp = 0.1767766952966369f;
        for (int i = 0; i < 32; i++) {
            int sign_bit = ((unsigned)i * 0x9E3779B9u) >> 31;
            float final_val = buf[i] * (sign_bit ? -rcp : rcp);
            yLocal.SetValue(i, (half)final_val);
        }

        inQueueX.FreeTensor(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t idx) {
        LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        DataCopy(yGm[idx * OUTPUT_SIZE], yLocal, OUTPUT_SIZE);
        inQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> inQueueY;
    GlobalTensor<uint8_t> xGm;
    GlobalTensor<half> yGm;
};

extern "C" __global__ __aicore__ void tq3_dequant(GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    uint32_t total_blocks = tiling_data.size / BLOCK_SIZE;
    uint32_t block_per_core = total_blocks / tiling_data.block_dim;
    uint32_t block_offset = GetBlockIdx() * block_per_core;

    KernelTq3Dequant op;
    op.Init(workspace, workspace + tiling_data.size * 2, workspace, tiling,
            block_per_core, block_offset);
    op.Process();
}
