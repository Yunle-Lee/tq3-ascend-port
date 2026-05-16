#include "kernel_operator.h"

extern "C" __global__ __aicore__ void tq3_dequant(GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
}