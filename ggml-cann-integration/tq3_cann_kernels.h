#pragma once

#ifndef TQ3_CANN_KERNELS_H
#define TQ3_CANN_KERNELS_H

#include <cstdint>
#include <cmath>
#include <cstring>

// ─── TQ3_4S Dequant Helpers ──────────────────────────────────────────

static float tq3_golden_sign(int i) {
    return ((((unsigned)i * 0x9E3779B9u) >> 31) & 1) ? -1.0f : 1.0f;
}

static const float tq3_centroids[8] = {
    -2.1519f, -1.3439f, -0.7560f, -0.2451f,
     0.2451f,  0.7560f,  1.3439f,  2.1519f
};

static float e3m5_decode(uint8_t v) {
    int e = (v >> 5) & 7, m = v & 0x1F;
    return ldexpf(1.0f + m / 32.0f, e - 9);
}

static float tq3_4s_decode_e3m5(uint8_t v) {
    return e3m5_decode(v);
}

static void unpack_3bit(const uint8_t qs[12], uint8_t idx[32]) {
    for (int i = 0; i < 32; i++) {
        int bp = (i * 3) / 8, bo = (i * 3) % 8;
        uint32_t w = qs[bp] | ((uint32_t)(bp + 1 < 12 ? qs[bp + 1] : 0) << 8);
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

static void tq3_4s_dequant_block(const uint8_t blk[16], float* out, int stride = 1) {
    float sc[4];
    for (int g = 0; g < 4; g++) sc[g] = e3m5_decode(blk[g]);

    uint8_t idx[32];
    unpack_3bit(blk + 4, idx);

    float buf[32];
    for (int i = 0; i < 32; i++) buf[i] = tq3_centroids[idx[i]] * sc[i / 8];

    wht_32(buf);

    const float rcp = 0.1767766952966369f;
    for (int i = 0; i < 32; i++) out[i * stride] = buf[i] * tq3_golden_sign(i) * rcp;
}

static float fp16_to_f32(uint16_t h) {
    uint32_t s = (h >> 15) & 1;
    uint32_t e = (h >> 10) & 0x1F;
    uint32_t m = h & 0x3FF;
    if (e == 0) {
        int shift = 9;
        while (!(m & 0x400)) { m <<= 1; shift++; }
        m &= 0x3FF;
        e = 1 - shift + 15;
    }
    uint32_t f32 = (s << 31) | ((e + 112) << 23) | (m << 13);
    float result;
    memcpy(&result, &f32, 4);
    return result;
}

static void tq3_1s_dequant_block(const uint8_t blk[16], float* out, int stride = 1) {
    uint16_t d0_raw, d1_raw;
    memcpy(&d0_raw, blk, 2);
    memcpy(&d1_raw, blk + 2, 2);

    float s0 = fp16_to_f32(d0_raw);
    float s1 = fp16_to_f32(d1_raw);

    uint8_t idx[32];
    unpack_3bit(blk + 4, idx);

    float buf[32];
    for (int i = 0; i < 32; i++) {
        float sc = (i < 16) ? s0 : s1;
        buf[i] = tq3_centroids[idx[i]] * sc;
    }

    wht_32(buf);

    const float rcp = 0.1767766952966369f;
    for (int i = 0; i < 32; i++) out[i * stride] = buf[i] * tq3_golden_sign(i) * rcp;
}

#endif // TQ3_CANN_KERNELS_H
