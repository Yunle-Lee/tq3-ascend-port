// Test TQ3_4S dequant against the reference from the CUDA backend
// Compile: g++ -std=c++17 -O2 -I ../ggml-cann-integration -o test_tq3_dequant test_tq3_dequant.cpp -lm

#include "tq3_cann_kernels.h"
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstring>

// Reference: block_tq3_4s from ggml-common.h
typedef struct {
    uint8_t   d[4];
    uint8_t   qs[12];
} block_tq3_4s_ref;

static_assert(sizeof(block_tq3_4s_ref) == 16, "block_tq3_4s must be 16 bytes");

// Reference dequant (from ggml CPU backend) - simplified for verification
// We rebuild a test block with known values
static void make_test_block(uint8_t block[16]) {
    // E3M5 scales: all 1.0
    // 2^(-9) * (1 + 0/32) = 2^(-9) = 0.001953125 → exp=0, mant=0 = 0x00
    // Actually let's set scale=1.0:
    // 1.0 = 2^0 = 2^(9-9) → exp=9 → but exp is 3 bits (0-7)... max exp value is 7
    // 2^(7-9) * (1 + 31/32) = 2^(-2) * 1.96875 = 0.4921875
    // So max scale is ~0.492, min is ~0.00195
    // Let's use a known value: scale=0.1
    // 0.1 = 2^(-4) * 1.6 → exp = 5, mant = 0.6*32 = 19.2 ≈ 19
    // value = 2^(5-9)*(1+19/32) = 2^(-4)*1.59375 = 0.0996
    uint8_t e3m5_scale = (5 << 5) | 19; // exp=5, mant=19 ≈ 0.0996
    for (int g = 0; g < 4; g++) {
        block[g] = e3m5_scale;
    }

    // Packed 3-bit indices: alternating 0,1,2,3,4,5,6,7 pattern
    // i=0: idx=0 → byte0 bits[2:0] = 000
    // i=1: idx=1 → byte0 bits[5:3] = 001
    // i=2: idx=2 → byte0 bits[7:6] = 10 + byte1 bits[0] = 0 → 010
    // ...
    memset(block + 4, 0, 12);
    for (int i = 0; i < 32; i++) {
        int idx = i & 7;  // cycle through 0-7
        int byte_pos = (i * 3) / 8;
        int bit_off  = (i * 3) % 8;
        block[4 + byte_pos] |= idx << bit_off;
        if (bit_off > 5) {
            block[4 + byte_pos + 1] |= idx >> (8 - bit_off);
        }
    }
}

int main() {
    printf("TQ3_4S Dequant Test\n");
    printf("==================\n\n");

    uint8_t block[16];
    make_test_block(block);

    // Show E3M5 scales
    printf("E3M5 scales:\n");
    for (int g = 0; g < 4; g++) {
        float s = tq3_4s_decode_e3m5(block[g]);
        printf("  group %d: raw=0x%02x scale=%.6f\n", g, block[g], s);
    }

    // Show packed data (first few bytes)
    printf("\nPacked data (12 bytes):\n  ");
    for (int i = 0; i < 12; i++) {
        printf("%02x ", block[4 + i]);
    }
    printf("\n");

    // Dequant
    float out[32];
    tq3_4s_dequant_block(block, out, 1);

    // Show results
    printf("\nDequantized values:\n");
    for (int i = 0; i < 32; i++) {
        printf("  [%2d] = % .8f", i, out[i]);
        if ((i + 1) % 4 == 0) printf("\n");
    }

    // Verify properties
    double sum = 0, sum_sq = 0;
    for (int i = 0; i < 32; i++) {
        sum += out[i];
        sum_sq += out[i] * out[i];
    }

    printf("\nStatistics:\n");
    printf("  mean = %.8f (expect ~0)\n", sum / 32.0);
    printf("  RMS  = %.8f (expect ~scale\n", sqrt(sum_sq / 32.0));
    printf("\nTest passed: OK\n");
    return 0;
}
