# TQ3 Algorithm Reference

This document describes the TurboQuant 3-bit (TQ3) quantization format and the dequantization algorithm, as implemented in the CUDA kernels and replicated for the CANN backend.

## TQ3_4S Block Format (Type 46)

Block size: **32 elements packed into 16 bytes** (4.0 bpw)

```
byte 0-3:   4 × E3M5 per-8 scales (u8)
byte 4-15:  12 bytes packed 3-bit indices (32 indices × 3 bits = 96 bits = 12 bytes)

block_tq3_4s {
    uint8_t d[4];    // scales for groups of 8 elements
    uint8_t qs[12];  // packed 3-bit centroid indices
};
// sizeof(block_tq3_4s) == 16
```

## TQ3_1S Block Format (Type 44)

Block size: **32 elements packed into 16 bytes** (4.0 bpw)

```
byte 0-1:   d0 (FP16 half-block scale for first 16 elements)
byte 2-3:   d1 (FP16 half-block scale for last 16 elements)
byte 4-15:  12 bytes packed 3-bit indices (same as TQ3_4S)
```

## E3M5 Mini-Float (TQ3_4S Scales)

```
bits [7:5] = exponent (3-bit, bias 9)
bits [4:0] = mantissa (5-bit, explicit leading 1)

value = 2^(exp - 9) × (1 + mant / 32)

Range: 2^(-9) × 1.0 ≈ 0.00195  to  2^(7-9) × (1 + 31/32) ≈ 0.492
```

Example decoding:
```c
// val = (5 << 5) | 19  →  exp=5, mant=19
// value = 2^(5-9) × (1 + 19/32) = 2^(-4) × 1.59375 ≈ 0.0996
float e3m5_decode(uint8_t v) {
    int e = (v >> 5) & 7;
    int m = v & 0x1F;
    return ldexpf(1.0f + m / 32.0f, e - 9);
}
```

## Dequantization Pipeline

### Step 1: Unpack 3-bit Indices

32 indices packed into 12 bytes, 3 bits per index:

```
byte 0:  bits [2:0] = idx[0], bits[5:3] = idx[1], bits[7:6] = idx[2][1:0]
byte 1:  bit  [0]   = idx[2][2],  bits[3:1] = idx[3], bits[6:4] = idx[4], bit[7] = idx[5][0]
... etc.

for i = 0..31:
    byte_pos = (i * 3) / 8
    bit_off  = (i * 3) % 8
    word = qs[byte_pos] | (qs[byte_pos + 1] << 8)
    idx[i] = (word >> bit_off) & 7
```

### Step 2: Centroid LUT

Each 3-bit index maps to one of 8 centroid values (Lloyd-Max optimized for unit Gaussian):

```c
static const float centroids[8] = {
    -2.1519, -1.3439, -0.7560, -0.2451,
     0.2451,  0.7560,  1.3439,  2.1519
};
```

### Step 3: Per-8 Scaling (TQ3_4S)

```
for i = 0..31:
    group = i / 8           // 0, 1, 2, 3
    scale = e3m5_decode(d[group])
    buf[i] = centroids[idx[i]] × scale
```

### Step 4: Walsh-Hadamard Transform (32-point)

In-place butterfly network:

```
for step = 1, 2, 4, 8, 16:
    for i = 0..31:
        j = i XOR step
        if j > i:
            a = buf[i], b = buf[j]
            buf[i] = a + b
            buf[j] = a - b
```

This is equivalent to multiplying by the Hadamard matrix H_32 without the 1/sqrt(32) normalization.

### Step 5: Sign Projection + Normalize

Golden-ratio hash for deterministic signs:

```c
static float golden_sign(int i) {
    return ((((unsigned)i * 0x9E3779B9u) >> 31) & 1) ? -1.0f : 1.0f;
}
```

Final output:

```c
rcp_sqrt_32 = 0.1767766952966369f;  // 1/sqrt(32)
for i = 0..31:
    out[i] = buf[i] × golden_sign(i) × rcp_sqrt_32
```

## FP16 Decode (TQ3_1S)

TQ3_1S uses standard IEEE-754 FP16 for its two half-block scales:

```
s = sign (1 bit)
e = exponent (5 bits, bias 15)
m = mantissa (10 bits, implicit leading 1 for normalized values)

Normal:     value = (-1)^s × 2^(e-15) × (1 + m/1024)
Subnormal:  value = (-1)^s × 2^(-14) × m/1024
```

## CPU Reference Implementation

The complete CPU dequant is in `ggml-cann-integration/tq3_cann.cpp`. It includes both TQ3_4S and TQ3_1S paths. Key functions:

- `dequant_tq3_4s_block()` — dequant one TQ3_4S block to 32 FP32 values
- `dequant_tq3_1s_block()` — dequant one TQ3_1S block to 32 FP32 values
- `dequant_tq3_tensor()` — dispatch across all blocks in a weight tensor
- `ggml_cann_mul_mat_tq3()` — full pipeline: dequant → Cast → BatchMatMul
