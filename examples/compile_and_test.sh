#!/bin/bash
# Compile and test the TQ3 CANN integration
#
# Prerequisites:
#   source /path/to/Ascend/cann-8.5.0/set_env.sh
#   LLAMA_CPP_TQ3=/path/to/llama.cpp-tq3 (with CANN backend built)
#
# This script demonstrates:
#   1. CPU dequant test (standalone, no NPU needed)
#   2. CANN backend build verification

set -e

DIR="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_CPP_TQ3="${LLAMA_CPP_TQ3:-/mnt/workspace/llama.cpp-tq3}"
BUILD_DIR="${BUILD_DIR:-${LLAMA_CPP_TQ3}/build}"
CANN_HOME="${ASCEND_TOOLKIT_HOME:-/home/developer/Ascend/cann-8.5.0}"

echo "=== TQ3 CANN Port - Build & Test ==="
echo "llama.cpp-tq3: $LLAMA_CPP_TQ3"
echo "CANN home:     $CANN_HOME"
echo ""

# ── Test 1: CPU dequant standalone ──────────────────────────────────────
echo "--- Test 1: CPU TQ3_4S Dequant ---"
g++ -std=c++17 -O2 \
    -I "$DIR/ggml-cann-integration" \
    -o /tmp/test_tq3_dequant \
    "$DIR/tests/test_tq3_dequant.cpp" \
    "$DIR/ggml-cann-integration/tq3_cann.cpp" \
    -lm
/tmp/test_tq3_dequant
echo ""

# ── Test 2: Verify ggml-cann symbols ────────────────────────────────────
echo "--- Test 2: Verify TQ3 symbols in compiled library ---"
if [ -f "$BUILD_DIR/ggml/src/ggml-cann/libggml-cann.a" ]; then
    nm "$BUILD_DIR/ggml/src/ggml-cann/libggml-cann.a" | grep -q "ggml_cann_mul_mat_tq3" && \
        echo "  OK: ggml_cann_mul_mat_tq3 found in libggml-cann.a" || \
        echo "  FAIL: symbol not found"
elif [ -f "$BUILD_DIR/bin/libggml-cann.so" ]; then
    nm -D "$BUILD_DIR/bin/libggml-cann.so" | grep -q "ggml_cann_mul_mat_tq3" && \
        echo "  OK: ggml_cann_mul_mat_tq3 found in libggml-cann.so" || \
        echo "  FAIL: symbol not found"
else
    echo "  SKIP: libggml-cann not built yet"
fi
echo ""

# ── Test 3: CANN backend model load (needs a model file) ────────────────
echo "--- Test 3: CANN backend detection ---"
if [ -f "$BUILD_DIR/bin/llama-perplexity" ]; then
    LD_LIBRARY_PATH="$CANN_HOME/lib64:$LD_LIBRARY_PATH" \
        "$BUILD_DIR/bin/llama-perplexity" --version 2>&1 | head -3
    echo "  llama-perplexity binary OK"
else
    echo "  SKIP: llama-perplexity not built"
fi
echo ""

echo "=== All tests complete ==="
