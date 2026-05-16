#!/bin/bash
# Apply the TQ3 CANN backend patch to llama.cpp-tq3
# Usage: ./apply.sh /path/to/llama.cpp-tq3

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 /path/to/llama.cpp-tq3"
    exit 1
fi

TARGET="$1"
PATCH_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCH="$PATCH_DIR/0001-add-tq3-cann-backend.patch"

if [ ! -f "$PATCH" ]; then
    echo "Error: patch not found at $PATCH"
    exit 1
fi

if [ ! -d "$TARGET" ]; then
    echo "Error: target directory $TARGET not found"
    exit 1
fi

cd "$TARGET"

# Check if patch is already applied
if git log --oneline -1 | grep -q "tq3-cann" 2>/dev/null; then
    echo "Patch appears to already be applied."
    exit 0
fi

git am "$PATCH"
echo "Patch applied successfully."
echo ""
echo "Now build with CANN backend:"
echo "  source /path/to/Ascend/cann-8.5.0/set_env.sh"
echo "  mkdir build && cd build"
echo "  cmake .. -DGGML_CANN=ON -DCANN_INSTALL_DIR=\${ASCEND_TOOLKIT_HOME} -DCMAKE_BUILD_TYPE=Release"
echo "  cmake --build . -j\$(nproc)"
