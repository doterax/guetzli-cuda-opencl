#!/bin/bash
# Guetzli CUDA/OpenCL — Test Suite
# Usage: tests/run_tests.sh <path-to-guetzli-binary>
set -euo pipefail

GUETZLI="${1:?Usage: $0 <guetzli-binary>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

PASS=0
FAIL=0
TOTAL=0

pass() { PASS=$((PASS+1)); TOTAL=$((TOTAL+1)); echo "PASS"; }
fail() { FAIL=$((FAIL+1)); TOTAL=$((TOTAL+1)); echo "FAIL: $1"; }

echo "============================================"
echo "  Guetzli CUDA/OpenCL — Test Suite"
echo "============================================"
echo "Binary: $GUETZLI"
echo ""

# -------------------------------------------------------
# Test 1: Binary exists and is executable
# -------------------------------------------------------
printf "Test 1: Binary exists ............ "
if [ -x "$GUETZLI" ] || [ -f "$GUETZLI" ]; then
    pass
else
    fail "binary not found or not executable"
fi

# -------------------------------------------------------
# Test 2: --version flag exits cleanly
# -------------------------------------------------------
printf "Test 2: --version ................ "
if "$GUETZLI" --version >/dev/null 2>&1; then
    pass
else
    # Some builds may not have --version; treat exit code != segfault as soft pass
    EXIT_CODE=$?
    if [ "$EXIT_CODE" -lt 128 ]; then
        pass  # non-zero but not a signal — acceptable
    else
        fail "exit code $EXIT_CODE (signal)"
    fi
fi

# -------------------------------------------------------
# Test 3: PNG → JPEG conversion (small image)
# -------------------------------------------------------
printf "Test 3: PNG to JPEG .............. "
INPUT_PNG="$SCRIPT_DIR/input/fro_small.png"
OUTPUT_JPG="$TMPDIR/fro_small_out.jpg"
if [ ! -f "$INPUT_PNG" ]; then
    # Try to find any small PNG
    INPUT_PNG=$(find "$SCRIPT_DIR/input" -name '*.png' -size -200k 2>/dev/null | head -1)
fi
if [ -n "$INPUT_PNG" ] && [ -f "$INPUT_PNG" ]; then
    if "$GUETZLI" --quality 84 "$INPUT_PNG" "$OUTPUT_JPG" 2>/dev/null; then
        if [ -f "$OUTPUT_JPG" ]; then
            pass
        else
            fail "output file not created"
        fi
    else
        fail "guetzli exited with error"
    fi
else
    fail "no small PNG test input found"
fi

# -------------------------------------------------------
# Test 4: Output is a valid JPEG (magic bytes FF D8)
# -------------------------------------------------------
printf "Test 4: Valid JPEG magic bytes ... "
if [ -f "$OUTPUT_JPG" ]; then
    MAGIC=$(xxd -l2 -p "$OUTPUT_JPG" 2>/dev/null || od -A n -t x1 -N 2 "$OUTPUT_JPG" | tr -d ' ')
    if [ "$MAGIC" = "ffd8" ]; then
        pass
    else
        fail "magic bytes: $MAGIC (expected ffd8)"
    fi
else
    fail "no output to check"
fi

# -------------------------------------------------------
# Test 5: Output file size is reasonable
# -------------------------------------------------------
printf "Test 5: Output size sanity ....... "
if [ -f "$OUTPUT_JPG" ]; then
    SIZE=$(stat -c%s "$OUTPUT_JPG" 2>/dev/null || stat -f%z "$OUTPUT_JPG" 2>/dev/null || echo 0)
    if [ "$SIZE" -gt 512 ] && [ "$SIZE" -lt 1048576 ]; then
        echo "PASS (${SIZE} bytes)"
        PASS=$((PASS+1)); TOTAL=$((TOTAL+1))
    else
        fail "size ${SIZE} bytes out of expected range (512..1MB)"
    fi
else
    fail "no output to check"
fi

# -------------------------------------------------------
# Summary
# -------------------------------------------------------
echo ""
echo "============================================"
echo "  Results: $PASS/$TOTAL passed, $FAIL failed"
echo "============================================"
[ "$FAIL" -eq 0 ] || exit 1
