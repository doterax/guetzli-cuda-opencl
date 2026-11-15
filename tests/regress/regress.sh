#!/bin/bash
# Minimal output-regression harness for guetzli backends.
#
# Usage:
#   tests/regress/regress.sh <guetzli-binary> [mode ...]
#     modes: cpu opencl cuda   (default: cpu)
#            checkcl checkcuda: GPU and CPU both run and are compared; the
#            CPU result is used, so output must match the cpu baseline.
#            fallback-opencl fallback-cuda: GPU init failure is injected
#            (GUETZLI_TEST_GPU_FAIL=1); output must match the cpu baseline.
#   UPDATE=1 tests/regress/regress.sh <binary> cpu opencl   # (re)write baselines
#
# Inputs are tiny (mostly >100 px so GPU paths are taken, but cheap to run).
# For each mode the md5 of every output is compared against
# tests/regress/expected/<mode>.md5. For check* modes, the number of
# "CHK" mismatch lines reported by the binary is printed as well.
# Exit code: number of failures (missing baseline counts as "NEW", not a failure).
set -u

BIN="${1:?Usage: $0 <guetzli-binary> [mode ...]}"
shift
MODES=("$@")
[ ${#MODES[@]} -eq 0 ] && MODES=(cpu)

DIR="$(cd "$(dirname "$0")" && pwd)"
IN="$DIR/input"
EXP="$DIR/expected"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
mkdir -p "$EXP"

QUALITY="${QUALITY:-84}"
FAIL=0

flag_for() {
    case "$1" in
        cpu) echo "--c" ;;
        opencl) echo "--opencl" ;;
        cuda) echo "--cuda" ;;
        checkcl) echo "--checkcl" ;;
        checkcuda) echo "--checkcuda" ;;
        fallback-opencl) echo "--opencl" ;;
        fallback-cuda) echo "--cuda" ;;
        *) echo "unknown mode $1" >&2; exit 2 ;;
    esac
}

md5of() { md5sum "$1" | cut -d' ' -f1; }

for mode in "${MODES[@]}"; do
    flag="$(flag_for "$mode")"
    expfile="$EXP/$mode.md5"
    inject=0
    case "$mode" in
        fallback-*) expfile="$EXP/cpu.md5"; inject=1 ;;
        check*) expfile="$EXP/cpu.md5" ;;
    esac
    newfile="$OUT/$mode.md5"
    : > "$newfile"
    echo "== mode: $mode ($flag)"
    for f in "$IN"/*; do
        name="$(basename "$f")"
        extra=""
        case "$name" in rgba_*) extra="--blend-on-white" ;; esac
        out="$OUT/${mode}_${name%.*}.jpg"
        log="$OUT/${mode}_${name%.*}.log"
        printf "  %-28s " "$name"
        GUETZLI_TEST_GPU_FAIL=$inject "$BIN" --verbose --quality "$QUALITY" $flag $extra "$f" "$out" >"$log" 2>&1
        rc=$?
        if [ $rc -ne 0 ] || [ ! -s "$out" ]; then
            echo "FAIL (exit $rc)"; tail -5 "$log" | sed 's/^/      /'
            FAIL=$((FAIL+1)); continue
        fi
        sum="$(md5of "$out")"
        echo "$sum  $name" >> "$newfile"
        chk="$(grep -c 'CHK' "$log" || true)"
        want="$(grep "  $name\$" "$expfile" 2>/dev/null | cut -d' ' -f1)"
        if [ -z "$want" ]; then status="NEW"
        elif [ "$want" = "$sum" ]; then status="OK"
        else status="CHANGED"; FAIL=$((FAIL+1)); fi
        printf "%-8s %s" "$status" "$sum"
        [ "$chk" != "0" ] && printf "  (CHK mismatches: %s)" "$chk"
        echo
    done
    if [ "${UPDATE:-0}" = "1" ] && [ "$expfile" = "$EXP/$mode.md5" ]; then cp "$newfile" "$expfile"; echo "  baseline written: $expfile"; fi
done

# Cross-backend comparison (informational).
if [ ${#MODES[@]} -gt 1 ]; then
    echo "== cross-mode equality vs ${MODES[0]}"
    for mode in "${MODES[@]:1}"; do
        for f in "$IN"/*; do
            n="$(basename "$f")"; a="$OUT/${MODES[0]}_${n%.*}.jpg"; b="$OUT/${mode}_${n%.*}.jpg"
            [ -f "$a" ] && [ -f "$b" ] || continue
            if cmp -s "$a" "$b"; then r=same; else r=differs; fi
            printf "  %-10s %-28s %s\n" "$mode" "$n" "$r"
        done
    done
fi

echo "failures: $FAIL"
exit $FAIL
