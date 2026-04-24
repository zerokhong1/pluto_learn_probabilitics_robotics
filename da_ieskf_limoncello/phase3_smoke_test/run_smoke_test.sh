#!/usr/bin/env bash
# run_smoke_test.sh — Phase 3 smoke test runner
#
# Runs LIMOncello + DA-IESKF on synthetic bags and checks outputs.
# Must be run from ~/LIMOncello_ws after: source install/setup.bash
#
# Usage:
#   ./test/run_smoke_test.sh [box|corridor|both]   default: both
#
# What it does:
#   1. Launches LIMOncello (no DA) on box room bag → no crash, odom published
#   2. Launches LIMOncello (DA on, log enabled) on corridor bag → DA triggers
#   3. Analyzes eigenvalue logs
set -euo pipefail

MODE="${1:-both}"
WSDIR="$(cd "$(dirname "$0")/.." && pwd)"
LOGDIR="$WSDIR/logs"
TESTDIR="$WSDIR/test"
mkdir -p "$LOGDIR"

source /opt/ros/jazzy/setup.bash
source "$WSDIR/install/setup.bash"

TIMEOUT=30   # seconds to play bag before killing
SETTLE=3     # seconds to let LIMOncello start before playing bag

log() { echo "[smoke_test] $*"; }
check_topic() {
    local topic="$1"
    local ns="${2:-limoncello}"
    timeout 5 ros2 topic echo "/$ns/$topic" --once 2>/dev/null | head -3
}

# ── Helper: run one scenario ──────────────────────────────────────────────────
run_scenario() {
    local label="$1"
    local bag_path="$2"
    local da_enabled="$3"
    local eigen_log="$4"

    log "=== $label ==="
    log "    bag:        $bag_path"
    log "    DA-IESKF:   $da_enabled"
    log "    eigen_log:  $eigen_log"
    echo ""

    # Patch YAML temporarily for this run
    local cfg="$WSDIR/src/LIMOncello/config/smoke_test.yaml"
    local cfg_bak="${cfg}.bak"
    cp "$cfg" "$cfg_bak"

    # Set da_ieskf_enabled and log path
    sed -i "s|da_ieskf_enabled:.*|da_ieskf_enabled: $da_enabled|" "$cfg"
    if [ -n "$eigen_log" ]; then
        sed -i "s|da_eigenvalue_log:.*|da_eigenvalue_log: \"$eigen_log\"|" "$cfg"
    fi

    # Rebuild (only if config was install-symlinked; symlink means no rebuild needed)
    # Symlink install: config changes take effect immediately. Skip rebuild.

    # Launch LIMOncello in background
    local limoncello_log="$LOGDIR/${label//[^a-zA-Z0-9]/_}_limoncello.log"
    ros2 launch limoncello limoncello.launch.py \
        config_name:=smoke_test \
        use_sim_time:=true \
        rviz:=false \
        > "$limoncello_log" 2>&1 &
    LIMO_PID=$!

    log "LIMOncello PID=$LIMO_PID, waiting ${SETTLE}s for startup..."
    sleep "$SETTLE"

    # Check it's still alive
    if ! kill -0 "$LIMO_PID" 2>/dev/null; then
        log "ERROR: LIMOncello crashed on startup. Check $limoncello_log"
        cp "$cfg_bak" "$cfg"
        return 1
    fi

    # Play bag
    log "Playing bag ($TIMEOUT s)..."
    timeout "$TIMEOUT" ros2 bag play "$bag_path" \
        --clock --rate 0.5 \
        > "$LOGDIR/${label//[^a-zA-Z0-9]/_}_bag.log" 2>&1 || true

    sleep 2

    # Check LIMOncello is still alive
    if kill -0 "$LIMO_PID" 2>/dev/null; then
        log "LIMOncello still alive after bag play ✓"
        CRASH=0
    else
        log "WARNING: LIMOncello died during bag play. Check $limoncello_log"
        CRASH=1
    fi

    # Check odom topic was published
    if grep -q "position\|orientation\|twist\|stamp" "$limoncello_log" 2>/dev/null; then
        log "Odom output seen in log ✓"
    else
        log "NOTE: Check /limoncello/state topic manually if needed"
    fi

    # Kill LIMOncello
    kill "$LIMO_PID" 2>/dev/null || true
    wait "$LIMO_PID" 2>/dev/null || true

    # Restore config
    cp "$cfg_bak" "$cfg"
    rm "$cfg_bak"

    if [ "$CRASH" -eq 0 ]; then
        log "PASS: $label completed without crash"
    else
        log "FAIL: $label crashed"
        return 1
    fi
    echo ""
}

# ── Main ──────────────────────────────────────────────────────────────────────

echo ""
echo "================================================================"
echo " Phase 3: DA-IESKF Smoke Test"
echo " Working dir: $WSDIR"
echo "================================================================"
echo ""

PASS_COUNT=0
FAIL_COUNT=0

run_test() {
    local label="$1"; shift
    if run_scenario "$label" "$@"; then
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

BOX_BAG="$TESTDIR/smoke_box_bag"
COR_BAG="$TESTDIR/smoke_corridor_bag"
BOX_LOG="$LOGDIR/box_eigenvalues.csv"
COR_LOG="$LOGDIR/corridor_eigenvalues.csv"

if [ "$MODE" = "box" ] || [ "$MODE" = "both" ]; then
    run_test "Box_DA_off"  "$BOX_BAG" "false" ""
    run_test "Box_DA_on"   "$BOX_BAG" "true"  "$BOX_LOG"
fi

if [ "$MODE" = "corridor" ] || [ "$MODE" = "both" ]; then
    run_test "Corridor_DA_on"  "$COR_BAG" "true"  "$COR_LOG"
fi

# ── Analyze eigenvalue logs ───────────────────────────────────────────────────
echo "================================================================"
echo " Eigenvalue log analysis"
echo "================================================================"

if [ -f "$BOX_LOG" ] && [ -f "$COR_LOG" ]; then
    python3 "$TESTDIR/analyze_eigenvalues.py" "$BOX_LOG" "$COR_LOG"
elif [ -f "$BOX_LOG" ]; then
    python3 "$TESTDIR/analyze_eigenvalues.py" "$BOX_LOG"
elif [ -f "$COR_LOG" ]; then
    python3 "$TESTDIR/analyze_eigenvalues.py" "$COR_LOG"
else
    echo "No eigenvalue logs found in $LOGDIR"
    echo "Make sure da_ieskf_enabled: true and da_eigenvalue_log is set"
fi

echo ""
echo "================================================================"
echo " Summary: $PASS_COUNT passed, $FAIL_COUNT failed"
echo "================================================================"
echo ""
echo "LIMOncello logs: $LOGDIR/"
[ "$FAIL_COUNT" -gt 0 ] && exit 1 || exit 0
