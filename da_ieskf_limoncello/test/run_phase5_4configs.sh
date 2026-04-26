#!/usr/bin/env bash
# run_phase5_4configs.sh — Phase 5: 4-config comparison on NTU VIRAL eee_03
#
# Configs:
#   A: baseline IESKF,   no LC
#   B: baseline IESKF,   LC enabled
#   C: DA-IESKF,         no LC   (Phase 4 result: 0.248m)
#   D: DA-IESKF,         LC enabled
#
# Core claim: DA-IESKF + LC (D) < DA-IESKF no-LC (C) < baseline no-LC (A)
#             baseline + LC (B) >= baseline no-LC (A)  ← LC hurts with tight cov

set -eo pipefail

WSDIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="$WSDIR/results/phase5"
BAG="$HOME/datasets/ntu_viral/eee_03/eee_03_ros2"
GT="$HOME/datasets/ntu_viral/gt/eee_03_gt_tum.txt"

source /opt/ros/jazzy/setup.bash 2>/dev/null || true
source "$WSDIR/install/setup.bash"
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"

mkdir -p "$RESULTS"

log() { echo "[phase5] $(date +%H:%M:%S) $*"; }

# ── TUM recorder (reuse from Phase 4) ────────────────────────────────────────
TUM_RECORDER="$WSDIR/test/odom_to_tum.py"

# ── Run one config ─────────────────────────────────────────────────────────────
run_config() {
    local label="$1"
    local config_name="$2"
    local tum_out="$RESULTS/${label}.tum"
    local limo_log="$RESULTS/${label}_limo.log"

    log "=== $label ($config_name) ==="
    rm -f "$tum_out"

    # Kill any stale SLAM/recorder processes from previous runs
    pkill -9 -f "limoncello" 2>/dev/null || true
    pkill -9 -f "odom_to_tum" 2>/dev/null || true
    sleep 2

    ros2 launch limoncello limoncello.launch.py \
        config_name:="${config_name}" \
        use_sim_time:=true \
        rviz:=false \
        > "$limo_log" 2>&1 &
    LIMO_PID=$!

    python3 "$TUM_RECORDER" "$tum_out" > /dev/null 2>&1 &
    TUM_PID=$!

    sleep 3

    log "Playing bag..."
    ros2 bag play "$BAG" --clock --rate 1.0 > "$RESULTS/${label}_bag.log" 2>&1
    log "Bag finished."

    sleep 3

    kill "$LIMO_PID" "$TUM_PID" 2>/dev/null || true
    pkill -f "limoncello" 2>/dev/null || true
    wait "$LIMO_PID" 2>/dev/null || true
    sleep 2

    local n_poses
    n_poses=$(wc -l < "$tum_out" 2>/dev/null || echo 0)
    log "$label: $n_poses poses"
    [ "$n_poses" -lt 100 ] && log "WARNING: too few poses — check $limo_log"
    echo ""
}

# ── Build configs ─────────────────────────────────────────────────────────────
# Config A: baseline (DA off, LC disabled — da_ieskf_enabled: false)
# Config B: baseline + LC  (DA off, but LC runs — same YAML, LC always runs now)
# Config C: DA-IESKF no LC (already have from Phase 4, just copy)
# Config D: DA-IESKF + LC

# Config A and B use the same YAML (LC is always active in code now).
# To disable LC for A/C we need a flag — use da_eigenvalue_log path as proxy
# or simply accept LC runs for all; compare A vs D directly.
# For now: run all 4, LC is always enabled in Phase 5 binary.

log "=== Phase 5: 4-config evaluation on eee_03 ==="
log "Results: $RESULTS"
echo ""

# Config A: IESKF baseline + LC
run_config "config_a_ieskf_lc"   "ntu_viral_eee03"
# Config C: DA-IESKF + LC
run_config "config_d_da_lc"      "ntu_viral_eee03_da"

# ── Evaluate ──────────────────────────────────────────────────────────────────
log "Evaluating with evo_ape..."
echo ""

for label in config_a_ieskf_lc config_d_da_lc; do
    TUM="$RESULTS/${label}.tum"
    OUT="$RESULTS/evo_${label}.txt"
    [ -f "$TUM" ] || { log "SKIP $label: no TUM"; continue; }
    rm -f "$RESULTS/evo_${label}.zip"
    evo_ape tum "$GT" "$TUM" \
        --align --correct_scale \
        --save_results "$RESULTS/evo_${label}.zip" \
        2>&1 | tee "$OUT"
    echo ""
done

# ── Summary ───────────────────────────────────────────────────────────────────
{
    echo "=== Phase 5 Summary: 4-config comparison on eee_03 ==="
    echo "Date: $(date -u)"
    echo ""
    echo "  Reference (Phase 4, no LC):"
    echo "    config_a_no_lc (IESKF):    0.281 m"
    echo "    config_c_da_no_lc (DA):    0.248 m"
    echo ""
    echo "  Phase 5 (with LC):"
    for label in config_a_ieskf_lc config_d_da_lc; do
        OUT="$RESULTS/evo_${label}.txt"
        [ -f "$OUT" ] || continue
        RMSE=$(grep "rmse" "$OUT" | awk '{print $NF}' | head -1)
        echo "    $label: APE RMSE = ${RMSE:-N/A} m"
    done
    echo ""
    echo "  Prediction:"
    echo "    config_a_ieskf_lc >= 0.281 m  (LC hurts tight-cov odometry)"
    echo "    config_d_da_lc    <  0.248 m  (LC helps honest-cov odometry)"
    if [ -f "$RESULTS/eee03_lc_log.txt" ]; then
        LC_COUNT=$(grep -c "ACCEPTED" "$RESULTS/eee03_lc_log.txt" 2>/dev/null || echo 0)
        echo "    Loop closures accepted: $LC_COUNT"
    fi
} | tee "$RESULTS/summary.txt"

echo ""
log "Done. Results in $RESULTS/"
