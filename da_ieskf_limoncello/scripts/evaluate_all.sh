#!/usr/bin/env bash
# evaluate_all.sh — batch evaluation of all 4 configs × all sequences
#
# Configs:
#   A: IESKF,    no loop closure   (baseline)
#   B: IESKF +  loop closure       (shows LC makes things worse in degenerate)
#   C: DA-IESKF, no loop closure   (shows DA improves odometry alone)
#   D: DA-IESKF + loop closure     (expected best on degenerate sequences)
#
# Results written to ~/results/<config>/<sequence>/
set -euo pipefail

RESULTS_ROOT="${RESULTS_ROOT:-$HOME/results}"
DATASETS_ROOT="${DATASETS_ROOT:-$HOME/datasets}"
ROS_WS="${ROS_WS:-$HOME/ros2_ws_limo}"
LOG_DIR="$RESULTS_ROOT/logs"
mkdir -p "$LOG_DIR"

echo "=== DA-IESKF batch evaluation ==="
echo "Date:    $(date -u)"
echo "Results: $RESULTS_ROOT"
echo ""

source /opt/ros/jazzy/setup.bash
source "$ROS_WS/install/setup.bash"

# ── Sequences ────────────────────────────────────────────────────────────────
declare -A BAGS
BAGS["ntu_day_01"]="$DATASETS_ROOT/mcd/ntu_day_01"
BAGS["ntu_day_02"]="$DATASETS_ROOT/mcd/ntu_day_02"
BAGS["ntu_day_10"]="$DATASETS_ROOT/mcd/ntu_day_10"
BAGS["grand_tour"]="$DATASETS_ROOT/grand_tour"
BAGS["city02"]="$DATASETS_ROOT/city02"
BAGS["r_campus"]="$DATASETS_ROOT/r_campus"

declare -A CONFIGS
CONFIGS["A"]="da_ieskf_enabled:=false loop_closure_enabled:=false"
CONFIGS["B"]="da_ieskf_enabled:=false loop_closure_enabled:=true"
CONFIGS["C"]="da_ieskf_enabled:=true  loop_closure_enabled:=false"
CONFIGS["D"]="da_ieskf_enabled:=true  loop_closure_enabled:=true"

CONFIG_NAMES["A"]="IESKF_noLC"
CONFIG_NAMES["B"]="IESKF_LC"
CONFIG_NAMES["C"]="DA-IESKF_noLC"
CONFIG_NAMES["D"]="DA-IESKF_LC"

# ── Phase 1 gate: ntu_day_02 baseline ────────────────────────────────────────
run_sequence() {
    local cfg="$1"
    local seq="$2"
    local bag_dir="${BAGS[$seq]}"
    local out_dir="$RESULTS_ROOT/$cfg/$seq"

    if [ ! -d "$bag_dir" ]; then
        echo "  [SKIP] $seq: bag not found at $bag_dir"
        return 0
    fi

    mkdir -p "$out_dir"
    local tum_out="$out_dir/estimated.tum"
    local log_out="$LOG_DIR/${cfg}_${seq}.log"

    if [ -f "$tum_out" ]; then
        echo "  [SKIP] $cfg/$seq: result exists"
        return 0
    fi

    echo "  [RUN]  $cfg/$seq ..."

    # Launch LIMOncello (background)
    ros2 launch limoncello limoncello.launch.py \
        config_name:=mcd_mid70 \
        use_sim_time:=true \
        tum_output:="$tum_out" \
        eigenvalue_log:="$out_dir/eigenvalues.csv" \
        ${CONFIGS[$cfg]} \
        > "$log_out" 2>&1 &
    LIMO_PID=$!

    # Play bag
    sleep 3
    ros2 bag play "$bag_dir" --clock 2>>"$log_out"

    # Wait for LIMOncello to finish writing
    sleep 5
    kill $LIMO_PID 2>/dev/null || true
    wait $LIMO_PID 2>/dev/null || true

    echo "  [DONE] $cfg/$seq → $tum_out"
}

# ── Evaluate with evo ─────────────────────────────────────────────────────────
evaluate_sequence() {
    local cfg="$1"
    local seq="$2"
    local bag_dir="${BAGS[$seq]}"
    local out_dir="$RESULTS_ROOT/$cfg/$seq"
    local tum_out="$out_dir/estimated.tum"
    local gt_tum="$bag_dir/gt.tum"

    if [ ! -f "$tum_out" ]; then
        echo "  [SKIP eval] $cfg/$seq: no result TUM"
        return 0
    fi
    if [ ! -f "$gt_tum" ]; then
        echo "  [SKIP eval] $cfg/$seq: no GT TUM at $gt_tum"
        return 0
    fi

    echo "  [EVAL] $cfg/$seq ..."
    evo_ape tum "$gt_tum" "$tum_out" \
        -va \
        --save_results "$out_dir/evo_ape.zip" \
        --save_plot "$out_dir/evo_ape.pdf" \
        2>&1 | tee "$out_dir/evo_ape.txt" | tail -5

    # Extract RMSE to summary CSV
    RMSE=$(grep "rmse" "$out_dir/evo_ape.txt" | awk '{print $NF}')
    echo "$cfg,$seq,$RMSE" >> "$RESULTS_ROOT/summary.csv"
}

# ── Main loop ─────────────────────────────────────────────────────────────────
echo "timestamp,config,sequence,rmse_m" > "$RESULTS_ROOT/summary.csv"

# Phase 1 gate first: ntu_day_02 with config A (baseline)
echo "--- Phase 1 gate: ntu_day_02 baseline ---"
run_sequence "A" "ntu_day_02"
evaluate_sequence "A" "ntu_day_02"

GATE_RMSE=$(grep "^A,ntu_day_02," "$RESULTS_ROOT/summary.csv" | cut -d, -f3)
if [ -n "$GATE_RMSE" ]; then
    GATE_PASS=$(python3 -c "print('PASS' if float('$GATE_RMSE') <= 0.190 else 'FAIL')")
    echo ""
    echo "Phase 1 gate: ntu_day_02 APE = $GATE_RMSE m (threshold 0.190 m) → $GATE_PASS"
    if [ "$GATE_PASS" = "FAIL" ]; then
        echo "WARNING: LIMOncello baseline does not reproduce paper results."
        echo "  Check config, extrinsics, and bag topic names before continuing."
        echo "  Proceeding anyway for diagnostics..."
    fi
fi
echo ""

# Full evaluation
echo "--- Full evaluation: all configs × all sequences ---"
for cfg in A B C D; do
    echo ""
    echo "Config $cfg (${CONFIG_NAMES[$cfg]}):"
    for seq in "${!BAGS[@]}"; do
        run_sequence "$cfg" "$seq"
    done
done

echo ""
echo "--- Evaluation with evo ---"
for cfg in A B C D; do
    for seq in "${!BAGS[@]}"; do
        evaluate_sequence "$cfg" "$seq"
    done
done

echo ""
echo "=== Batch evaluation complete ==="
echo "Summary: $RESULTS_ROOT/summary.csv"
echo ""
cat "$RESULTS_ROOT/summary.csv"
echo ""
echo "Run: python3 $(dirname "$0")/plot_results.py --results $RESULTS_ROOT"
