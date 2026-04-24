#!/usr/bin/env bash
# run_phase4_eee03.sh — Phase 4: DA-IESKF on real NTU VIRAL eee_03 dataset
#
# Runs 2 configs and evaluates with evo:
#   baseline: IESKF (DA off)
#   da:       DA-IESKF (DA on, eigenvalue log enabled)
#
# Output:
#   ~/LIMOncello_ws/results/eee03/baseline.tum
#   ~/LIMOncello_ws/results/eee03/da.tum
#   ~/LIMOncello_ws/results/eee03/eigenvalues_da.csv
#   ~/LIMOncello_ws/results/eee03/evo_baseline.txt
#   ~/LIMOncello_ws/results/eee03/evo_da.txt
#   ~/LIMOncello_ws/results/eee03/summary.txt

set -eo pipefail

WSDIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="$WSDIR/results/eee03"
BAG="$HOME/datasets/ntu_viral/eee_03/eee_03_ros2"
GT="$HOME/datasets/ntu_viral/gt/eee_03_gt_tum.txt"

source /opt/ros/jazzy/setup.bash
source "$WSDIR/install/setup.bash"
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$RESULTS"

log() { echo "[phase4] $(date +%H:%M:%S) $*"; }

# ── TUM recorder node ─────────────────────────────────────────────────────────
# LIMOncello publishes nav_msgs/Odometry on /limoncello/state
# We subscribe and write TUM lines: timestamp tx ty tz qx qy qz qw

TUM_RECORDER="$WSDIR/test/odom_to_tum.py"
cat > "$TUM_RECORDER" << 'PYEOF'
#!/usr/bin/env python3
"""Subscribe to /limoncello/state (Odometry) and write TUM file."""
import sys, rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

class TumWriter(Node):
    def __init__(self, path):
        super().__init__('tum_writer')
        self.f = open(path, 'w')
        self.create_subscription(Odometry, '/limoncello/state', self.cb, 100)
        self.get_logger().info(f'Writing TUM to {path}')

    def cb(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.f.write(f'{t:.9f} {p.x:.6f} {p.y:.6f} {p.z:.6f} '
                     f'{q.x:.6f} {q.y:.6f} {q.z:.6f} {q.w:.6f}\n')
        self.f.flush()

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/traj.tum'
    rclpy.init()
    node = TumWriter(path)
    rclpy.spin(node)
    node.f.close()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
PYEOF
chmod +x "$TUM_RECORDER"

# ── Run one config ─────────────────────────────────────────────────────────────
run_config() {
    local label="$1"
    local config_name="$2"
    local tum_out="$RESULTS/${label}.tum"
    local limo_log="$RESULTS/${label}_limo.log"
    local bag_rate="${3:-1.0}"

    log "=== $label ==="
    rm -f "$tum_out"

    # Start LIMOncello via launch (sets namespace='limoncello' → /limoncello/state)
    ros2 launch limoncello limoncello.launch.py \
        config_name:="${config_name}" \
        use_sim_time:=true \
        rviz:=false \
        > "$limo_log" 2>&1 &
    LIMO_PID=$!

    # Start TUM recorder
    python3 "$TUM_RECORDER" "$tum_out" > /dev/null 2>&1 &
    TUM_PID=$!

    log "LIMOncello PID=$LIMO_PID, TUM recorder PID=$TUM_PID"
    sleep 3   # wait for startup + IMU calibration

    # Play bag
    log "Playing eee_03 bag at ${bag_rate}x..."
    ros2 bag play "$BAG" --clock --rate "$bag_rate" > "$RESULTS/${label}_bag.log" 2>&1
    log "Bag finished."

    sleep 3   # flush output

    # Kill launch wrapper AND all child processes (limoncello node, component manager, etc.)
    kill "$LIMO_PID" "$TUM_PID" 2>/dev/null || true
    pkill -f "limoncello" 2>/dev/null || true
    wait "$LIMO_PID" 2>/dev/null || true
    sleep 2   # let ROS2 executor tear down cleanly

    local n_poses
    n_poses=$(wc -l < "$tum_out" 2>/dev/null || echo 0)
    log "$label: $n_poses poses written to $tum_out"

    if [ "$n_poses" -lt 100 ]; then
        log "WARNING: very few poses ($n_poses). Check $limo_log"
        return 1
    fi
    return 0
}

# ── Run baseline (DA off) ─────────────────────────────────────────────────────
log "Starting Phase 4 evaluation on eee_03 (181s sequence)"
log "Results: $RESULTS"
echo ""

run_config "baseline" "ntu_viral_eee03"
echo ""

# ── Run DA-IESKF ──────────────────────────────────────────────────────────────
run_config "da" "ntu_viral_eee03_da"
echo ""

# ── Evaluate with evo ─────────────────────────────────────────────────────────
log "Evaluating with evo_ape..."

for label in baseline da; do
    TUM="$RESULTS/${label}.tum"
    OUT="$RESULTS/evo_${label}.txt"
    if [ ! -f "$TUM" ]; then
        log "SKIP evo for $label: no TUM file"
        continue
    fi

    rm -f "$RESULTS/evo_${label}.zip"
    evo_ape tum "$GT" "$TUM" \
        --align --correct_scale \
        --save_results "$RESULTS/evo_${label}.zip" \
        2>&1 | tee "$OUT"
    echo ""
done

# ── Summary ───────────────────────────────────────────────────────────────────
{
    echo "=== Phase 4 Summary: DA-IESKF on eee_03 ==="
    echo "Date: $(date -u)"
    echo ""
    for label in baseline da; do
        OUT="$RESULTS/evo_${label}.txt"
        if [ -f "$OUT" ]; then
            RMSE=$(grep "rmse" "$OUT" | awk '{print $NF}' | head -1)
            echo "  $label: APE RMSE = ${RMSE:-N/A} m"
        fi
    done
    echo ""
    if [ -f "$RESULTS/eigenvalues_da.csv" ]; then
        N_ROWS=$(( $(wc -l < "$RESULTS/eigenvalues_da.csv") - 1 ))
        N_DEG=$(awk -F',' 'NR>1 && $3=="1"' "$RESULTS/eigenvalues_da.csv" | wc -l)
        echo "  DA eigenvalue log: $N_ROWS frames, $N_DEG degenerate"
    fi
} | tee "$RESULTS/summary.txt"

echo ""
log "Done. Results in $RESULTS/"
