# DA-IESKF Integration into LIMOncello — Road to RA-L / ICRA 2027

## Paper Thesis

Standard IESKF produces overconfident covariance in geometrically degenerate environments (corridors, tunnels). This makes loop closure actively harmful: the pose graph optimizer receives inconsistent information (tight odometry covariance vs large loop closure correction), producing a worse solution than no loop closure at all (-49.4% degradation in simulation).

Degeneracy-Aware IESKF (DA-IESKF) inflates covariance along degenerate directions detected via Hessian eigenvalue analysis. This keeps the estimation problem consistent: honest uncertainty allows (1) mid-trajectory corrections at intermittent features, and (2) loop closure to dominate in degenerate directions (+40.8% improvement in simulation).

**Core claim**: Degeneracy-aware covariance is a prerequisite for loop closure to function correctly in degenerate environments. Without it, loop closure degrades performance.

## Simulation Results (already done)

```
Filter       | Pre-LC RMSE | Post-LC RMSE | Improvement
-------------|-------------|--------------|------------
DA-IESKF     | 3.51 m      | 2.08 m       | +40.8%
IESKF        | 6.79 m      | 10.14 m      | -49.4%

Mean P[x,x] ratio (DA/IE): 3.91x in degenerate corridor
IESKF x-drift: 23.72 m (x=23.92 vs GT x=0.20)

Non-degenerate control:
DA-IESKF: 3.03 m → 2.47 m (+18.6%)
IESKF:    1.88 m → 1.42 m (+24.3%)
```

Known limitation: DA-IESKF has ~1m pre-LC regression in non-degenerate environments (3.03m vs 1.88m). Adaptive threshold tuning is future work.

Code: `src/pluto_experiments/pluto_experiments/ieskf_showdown/gap_a_loop_closure.py`

---

## Repository Layout (this repo)

```
da_ieskf_limoncello/
├── include/
│   ├── DegeneracyDetector.hpp      ← eigenvalue analysis + covariance inflation
│   ├── PoseGraph.hpp               ← GTSAM pose graph wrapper
│   └── ScanContextManager.hpp      ← loop closure detection + GICP alignment
├── config/
│   └── da_ieskf_params.yaml        ← all tunable parameters
├── patches/
│   └── State_hpp_changes.md        ← annotated guide for modifying LIMOncello
└── scripts/
    ├── setup_monster.sh            ← install GTSAM, evo on monster
    ├── download_all.sh             ← download MCD, GrandTour, City02, R-Campus
    ├── evaluate_all.sh             ← batch run all 4 configs × all sequences
    └── plot_results.py             ← generate all paper figures from CSV logs
```

Copy `da_ieskf_limoncello/include/*.hpp` into `~/ros2_ws_limo/src/LIMOncello/include/`  
Apply changes from `patches/State_hpp_changes.md` to `include/State.hpp`

---

## Phase 0: Dataset Download (monster, start immediately)

```bash
# On monster (bash only)
cd ~/pluto_robot/da_ieskf_limoncello/scripts
chmod +x *.sh
./setup_monster.sh          # installs deps (~10 min)
./download_all.sh           # starts downloads (~overnight)
```

---

## Phase 1: Verify Baseline LIMOncello on Real Data (Week 1-2)

```bash
# On monster
mkdir -p ~/ros2_ws_limo/src
cd ~/ros2_ws_limo/src
git clone https://github.com/CPerezRuiz335/LIMOncello.git
cd LIMOncello && git submodule update --init --recursive
cd ~/ros2_ws_limo
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# Run on ntu_day_02 first (smallest, target APE: 0.156m)
source ~/ros2_ws_limo/install/setup.bash
ros2 launch limoncello limoncello.launch.py config_name:=mcd_mid70 use_sim_time:=true &
ros2 bag play ~/datasets/mcd/ntu_day_02/

# Evaluate
evo_ape tum ~/datasets/mcd/ntu_day_02/gt.tum ~/results/baseline/ntu_day_02.tum \
    -va --plot --save_results ~/results/baseline/ntu_day_02.zip
```

Gate: reproduce ntu_day_02 APE ≤ 0.190m (paper: 0.156m ± 20%).

---

## Phase 2: Implement DA-IESKF in C++ (Week 3-6)

1. Copy headers: `cp da_ieskf_limoncello/include/*.hpp ~/ros2_ws_limo/src/LIMOncello/include/`
2. Apply State.hpp changes per `patches/State_hpp_changes.md`
3. Update `CMakeLists.txt` to add GTSAM dependency
4. Run on ntu_day_02, verify eigenvalue CSV is generated

Gate: DA-IESKF runs on ntu_day_02 without crashing. Eigenvalues logged to CSV.

---

## Phase 3: Add Loop Closure (Week 5-8)

1. `ScanContextManager` runs at each frame, stores descriptors
2. On loop candidate: run GICP, check fitness score
3. If accepted: add to GTSAM factor graph, run LM optimizer
4. Apply corrections to trajectory and i-Octree (partial rebuild)

Gate: loop closure triggers on ntu_day_01 (has revisits). Trajectory improves.

---

## Phase 4: Full Evaluation (Week 7-10)

```bash
# On monster
cd ~/pluto_robot/da_ieskf_limoncello/scripts
./evaluate_all.sh           # runs all 4 configs × all sequences (~4-8 hrs)
python3 plot_results.py     # generates paper figures
```

Configs:
- A: IESKF, no LC  
- B: IESKF + LC  
- C: DA-IESKF, no LC  
- D: DA-IESKF + LC  ← expected best on degenerate sequences

---

## Phase 5: Paper Writing (Week 9-12)

Target: RA-L with ICRA 2027 presentation option (Seoul, May 24-28, 2027).  
Fallback: Direct ICRA 2027 (deadline ~Sep 15, 2026).

Title: "Why Loop Closure Fails in Degenerate Environments: Degeneracy-Aware Filtering on SGal(3) for Consistent LiDAR-Inertial SLAM"

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| City02 unavailable | HIGH | Use MCD tunnel-like sequences as backup |
| IESKF+LC degradation doesn't reproduce on real data | HIGH | Pivot to "DA improves LC effectiveness" |
| GTSAM integration slow | MEDIUM | Use Ceres or simple LM solver as fallback |
| Non-degen regression too large | MEDIUM | Adaptive threshold (scale by eigenvalue gap) |

---

## Collaborators to contact (after Phase 1)

- Carlos Perez-Ruiz (LIMOncello, IRI Barcelona) — email with baseline repro + sim results
- Joan Solà (LIMOncello co-author) — if Perez-Ruiz responds positively
