# DA-IESKF + Loop Closure — Handoff Document

**Date:** 2026-04-26  
**Status:** Phase 5 complete, Phase 6 ready to start  
**Author:** Generated for Claude Code continuation on a new machine

---

## What This Project Is

A fork of LIMOncello (LiDAR-IMU SLAM) extended with:
1. **DA-IESKF** — Degeneracy-Aware IESKF: inflates Kalman covariance along degenerate eigenvectors (corridors, long walls, open fields) so the filter is honest about how uncertain it is in those directions.
2. **Loop Closure (Phase 5)** — Scan Context detection → SVD-ICP verification → Eigen-only pose graph optimizer. The central thesis: DA-IESKF's honest (inflated) covariance means the pose graph correctly trusts LC constraints in degenerate directions; standard IESKF's overconfident covariance means LC gets blocked.

---

## Repository Layout

```
/home/thailuu/pluto_robot/                  ← DA-IESKF algorithm headers only
  da_ieskf_limoncello/include/
    DegeneracyDetector.hpp                  ← eigenvalue analysis of H_lidar, inflates P
    PoseGraph.hpp                           ← pure Eigen GN optimizer (no GTSAM/Boost)
    ScanContextManager.hpp                  ← 20-ring×60-sector descriptor, ring-key pre-filter

/home/thailuu/LIMOncello_ws/               ← ROS2 workspace
  src/LIMOncello/
    src/main.cpp                           ← main SLAM node (all LC logic is here)
    include/Core/State.hpp                 ← IESKF state; da_det.inflate() in update step
    include/DegeneracyDetector.hpp         ← symlink/copy from pluto_robot
    include/PoseGraph.hpp                  ← symlink/copy from pluto_robot
    include/ScanContextManager.hpp         ← symlink/copy from pluto_robot
    config/ntu_viral_eee03.yaml            ← std IESKF + LC config
    config/ntu_viral_eee03_da.yaml         ← DA-IESKF + LC config
  test/
    run_phase4_eee03.sh                    ← baseline (no LC) evaluation
    run_phase5_4configs.sh                 ← 4-config Phase 5 evaluation
    odom_to_tum.py                         ← subscribes /limoncello/state → TUM file
  results/
    phase4/                                ← baseline APE results
    phase5/                                ← Phase 5 APE results
```

---

## Build

```bash
cd /home/thailuu/LIMOncello_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select limoncello --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

Headers in `pluto_robot/` are included via CMakeLists.txt include path — no separate build needed.

---

## Dataset

**NTU VIRAL eee_03** (`/home/thailuu/datasets/ntu_viral/`):
- Bag: `eee_03.bag` (ROS2 bag)
- DA bag: `ntu_viral_eee03_da/` (same data, different config trigger)
- Ground truth: `eee_03_gt_tum.txt`
- Sequence: ~190s indoor corridor, ~0.248m APE RMSE baseline

**Other sequences available** (for Phase 6): eee_01, eee_02, nya_01

---

## Phase 4 Results (Baseline, no loop closure)

| Config | APE RMSE |
|--------|----------|
| std IESKF (config_a_no_lc) | 0.281 m |
| DA-IESKF (config_c_da_no_lc) | **0.248 m** |

DA-IESKF beats std IESKF by 0.033m on eee_03. These are the reference numbers.

---

## Phase 5 Results (Loop Closure, run18 — final clean run)

| Config | APE RMSE | vs baseline | Notes |
|--------|----------|-------------|-------|
| IESKF + LC | 0.337 m | +0.056 m | LC hurts — tight cov blocks LC then distorts |
| DA-IESKF + LC | 0.265 m | +0.017 m | LC barely hurts — 3× more tolerant |

**Prediction A confirmed:** IESKF+LC > IESKF baseline ✓  
**Prediction B not confirmed (absolute):** DA-IESKF+LC = 0.265m, target was < 0.248m  
**Prediction B directionally confirmed:** DA-IESKF+LC margin is 3× smaller than IESKF+LC margin ✓

**Root cause of unmet absolute threshold:** eee_03's Scan Context detections fire at t≈25-29s (frames 255 and 290) when accumulated IESKF drift is only ~2mm. The 0.248m RMSE accumulates later in the sequence where there are no genuine loop closures. The LC correction at frame 255 is a 9mm shift — negligible relative to 0.248m total RMSE.

---

## Architecture: Loop Closure in main.cpp

The LC block runs **outside** `mtx_state_` (to not block 400Hz IMU propagation during GICP/optimization). After `pose_graph_.optimize()`, correction is applied **directly to `state_`** under `mtx_state_` — this is the critical design choice:

```
LiDAR callback (10Hz):
  1. mtx_state_.lock() → state_.update(filtered, ioctree_) → extract cur_pose, pose_cov → mtx_state_.unlock()
  2. mtx_lc_.lock() [outside mtx_state_!]:
     a. pose_graph_.addOdometryFactor(prev_frame, cur_frame, raw_T_rel, pose_cov)
        - raw_T_rel = prev_raw_ieskf_pose_.inverse() * cur_pose  ← raw, not optimized prev!
     b. Every kSCInterval=5 frames: sc_manager_.addFrame() + detectLoop()
     c. If LC detected + min gap (>30 frames):
        - Near-range filter (15m sphere) + 1.5m VoxelGrid on both clouds
        - Eigen SVD ICP, 15 iterations, max_dist=3.0m
        - pose_graph_.addLoopClosureFactor(from, to, T, fitness)  ← rejected if fitness > 0.3m²
     d. If hasNewLoopClosures():
        - pose_graph_.optimize()  ← Gauss-Newton, SimplicialLDLT, 10 iters
        - mtx_state_.lock() → state_.X.element<0>() = SGal3d(opt_p, opt_q, v, t) → mtx_state_.unlock()
        - prev_raw_ieskf_pose_ = opt  ← corrected pose is new base
     e. else: prev_raw_ieskf_pose_ = cur_pose
     f. lc_frame_counter_++
  3. pub_state_->publish(toROS(state_, sweep_time))

IMU callback (400Hz):
  1. mtx_state_.lock() → state_.predict(imu, dt) → mtx_state_.unlock()
  2. pub_state_->publish(toROS(state_, imu.stamp))  ← naturally reflects LC correction
```

**Critical invariant:** `prev_raw_ieskf_pose_` tracks the raw IESKF pose at the previous frame. After LC fires, it's set to the optimized pose (because `state_` was corrected to match, so next frame's `cur_pose` will be IESKF-integrated from the corrected base — making `raw_T_rel` clean incremental motion).

---

## Key Code Locations

### State.hpp — DA-IESKF inflation
The DA update happens at the end of `state_.update()`. Look for:
```cpp
da_det.inflate(P, deg);  // inflates P along degenerate eigenvectors by alpha=100
```
`P` is a 23×23 matrix. SGal3d tangent order: `velocity[0:3], rotation[3:6], translation[6:9]`.  
So `P.block<3,3>(3,3)` = rotation cov, `P.block<3,3>(6,6)` = translation cov.

### PoseGraph.hpp — Information matrix balance
```
info_odom = pose_cov.inverse()   (from IESKF P, large in degenerate dirs → small info → LC wins)
info_lc   = diag(1/0.2², 1/0.02²)  (fixed: translation 0.2m, rotation 0.02rad)
```
The Gauss-Newton accumulates `Hᵀ Ω H` per factor. With N odometry factors between frames 0→B, the LC at frame B gets effective weight ≈ `info_lc / (N·info_odom_avg)`. For DA-IESKF in degenerate corridor: `info_odom` is small → LC dominates.

### ICP fitness threshold (PoseGraph.hpp line 138)
```cpp
double lc_fitness_threshold = 0.3;   // m² 
```
On eee_03: genuine LCs score ~0.04m², marginal ones ~0.6m²+. Threshold 0.3m² gives clean separation.

---

## Common Bugs Already Fixed — Don't Re-introduce

1. **Hash-ordered stride cap**: PCL VoxelGrid output is hash-ordered, not spatial. Stride subsampling (e.g., every Nth point) gives spatially non-uniform sets → ICP diverges with fitness 2-4m². Fix: near-range filter (15m) + VoxelGrid (1.5m leaf), no stride.

2. **Optimized prev in T_rel**: Using `getOptimizedPose(k-1)` as `prev` when computing the next odometry factor injects the LC correction as a spurious T_rel → undoes the correction. Fix: always track `prev_raw_ieskf_pose_` (raw IESKF, updated to `opt` only when LC fires and `state_` is simultaneously corrected).

3. **400Hz state copy data race**: `State pub_imu = state_` at 400Hz without `mtx_state_` lock copies a 6KB struct concurrently with the LiDAR callback writing it → UB, corrupted poses, +0.035m regression. Fix: apply correction directly to `state_` under lock; IMU callback publishes `state_` directly.

4. **lc_fitness_threshold = -1.0**: This was a diagnostic mode disabling all LC. The correct value is `0.3`. If you see DA-IESKF+LC performing identically to DA-IESKF (no LC), check this value first.

5. **Dense LDLT**: Was O(N³), took ~12s at N=255 nodes → blocked ROS2 executor, dropped scans. Fix: SimplicialLDLT sparse solver, O(N) for chain graph.

---

## Phase 6 Plan (Next Steps)

**Goal:** Demonstrate `DA-IESKF+LC < DA-IESKF baseline` on sequences where loops close after meaningful drift.

**Why eee_03 doesn't work for the absolute claim:** The sequence has genuine loop closures only at t≈25-29s when drift is ~2mm. The 0.248m RMSE accumulates between t=60-190s with no revisits.

**Recommended sequences:**
- `eee_01`, `eee_02` — different indoor NTU VIRAL sequences, likely different loop structure
- `nya_01` — outdoor, may have longer loops with more accumulated drift before closure

**How to run Phase 6:**
```bash
# Edit test/run_phase5_4configs.sh to point at new sequence
# Or create test/run_phase6_multiseq.sh
bash /home/thailuu/LIMOncello_ws/test/run_phase5_4configs.sh
```

**What to look for:** LC detection time (check RCLCPP_INFO `LC: frame X → Y`) — you want frame X and Y to be far apart in time (>60s between them) so significant drift accumulated between passes.

**How to debug LC detections:**
```bash
# During a run, watch the SLAM node logs for:
# "LC: frame A → B, SC=0.xxx, ICP=0.xxxx [ACCEPTED/REJECTED]"
# "PG: frame N raw=(x,y,z) opt=(x,y,z) delta=(dx,dy,dz)"
```
A meaningful correction should show `|delta| > 0.05m`. If delta is <10mm, the LC fired too early.

---

## How to Run Phase 5 (Reproduce run18)

```bash
cd /home/thailuu/LIMOncello_ws
source install/setup.bash
bash test/run_phase5_4configs.sh 2>&1 | tee /tmp/phase5_runX.log
```

Results written to `/home/thailuu/LIMOncello_ws/results/phase5/`.

Expected output (run18 reference):
```
config_a_ieskf_lc: APE RMSE = 0.337529 m
config_d_da_lc:    APE RMSE = 0.265570 m
```
Pose count should be ~71K for both (full 400Hz trajectory, ~190s × 400Hz ≈ 76K).

---

## Git History

### `/home/thailuu/pluto_robot` (algorithm headers)
```
37320b2  Fix PoseGraph: restore LC threshold 0.3, replace GTSAM with pure Eigen
29ff890  Update root README: add DA-IESKF results and progress table
f743c22  Add DA-IESKF README with results, algorithm, and integration guide
a05eb42  Fix DA-IESKF: use translation block instead of velocity block
09a64a4  Fix run_phase4_eee03.sh: evo overwrite prompt + degen count
0ffea0f  Phase 4: DA-IESKF real data evaluation on NTU VIRAL eee_03
```

### `/home/thailuu/LIMOncello_ws/src/LIMOncello` (ROS2 node)
```
82ecf7c  Phase 5: DA-IESKF loop closure with Eigen-only pose graph (no GTSAM)
a42c5a4  Correct filtering frame, now in baselink frame, not world frame.
```

---

## Environment

- ROS2 Humble on Ubuntu 22.04
- Build: `colcon build --packages-select limoncello --cmake-args -DCMAKE_BUILD_TYPE=Release`
- Evaluation: `evo_ape tum --align --correct_scale` (Sim3 Umeyama)
- Python recorder: `test/odom_to_tum.py` subscribes to `/limoncello/state`

**Dependencies NOT needed:**
- GTSAM — deliberately removed (heap corruption in ROS2 callbacks)
- No PCL registration — ICP is pure Eigen SVD, zero PCL registration

**Dependencies that ARE needed:**
- PCL (for VoxelGrid filter only, `pcl/filters/voxel_grid.h`)
- Eigen3 (Dense + Sparse)
- manif (SGal3d state representation)
