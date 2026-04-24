# DA-IESKF for LIMOncello

Degeneracy-Aware Iterated Error-State Kalman Filter integrated into the
[LIMOncello](https://github.com/fetty31/LIMOncello) LiDAR-Inertial SLAM system.

When a LiDAR scan provides little geometric constraint in certain directions
(long corridors, tunnels, flat open spaces), standard IESKF over-trusts the
LiDAR update and drifts. DA-IESKF detects these degenerate directions from the
Hessian eigenvalue structure and inflates the covariance along them, forcing the
filter to rely on IMU propagation instead.

## Results

### NTU VIRAL `eee_03` — Indoor academic building, 181 s

| Method | APE RMSE | APE Max |
|---|---|---|
| Baseline IESKF | 0.281 m | 0.658 m |
| **DA-IESKF** | **0.248 m** | **0.597 m** |
| **Improvement** | **−11.8 %** | −9.3 % |

- Degenerate frames detected: **12.9 %** (1368 / 10578 IESKF iterations)
- Sensor: Ouster OS1-16 at 10 Hz, VN-100 IMU at 385 Hz

### Phase 3 Smoke Test — Synthetic bags

| Scene | Degen frames | Mean ratio |
|---|---|---|
| Box room (6 walls, non-degenerate) | ~0 % | 0.81 |
| Corridor (2 walls + floor) | 99.7 % | 0.021 |

## Algorithm

```
For each LiDAR scan s, IESKF iteration k:

  1. Match scan points to local map → N plane residuals
  2. Build Jacobian H  [N × DoFObs]
  3. Compute information:  HTH = Hᵀ H / R

  4. Extract translation block:  A = HTH[6:9, 6:9]
         (SGal3d tangent order: velocity[0:3], rotation[3:6], translation[6:9])

  5. Eigen-decompose A:  λ₁ ≤ λ₂ ≤ λ₃
     ratio = λ_min / λ_max   (= 1 in isotropic room, ≈ 0 in corridor)

  6. If ratio < threshold AND λ_max > ε:
         for each eigenvector vᵢ with λᵢ/λ_max < threshold:
             P ← P + α · vᵢ vᵢᵀ          (covariance inflation)

  7. Continue IESKF update with inflated P
     → Kalman gain K ≈ 0 along degenerate dirs → IMU dominates there
```

Key design choices vs naive implementations:

- **Translation block, not full state**: SGal3d tangent order puts translation
  at indices 6–8. Using `topLeftCorner<3,3>()` (velocity block) produces
  τ-weighted false positives — every early-sweep point creates apparent
  anisotropy unrelated to geometry.

- **Guard for empty map**: when `λ_max < 1e-6` there are no plane matches
  (empty iOctree during startup). The IESKF gain is already zero in this case;
  inflation is a no-op and incorrectly flags the initialization phase.

- **Soft inflation, not hard rejection**: `P += α v vᵀ` admits calibrated
  uncertainty and preserves the posterior mean. Hard rejection (zero out H
  columns) loses information permanently.

## Repository layout

```
da_ieskf_limoncello/
├── include/
│   ├── DegeneracyDetector.hpp     # Core algorithm — drop-in header
│   ├── PoseGraph.hpp              # Loop closure graph (GTSAM, Phase 5)
│   └── ScanContextManager.hpp     # Scan Context descriptor (Phase 5)
├── config/
│   └── da_ieskf_params.yaml       # Reference parameter file
├── patches/
│   └── State_hpp_changes.md       # Diff guide for LIMOncello State.hpp
├── phase3_smoke_test/
│   ├── generate_synthetic_bag.py  # Builds box-room and corridor ROS2 bags
│   ├── run_smoke_test.sh          # Runs both scenarios and checks logs
│   ├── analyze_eigenvalues.py     # Validates corridor >> box degeneracy
│   ├── smoke_test.yaml            # LIMOncello config for smoke test
│   └── sample_logs/               # Reference CSV outputs (box, corridor)
├── scripts/
│   ├── run_phase4_eee03.sh        # Phase 4: real-data evaluation on eee_03
│   ├── evaluate_all.sh            # Multi-sequence evaluation (Phase 6)
│   ├── plot_results.py            # Figures for paper
│   ├── download_all.sh            # Dataset download helper
│   └── setup_monster.sh           # Remote server setup
└── results/
    └── eee03_phase4_summary.txt   # Phase 4 numerical results
```

## Integration into LIMOncello

### 1. Copy headers

```bash
cp da_ieskf_limoncello/include/DegeneracyDetector.hpp \
   ~/LIMOncello_ws/src/LIMOncello/include/

cp da_ieskf_limoncello/include/PoseGraph.hpp \
   da_ieskf_limoncello/include/ScanContextManager.hpp \
   ~/LIMOncello_ws/src/LIMOncello/include/
```

### 2. Patch `Config.hpp`

Add to the `IKFoM` struct:

```cpp
bool        da_ieskf_enabled       = false;
float       da_eigenvalue_threshold = 0.15f;
float       da_inflation_alpha      = 100.0f;
std::string da_eigenvalue_log       = "";
```

### 3. Patch `ROSutils.hpp`

Add after the plane parameters:

```cpp
n->get_parameter("IKFoM.da_ieskf_enabled",        cfg.ikfom.da_ieskf_enabled);
n->get_parameter("IKFoM.da_eigenvalue_threshold",  cfg.ikfom.da_eigenvalue_threshold);
n->get_parameter("IKFoM.da_inflation_alpha",       cfg.ikfom.da_inflation_alpha);
n->get_parameter("IKFoM.da_eigenvalue_log",        cfg.ikfom.da_eigenvalue_log);
```

### 4. Patch `State.hpp`

Add includes at top:

```cpp
#include <fstream>
#include "DegeneracyDetector.hpp"
```

Add DA-IESKF block **after** `Mat<DoFObs> HTH = H.transpose() * H / R;`
and **before** `Mat<DoFS2> P_inv = P.inverse();`:

```cpp
if (cfg.ikfom.da_ieskf_enabled) {
    // SGal3d tangent: [velocity(0:3), rotation(3:6), translation(6:9)]
    // Use translation block — standard LOAM degeneracy indicator
    static da_ieskf::DegeneracyDetector da_det(
        cfg.ikfom.da_eigenvalue_threshold,
        cfg.ikfom.da_inflation_alpha,
        DoFS2, 3);

    Eigen::MatrixXd info_pos = HTH.template block<3,3>(6,6).template cast<double>();
    auto deg = da_det.analyse_info(info_pos);

    if (deg.is_degenerate) {
        da_det.inflate(P, deg);
    }

    if (!cfg.ikfom.da_eigenvalue_log.empty()) {
        static std::ofstream eigen_log_(cfg.ikfom.da_eigenvalue_log);
        static bool eigen_log_header_ = [&]() {
            eigen_log_ << "timestamp,ratio,is_degenerate,n_degen_dims\n";
            return true;
        }();
        (void)eigen_log_header_;
        eigen_log_ << std::fixed << std::setprecision(6)
                   << stamp << "," << deg.eigenvalue_ratio << ","
                   << (deg.is_degenerate ? 1 : 0) << ","
                   << deg.n_degenerate_dims << "\n";
    }
}
```

### 5. Add DA config to your YAML

```yaml
IKFoM:
  # ... existing params ...
  da_ieskf_enabled: true
  da_eigenvalue_threshold: 0.15   # ratio < threshold → degenerate direction
  da_inflation_alpha: 100.0       # covariance inflation scale
  da_eigenvalue_log: "/tmp/eigenvalues.csv"   # set "" to disable logging
```

### 6. Build

```bash
cd ~/LIMOncello_ws
colcon build --packages-select limoncello --cmake-args -DCMAKE_BUILD_TYPE=Release
```

## Running the evaluation

### Phase 3 — Synthetic smoke test

```bash
cd ~/LIMOncello_ws
source install/setup.bash

# Generate synthetic bags (requires rosbags Python library)
python3 test/generate_synthetic_bag.py

# Run both scenarios and validate eigenvalue discrimination
bash test/run_smoke_test.sh both
```

Expected output:
```
[PASS] Corridor degen%  >  box degen% + 30
[PASS] Corridor ratio <<  box ratio / 5
[PASS] Corridor:  > 50% frames flagged
[PASS] No NaN in ratios
>>> ALL CHECKS PASS
```

### Phase 4 — NTU VIRAL real data

```bash
# Dataset: https://ntu-viral.github.io/
# Place bag at: ~/datasets/ntu_viral/eee_03/eee_03_ros2
# Ground truth: ~/datasets/ntu_viral/gt/eee_03_gt_tum.txt

source /opt/ros/jazzy/setup.bash
source ~/LIMOncello_ws/install/setup.bash
bash ~/LIMOncello_ws/test/run_phase4_eee03.sh
```

Outputs written to `~/LIMOncello_ws/results/eee03/`:
- `baseline.tum` / `da.tum` — trajectories in TUM format
- `eigenvalues_da.csv` — per-iteration degeneracy log
- `evo_baseline.txt` / `evo_da.txt` — evo_ape reports
- `summary.txt` — RMSE comparison

## Parameters

| Parameter | Default | Effect |
|---|---|---|
| `da_eigenvalue_threshold` | 0.15 | Ratio below which a direction is degenerate. Lower = more selective. Tune down to 0.05 for outdoor/unstructured environments. |
| `da_inflation_alpha` | 100.0 | Covariance inflation magnitude. Larger = filter ignores LiDAR more in degenerate dirs. 100 is effectively a soft rejection. |
| `da_eigenvalue_log` | `""` | Path to CSV log. Empty string disables logging. Useful for offline analysis and figure generation. |

## Debugging

If DA-IESKF makes trajectory worse:

1. **Check degenerate rate**: if > 30 % on a normal indoor sequence, threshold
   is too loose. Reduce `da_eigenvalue_threshold` to 0.05–0.10.

2. **Check which block is analyzed**: must be the translation block
   `HTH.block<3,3>(6,6)`, not `topLeftCorner<3,3>()` (velocity block —
   produces false positives from per-point timestamps).

3. **Check initialization**: if all degenerate frames are in the first 10 s,
   the empty-map guard (`lambda_max < 1e-6`) may need tuning.

4. **Verify process isolation**: when running baseline then DA back-to-back,
   ensure the previous LIMOncello node is fully killed before the next run.
   Use `pkill -f limoncello` after `kill $LAUNCH_PID`.

## Status

| Phase | Description | Status |
|---|---|---|
| 0–1 | Theory, Python simulation, gap analysis | Done |
| 2 | C++ integration into LIMOncello | Done |
| 3 | Synthetic bag smoke test (4/4 pass) | Done |
| 4 | Real data evaluation — NTU VIRAL eee_03 | Done |
| 5 | Loop closure (PoseGraph + ScanContext + GTSAM) | Pending |
| 6 | Full multi-sequence evaluation | Pending |
| 7 | Paper writing | Pending |

## Dependencies

- [LIMOncello](https://github.com/fetty31/LIMOncello) (ROS 2 Jazzy)
- [manif](https://github.com/artivis/manif) — SGal(3) Lie group (bundled with LIMOncello)
- [evo](https://github.com/MichaelGrupp/evo) — trajectory evaluation (`pip install evo`)
- [rosbags](https://github.com/rpng/rosbags) — synthetic bag generation (`pip install rosbags`)
- GTSAM — loop closure (Phase 5, requires separate install)
