# State.hpp Changes for DA-IESKF Integration

Apply these changes to `~/ros2_ws_limo/src/LIMOncello/include/State.hpp`.
Each section shows the original code and the replacement.

---

## 1. Add DA-IESKF includes at top of file

**After** the existing includes (around line 10–20):

```cpp
// --- ADD THESE ---
#include "DegeneracyDetector.hpp"
#include "PoseGraph.hpp"
#include "ScanContextManager.hpp"
#include <yaml-cpp/yaml.h>
```

---

## 2. Add DA-IESKF fields to the State struct

**Find** the State struct definition (search for `struct State`).
**After** the existing fields (bias, covariance P, etc.), add:

```cpp
  // ── DA-IESKF additions ──────────────────────────────────────────────────

  // Degeneracy detector (shared across all State instances via pointer)
  std::shared_ptr<da_ieskf::DegeneracyDetector> degeneracy_detector;

  // Last degeneracy analysis result (updated each LiDAR callback)
  da_ieskf::DegeneracyResult last_degeneracy;

  // Eigenvalue logger (nullptr = disabled)
  std::shared_ptr<da_ieskf::EigenvalueLogger> eigen_logger;
```

---

## 3. Add DA-IESKF fields to the SLAM struct (or equivalent manager class)

**Find** the class that owns the State and manages the IESKF loop.
Add these fields:

```cpp
  // ── Loop closure infrastructure ─────────────────────────────────────────
  da_ieskf::ScanContextManager sc_manager_;
  da_ieskf::PoseGraph          pose_graph_;

  // Keyframe selection
  Eigen::Matrix4d last_keyframe_pose_;
  int             keyframe_count_ = 0;

  // Parameters (loaded from YAML)
  double kf_min_dist_  = 0.5;
  double kf_min_rot_   = 0.1;
  bool   da_ieskf_enabled_ = true;
  bool   loop_closure_enabled_ = true;
```

---

## 4. Modify the IESKF update step to call degeneracy detection + inflation

**Find** the section where the Kalman gain is computed (search for `K =` or
`gain`). It will look roughly like:

```cpp
// Original:
MatXX K = P_ * H.transpose() * S.inverse();
P_ = (I - K * H) * P_;
```

**Replace with** (insert degeneracy check before the update):

```cpp
// DA-IESKF: analyse degeneracy BEFORE computing gain
if (degeneracy_detector && da_ieskf_enabled_) {
    // W = V^{-1} (per-point weights)
    Eigen::MatrixXd W = V.inverse();  // or build diagonal from lidar_sigma²
    last_degeneracy = degeneracy_detector->analyse(H, W);

    if (last_degeneracy.is_degenerate) {
        degeneracy_detector->inflate(P_, last_degeneracy);
    }

    if (eigen_logger) {
        eigen_logger->log(current_time_, last_degeneracy);
    }
}

// Original gain computation (unchanged):
MatXX K = P_ * H.transpose() * S.inverse();
P_ = (I - K * H) * P_;
```

**Note**: `S = H * P_ * H^T + V` must be recomputed after inflation if you
inflate P_ before computing S. The correct order is:

```cpp
// 1. Inflate P (DA-IESKF)
if (da_ieskf_enabled_ && last_degeneracy.is_degenerate)
    degeneracy_detector->inflate(P_, last_degeneracy);

// 2. Compute innovation covariance with inflated P
MatXX S = H * P_ * H.transpose() + V;

// 3. Compute gain
MatXX K = P_ * H.transpose() * S.inverse();

// 4. Update (Joseph form for numerical stability)
MatXX I_KH = I - K * H;
P_ = I_KH * P_ * I_KH.transpose() + K * V * K.transpose();
```

---

## 5. Add keyframe management to the LiDAR callback

**Find** the LiDAR callback (search for `lidarCallback` or `scan_callback`).
**After** the IESKF update, add:

```cpp
// ── Keyframe selection ────────────────────────────────────────────────────
Eigen::Matrix4d T_now = state_.X.matrix();  // current SE(3)/SGal(3) pose
bool add_kf = false;

if (keyframe_count_ == 0) {
    add_kf = true;
} else {
    Eigen::Matrix4d T_rel = last_keyframe_pose_.inverse() * T_now;
    double dt = T_rel.block<3,1>(0,3).norm();
    double dr = Eigen::AngleAxisd(
        Eigen::Matrix3d(T_rel.block<3,3>(0,0))).angle();
    add_kf = (dt > kf_min_dist_) || (dr > kf_min_rot_);
}

if (add_kf) {
    last_keyframe_pose_ = T_now;
    gtsam::Pose3 gtsam_pose(
        gtsam::Rot3(T_now.block<3,3>(0,0)),
        gtsam::Point3(T_now.block<3,1>(0,3)));

    // Extract 6×6 pose covariance block from full state covariance
    Eigen::Matrix<double, 6, 6> P_pose = state_.P.block<6,6>(0,0);

    pose_graph_.addKeyframe(current_time_, gtsam_pose, P_pose);

    // Add to Scan Context manager
    sc_manager_.addScan(keyframe_count_, current_time_, current_cloud_);
    keyframe_count_++;

    // ── Loop closure check ──────────────────────────────────────────────
    if (loop_closure_enabled_) {
        auto candidates = sc_manager_.detectLoops();
        for (const auto& cand : candidates) {
            auto gicp_res = sc_manager_.align(cand);
            if (sc_manager_.accept(gicp_res)) {
                // Convert Eigen 4×4 to GTSAM Pose3
                Eigen::Matrix4d T_rel_d = gicp_res.T_rel.cast<double>();
                gtsam::Pose3 T_lc(
                    gtsam::Rot3(T_rel_d.block<3,3>(0,0)),
                    gtsam::Point3(T_rel_d.block<3,1>(0,3)));
                pose_graph_.addLoopClosure(
                    cand.match_id, cand.query_id, T_lc, gicp_res.covariance);

                // Optimize and apply corrections
                if (pose_graph_.optimize()) {
                    auto corrections = pose_graph_.getCorrectionTransforms();
                    applyMapCorrections(corrections);  // see note below
                }
            }
        }
    }
}
```

---

## 6. Implement applyMapCorrections (partial i-Octree rebuild)

**Add** this private method to the SLAM class:

```cpp
void applyMapCorrections(const std::vector<gtsam::Pose3>& corrections) {
    // Only correct keyframes within the trailing window to bound compute time.
    int n = static_cast<int>(corrections.size());
    int start = std::max(0, n - correction_window_);

    for (int i = start; i < n; ++i) {
        const gtsam::Pose3& T_corr = corrections[i];
        if (T_corr.translation().norm() < 0.001 &&
            T_corr.rotation().toQuaternion().angularDistance(
                gtsam::Quaternion::Identity()) < 0.001)
            continue;  // no meaningful correction

        // TODO: transform the points stored at keyframe i in the map
        // by T_corr.  Specific API depends on the i-Octree implementation.
        // For LIMOncello iVox: call map_.transform(i, T_corr.matrix())
        //
        // map_.transformKeyframePoints(i, T_corr.matrix().cast<float>());
    }
}
```

---

## 7. Initialization in the constructor or onInit()

```cpp
// Load YAML params
std::string cfg_path = ament_index_cpp::get_package_share_directory(
    "limoncello") + "/config/da_ieskf_params.yaml";
YAML::Node cfg = YAML::LoadFile(cfg_path)["da_ieskf"];

double eigen_thresh  = cfg["degeneracy"]["eigenvalue_threshold"].as<double>(0.15);
double infl_alpha    = cfg["degeneracy"]["inflation_alpha"].as<double>(100.0);
da_ieskf_enabled_    = cfg["degeneracy"]["enabled"].as<bool>(true);
loop_closure_enabled_ = cfg["loop_closure"]["enabled"].as<bool>(true);
kf_min_dist_         = cfg["keyframe"]["min_dist_m"].as<double>(0.5);
kf_min_rot_          = cfg["keyframe"]["min_rot_rad"].as<double>(0.1);

// State dimension: SGal(3) has 10-dim error state
degeneracy_detector_ = std::make_shared<da_ieskf::DegeneracyDetector>(
    eigen_thresh, infl_alpha,
    /*state_dim=*/10, /*pose_dim=*/3);

if (cfg["degeneracy"]["log_eigenvalues"].as<bool>(false)) {
    std::string log_path = cfg["degeneracy"]["log_path"].as<std::string>("/tmp/eigenvalues.csv");
    eigen_logger_ = std::make_shared<da_ieskf::EigenvalueLogger>(log_path);
}
```

---

## 8. CMakeLists.txt additions

In `~/ros2_ws_limo/src/LIMOncello/CMakeLists.txt`, find the
`find_package` and `target_link_libraries` sections:

```cmake
# Add GTSAM dependency
find_package(GTSAM REQUIRED)
find_package(yaml-cpp REQUIRED)

# In target_link_libraries for the main executable/library:
target_link_libraries(limoncello
    # ... existing libs ...
    gtsam
    yaml-cpp
)

# Include path for our new headers
target_include_directories(limoncello PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
```

---

## Summary of files touched

| File | Change |
|------|--------|
| `include/State.hpp` | Add DA-IESKF fields, degeneracy inflation in update step |
| `include/DegeneracyDetector.hpp` | **New file** (copy from this repo) |
| `include/PoseGraph.hpp` | **New file** (copy from this repo) |
| `include/ScanContextManager.hpp` | **New file** (copy from this repo) |
| `CMakeLists.txt` | Add GTSAM, yaml-cpp dependencies |
| `config/limoncello_params.yaml` | Add `da_ieskf` block (copy from this repo) |
