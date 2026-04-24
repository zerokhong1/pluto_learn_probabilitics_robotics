# Pluto Robot — Probabilistic Robotics & DA-IESKF Research

Học **Probabilistic Robotics** (Thrun, Burgard, Fox) từ lý thuyết đến triển khai thực tế,
kết hợp với nghiên cứu **Degeneracy-Aware IESKF** tích hợp vào
[LIMOncello](https://github.com/fetty31/LIMOncello) LiDAR-Inertial SLAM.

---

## DA-IESKF — Kết quả thực tế

> **Cải thiện 11.8 % APE RMSE** trên NTU VIRAL `eee_03` (indoor academic building, 181 s)

| Method | APE RMSE | APE Max |
|---|---|---|
| Baseline IESKF | 0.281 m | 0.658 m |
| **DA-IESKF** | **0.248 m** | **0.597 m** |

Sensor: Ouster OS1-16 @ 10 Hz · VN-100 IMU @ 385 Hz · 12.9 % frames flagged degenerate

**Idea cốt lõi:** Khi LiDAR scan thiếu constraint theo một hướng (corridor, tunnel),
IESKF chuẩn đưa ra covariance quá tự tin gây drift. DA-IESKF phân tích eigenvalue
của translation information matrix `Σ nᵢnᵢᵀ/R`, phát hiện hướng degenerate và
inflate P dọc theo đó → filter tự động dựa vào IMU nhiều hơn ở những hướng đó.

Chi tiết → [da_ieskf_limoncello/README.md](da_ieskf_limoncello/README.md)

---

## Cấu trúc repository

```
pluto_robot/
├── da_ieskf_limoncello/         # DA-IESKF research (C++ / ROS 2)
│   ├── include/
│   │   ├── DegeneracyDetector.hpp    # Core algorithm — header-only
│   │   ├── PoseGraph.hpp             # Loop closure graph (GTSAM, Phase 5)
│   │   └── ScanContextManager.hpp    # Scan Context descriptor (Phase 5)
│   ├── phase3_smoke_test/            # Synthetic bag validation
│   ├── scripts/                      # run_phase4_eee03.sh, evaluate_all.sh …
│   ├── results/                      # Numerical results
│   └── README.md                     # Integration guide + algorithm
│
├── src/                         # Probabilistic Robotics learning (ROS 2)
│   ├── pluto_description/            # URDF/xacro robot model
│   ├── pluto_gazebo/                 # Launch files + Gazebo worlds
│   ├── pluto_filters/                # ch02–ch06 filter implementations
│   ├── pluto_experiments/            # Standalone experiment scripts
│   └── pluto_visualization/          # RViz2 display nodes
│
└── notebooks/                   # Theory notebooks (Jupyter)
    ├── ch01_introduction.ipynb
    ├── ch02_bayes_filter.ipynb
    ├── ch03_gaussian_filters.ipynb
    ├── ch04_nonparametric.ipynb
    ├── ch05_motion_models.ipynb
    └── ch06_measurement_models.ipynb
```

---

## DA-IESKF — Nhanh

### Cài đặt vào LIMOncello

```bash
# 1. Copy header
cp da_ieskf_limoncello/include/DegeneracyDetector.hpp \
   ~/LIMOncello_ws/src/LIMOncello/include/

# 2. Patch Config.hpp, ROSutils.hpp, State.hpp
#    → xem hướng dẫn đầy đủ trong da_ieskf_limoncello/README.md

# 3. Thêm vào YAML config
# IKFoM:
#   da_ieskf_enabled: true
#   da_eigenvalue_threshold: 0.15
#   da_inflation_alpha: 100.0
#   da_eigenvalue_log: "/tmp/eigenvalues.csv"

# 4. Build
cd ~/LIMOncello_ws
colcon build --packages-select limoncello --cmake-args -DCMAKE_BUILD_TYPE=Release
```

### Chạy Phase 4 (NTU VIRAL eee_03)

```bash
source /opt/ros/jazzy/setup.bash
source ~/LIMOncello_ws/install/setup.bash
bash da_ieskf_limoncello/scripts/run_phase4_eee03.sh
```

### Smoke test tổng hợp (Phase 3)

```bash
# Tạo bag tổng hợp rồi so sánh box room vs corridor
python3 ~/LIMOncello_ws/test/generate_synthetic_bag.py
bash ~/LIMOncello_ws/test/run_smoke_test.sh both
# Kỳ vọng: 4/4 PASS — corridor 99.7% degen vs box 0%
```

---

## DA-IESKF — Tiến độ

| Phase | Mô tả | Trạng thái |
|---|---|---|
| 0–1 | Lý thuyết, Python simulation, gap analysis | ✅ Xong |
| 2 | Tích hợp C++ vào LIMOncello | ✅ Xong |
| 3 | Smoke test tổng hợp (4/4 pass) | ✅ Xong |
| 4 | Real data — NTU VIRAL eee_03 | ✅ Xong (−11.8 %) |
| 5 | Loop closure (PoseGraph + ScanContext + GTSAM) | ⏳ Pending |
| 6 | Multi-sequence evaluation | ⏳ Pending |
| 7 | Viết paper | ⏳ Pending |

---

## Probabilistic Robotics — Học theo chương

### Yêu cầu

| Thành phần | Phiên bản |
|---|---|
| Ubuntu | 24.04 |
| ROS 2 | Jazzy |
| Gazebo | Harmonic |
| Python | 3.10+ |

### Cài đặt

```bash
git clone https://github.com/zerokhong1/pluto_learn_probabilitics_robotics.git
cd pluto_learn_probabilitics_robotics

rosdep install --from-paths src --ignore-src -r -y
pip install scipy matplotlib numpy

colcon build --symlink-install
source install/setup.bash
```

### Notebooks

```bash
pip install jupyter matplotlib numpy scipy
cd notebooks && jupyter notebook
```

| Notebook | Nội dung |
|---|---|
| ch01_introduction | Uncertainty, dead-reckoning, random walk |
| ch02_bayes_filter | Discrete Bayes Filter — định vị hành lang |
| ch03_gaussian_filters | KF, EKF, UKF, Information Filter |
| ch04_nonparametric | Particle Filter, Monte Carlo Localization |
| ch05_motion_models | Velocity model, Odometry model |
| ch06_measurement_models | Beam model, Likelihood field, EM fitting |

### Demo ROS 2

```bash
# Hallway (Discrete Bayes Filter)
ros2 launch pluto_gazebo hallway.launch.py

# Standalone simulation
ros2 launch pluto_gazebo standalone_demo.launch.py

# LiDAR-Inertial Odometry (IESKF trên SE(2))
ros2 run pluto_filters lio_2d_node
```

### Kết quả experiment nổi bật

**IESKF vs EKF — Hallway**

| Filter | RMSE |
|---|---|
| IESKF (SE(2) manifold) | **3.81 m** |
| EKF (Euclidean) | 10 650 m — diverges |

EKF không chuẩn hóa góc → tích lũy lỗi → diverge. IESKF dùng `Log/Exp` trên SE(2) → góc luôn đúng.

**IESKF vs EKF — Degenerate Corridor**

| Filter | Along-corridor RMSE |
|---|---|
| IESKF | 27.84 m |
| EKF | 29.43 m |

Hành lang 50 m thẳng, tường trơn — hướng dọc corridor là weakly observable.

---

## Luồng dữ liệu (ROS 2)

```
Gazebo / Python Simulator
    │  /scan · /imu · /odom
    ▼
    ├──▶ Bayes Filter     ──▶ /belief
    ├──▶ EKF / UKF        ──▶ /pose_estimate
    ├──▶ Particle Filter  ──▶ /particles
    └──▶ LIO 2D (IESKF)   ──▶ /odom_ieskf · TF: odom → base_link
```

---

## Tham khảo

- *Probabilistic Robotics* — Thrun, Burgard, Fox (2005)
- [LIMOncello — arXiv:2512.19567](https://arxiv.org/abs/2512.19567) — Perez-Ruiz & Solà (2025)
- [A micro Lie theory — arXiv:1812.01537](https://arxiv.org/abs/1812.01537) — Solà et al.
- [NTU VIRAL Dataset](https://ntu-viral.github.io/)
- [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/) · [Gazebo Harmonic](https://gazebosim.org/docs/harmonic/)
