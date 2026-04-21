# Pluto Robot — Probabilistic Robotics

Học **Probabilistic Robotics** (Thrun, Burgard, Fox) qua robot Pluto chạy trên ROS 2 Jazzy + Gazebo Harmonic.

Mỗi chương trong sách được cài đặt từ đầu — không dùng AMCL, nav2 hay bất kỳ black-box có sẵn nào.

---

## Yêu cầu hệ thống

| Thành phần | Phiên bản |
|---|---|
| Ubuntu | 24.04 |
| ROS 2 | Jazzy |
| Gazebo | Harmonic |
| Python | 3.10+ |
| GPU | NVIDIA (khuyến nghị) |

---

## Cài đặt

```bash
git clone https://github.com/zerokhong1/pluto_learn_probabilitics_robotics.git
cd pluto_learn_probabilitics_robotics

rosdep install --from-paths src --ignore-src -r -y
pip install scipy matplotlib numpy

colcon build --symlink-install
source install/setup.bash
```

---

## Demo ROS 2

### Demo 1 — Hallway (Discrete Bayes Filter, không cần GPU)

Robot tự định vị trong hành lang 1D bằng cảm biến cửa.

```bash
source install/setup.bash
ros2 launch pluto_gazebo hallway.launch.py
```

### Demo 2 — Standalone (Simulation thuần Python)

Tất cả bộ lọc (Bayes, Kalman, Particle Filter) chạy mà không cần Gazebo.

```bash
source install/setup.bash
ros2 launch pluto_gazebo standalone_demo.launch.py
```

### Demo 3 — Gazebo Full (Cần NVIDIA GPU)

LiDAR + IMU thật từ Gazebo Harmonic.

```bash
source install/setup.bash
ros2 launch pluto_gazebo gazebo_demo.launch.py
```

### Demo 4 — IESKF LIO (ch05) — 2D LiDAR-Inertial Odometry

IESKF trên SE(2) manifold, port từ LIMOncello (arXiv:2512.19567).

```bash
source install/setup.bash
ros2 run pluto_filters lio_2d_node
```

Publishes `/odom_ieskf` và TF `odom → base_link`.

---

## Notebooks — Học lý thuyết từng chương

```bash
pip install jupyter matplotlib numpy scipy
cd notebooks && jupyter notebook
```

| Notebook | Nội dung |
|---|---|
| [ch01_introduction.ipynb](notebooks/ch01_introduction.ipynb) | Uncertainty, dead-reckoning, random walk — sai số tăng như $\sqrt{t}$ |
| [ch02_bayes_filter.ipynb](notebooks/ch02_bayes_filter.ipynb) | Discrete Bayes Filter — định vị trong hành lang |
| [ch03_gaussian_filters.ipynb](notebooks/ch03_gaussian_filters.ipynb) | Kalman Filter, Extended KF, UKF, Information Filter |
| [ch04_nonparametric.ipynb](notebooks/ch04_nonparametric.ipynb) | Particle Filter, Monte Carlo Localization |
| [ch05_motion_models.ipynb](notebooks/ch05_motion_models.ipynb) | Velocity model, Odometry model, banana distribution |
| [ch06_measurement_models.ipynb](notebooks/ch06_measurement_models.ipynb) | Beam model, Likelihood field, EM fitting |

**Thứ tự học:** ch01 → ch02 → ch03 → ch04 → ch05 → ch06

---

## Experiments — Chạy standalone (không cần ROS 2)

### IESKF vs EKF — Hallway

```bash
cd src/pluto_experiments/pluto_experiments/ieskf_showdown
python3 hallway_comparison.py
```

| Filter | RMSE |
|---|---|
| IESKF (SE(2) manifold) | **3.81 m** |
| EKF (Euclidean) | 10650 m (**diverges**) |

EKF tích lũy lỗi góc không bị chuẩn hóa → diverge. IESKF dùng `Log/Exp` trên SE(2) → góc luôn đúng.

### IESKF vs EKF — Degenerate Corridor (analog City02 tunnel)

```bash
python3 degenerate_corridor.py
```

Hành lang 50 m thẳng, tường trơn không features (0–40 m). Chỉ hướng vuông góc với tường là observable; hướng dọc hành lang là **weakly observable**.

| Filter | Along-corridor RMSE | Total APE RMSE |
|---|---|---|
| IESKF | **27.84 m** | **27.84 m** |
| EKF | 29.43 m | 29.90 m |

IESKF drift ít hơn ~6% so với EKF trong điều kiện suy biến — analog kết quả City02 tunnel của paper LIMOncello.

---

## Cấu trúc project

```
src/
├── pluto_description/          # URDF/xacro robot
├── pluto_gazebo/
│   ├── launch/                 # hallway / standalone / gazebo_demo
│   └── worlds/
│       ├── pluto_hallway.sdf
│       └── degenerate_corridor.sdf   # 50m featureless tunnel
├── pluto_filters/
│   ├── bayes_filter/           # ch02 — Discrete Bayes
│   ├── kalman_filters/         # ch03 — KF / EKF / UKF / IF
│   ├── particle_filters/       # ch04 — Particle Filter MCL
│   ├── motion_models/          # ch05 — velocity + odometry
│   ├── measurement_models/     # ch06 — beam + likelihood field
│   └── ieskf_lio/              # ch05+ — IESKF on SE(2) (LIMOncello port)
│       ├── se2_manifold.py     # Exp, Log, ⊕, ⊖, Ad, Jr, Jr_inv
│       ├── ieskf.py            # Iterated Error-State Kalman Filter
│       ├── scan_matcher.py     # Point-to-line residuals + KDTree map
│       └── lio_2d.py           # ROS 2 node: IMU predict + LiDAR update
├── pluto_experiments/
│   ├── banana_distribution/
│   ├── filter_showdown/
│   ├── mcl_capstone/
│   └── ieskf_showdown/         # hallway_comparison + degenerate_corridor
└── pluto_visualization/        # eye_state, belief_display, particle_ghost
```

---

## Luồng hoạt động

```
Gazebo Sim / Python Simulator
    │  /scan (LaserScan)  /imu (Imu)  /odom (Odometry)
    ▼
ROS-Gz Bridge
    │
    ├──▶ Bayes Filter    ──▶ /belief           ──▶ RViz2
    ├──▶ EKF / UKF       ──▶ /pose_estimate    ──▶ RViz2
    ├──▶ Particle Filter ──▶ /particles        ──▶ RViz2
    └──▶ LIO 2D (IESKF)  ──▶ /odom_ieskf      ──▶ RViz2
                                TF: odom → base_link
```

---

## Kiến trúc IESKF (ch05+)

Port thuật toán **LIMOncello** (Perez-Ruiz & Solà, arXiv:2512.19567) từ SGal(3)/3D xuống SE(2)/2D.

| Paper (3D) | Pluto (2D) |
|---|---|
| SGal(3) state | SE(2) + velocity + biases |
| Iterated EKF update | Giữ nguyên — generic |
| Point-to-plane residual | Point-to-line (2D) |
| i-Octree map | scipy KDTree |
| IMU @ 200–400 Hz | Sim IMU @ 100 Hz |

Phép toán manifold cốt lõi (20 unit tests pass):

```python
Exp(τ)        # R³ → SE(2)  (closed-form, handle small θ)
Log(X)        # SE(2) → R³
X ⊕ τ         # right-plus:  X @ Exp(τ)
Y ⊖ X         # right-minus: Log(X⁻¹ @ Y)
Jr(τ)         # right Jacobian (3×3, derived từ series Σ (-1)ⁿ/(n+1)! adⁿ)
Jr_inv(τ)     # inverse Jacobian — dùng trong iterated update
```

---

## Troubleshooting

**Gazebo crash / không mở:**
```bash
ls /usr/share/glvnd/egl_vendor.d/10_nvidia.json   # kiểm tra NVIDIA EGL
gz sim -r src/pluto_gazebo/worlds/pluto_hallway.sdf
```

**RViz2 không hiện robot:**
```bash
ros2 topic echo /robot_description --once
```

**`lio_2d_node` không nhận scan:**
```bash
ros2 topic list | grep -E "scan|imu|odom_ieskf"
ros2 topic hz /scan
```

**Rebuild sau khi thay đổi code:**
```bash
colcon build --symlink-install --packages-select pluto_filters
source install/setup.bash
```

---

## Tham khảo

- *Probabilistic Robotics* — Thrun, Burgard, Fox (2005)
- [LIMOncello paper](https://arxiv.org/abs/2512.19567) — Perez-Ruiz & Solà (2025)
- [A micro Lie theory](https://arxiv.org/abs/1812.01537) — Solà et al.
- [ROS 2 Jazzy Docs](https://docs.ros.org/en/jazzy/)
- [Gazebo Harmonic Docs](https://gazebosim.org/docs/harmonic/)
