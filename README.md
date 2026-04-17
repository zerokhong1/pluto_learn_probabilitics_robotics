# Pluto Robot — Probabilistic Robotics

Học **Probabilistic Robotics** (Thrun, Burgard, Fox) qua robot Pluto chạy trên ROS 2 Jazzy + Gazebo Harmonic.

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
# Clone repo
git clone https://github.com/zerokhong1/pluto_learn_probabilitics_robotics.git
cd pluto_learn_probabilitics_robotics

# Cài dependencies
rosdep install --from-paths src --ignore-src -r -y

# Build
colcon build --symlink-install
source install/setup.bash
```

---

## Demo 1 — Hallway (Không cần GPU, chạy nhanh nhất)

Demo **Discrete Bayes Filter** trong hành lang 1D. Robot Pluto tự định vị bằng cảm biến cửa.

```bash
source install/setup.bash
ros2 launch pluto_gazebo hallway.launch.py
```

**Kết quả:** RViz2 mở ra, thấy robot di chuyển trong hành lang và phân phối xác suất vị trí cập nhật theo thời gian thực.

---

## Demo 2 — Standalone (Simulation thuần Python, không cần Gazebo)

Demo đầy đủ các bộ lọc (Bayes, Kalman, Particle Filter) với simulator nội bộ.

```bash
source install/setup.bash
ros2 launch pluto_gazebo standalone_demo.launch.py
```

**Kết quả:** RViz2 hiển thị robot Pluto với mắt và bảng ngực phản ánh trạng thái uncertainty của bộ lọc.

---

## Demo 3 — Gazebo Full (Cần NVIDIA GPU)

Demo hoàn chỉnh với môi trường 3D Gazebo, LiDAR, odometry thực.

```bash
source install/setup.bash
ros2 launch pluto_gazebo gazebo_demo.launch.py
```

> **Lưu ý:** Nếu Gazebo bị treo hoặc crash, kiểm tra driver NVIDIA và biến môi trường EGL đã được set đúng trong launch file.

---

## Notebooks — Học lý thuyết từng chương

Cài Jupyter trước:

```bash
pip install jupyter matplotlib numpy scipy
```

Chạy Jupyter:

```bash
cd notebooks
jupyter notebook
```

| Notebook | Nội dung |
|---|---|
| [ch01_introduction.ipynb](notebooks/ch01_introduction.ipynb) | Giới thiệu uncertainty, random walk |
| [ch02_bayes_filter.ipynb](notebooks/ch02_bayes_filter.ipynb) | Discrete Bayes Filter — định vị trong hành lang |
| [ch03_gaussian_filters.ipynb](notebooks/ch03_gaussian_filters.ipynb) | Kalman Filter, Extended KF, Information Filter |
| [ch04_nonparametric.ipynb](notebooks/ch04_nonparametric.ipynb) | Particle Filter, Monte Carlo Localization |
| [ch05_motion_models.ipynb](notebooks/ch05_motion_models.ipynb) | Velocity model, Odometry model |
| [ch06_measurement_models.ipynb](notebooks/ch06_measurement_models.ipynb) | Beam model, Likelihood field model |

**Thứ tự học:** ch01 → ch02 → ch03 → ch04 → ch05 → ch06

---

## Cấu trúc project

```
pluto_robot/
├── notebooks/          # Jupyter notebooks từng chương
├── src/
│   ├── pluto_description/   # URDF/xacro mô tả robot
│   ├── pluto_gazebo/        # Launch files + world SDF
│   ├── pluto_filters/       # Cài đặt các bộ lọc (Bayes, KF, PF)
│   ├── pluto_experiments/   # Script thí nghiệm
│   └── pluto_visualization/ # RViz2 markers, mắt, bảng ngực
```

---

## Luồng hoạt động

```
Gazebo Sim
    │  /scan (LaserScan)
    │  /odom (Odometry)
    ▼
ROS-Gz Bridge
    │
    ├──▶ Bayes Filter Node  ──▶ /belief (MarkerArray) ──▶ RViz2
    ├──▶ Kalman Filter Node ──▶ /pose_estimate         ──▶ RViz2
    └──▶ Particle Filter    ──▶ /particles             ──▶ RViz2
```

---

## Troubleshooting

**Gazebo không mở / crash ngay khi khởi động:**
```bash
# Kiểm tra EGL NVIDIA
ls /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Chạy thủ công để xem lỗi
gz sim -r src/pluto_gazebo/worlds/pluto_hallway.sdf
```

**RViz2 không hiện robot:**
```bash
# Kiểm tra robot_description có được publish chưa
ros2 topic echo /robot_description --once
```

**Không thấy /scan hoặc /odom:**
```bash
# Kiểm tra bridge đang chạy
ros2 node list | grep bridge
ros2 topic list | grep -E "scan|odom"
```

---

## Tham khảo

- *Probabilistic Robotics* — Thrun, Burgard, Fox (2005)
- [ROS 2 Jazzy Docs](https://docs.ros.org/en/jazzy/)
- [Gazebo Harmonic Docs](https://gazebosim.org/docs/harmonic/)
