"""
2D LiDAR-Inertial Odometry using IESKF on SE(2).

Tightly-coupled pipeline (paper Fig. 2 in 2D):
  IMU @ 100 Hz  → forward propagation (predict)
  LiDAR @ 10 Hz → iterated update with point-to-line scan matching

State: x = (T ∈ SE(2), v ∈ R, b_gyro ∈ R, b_accel ∈ R)
  T        — 2D pose (x, y, θ)        [SE(2) manifold]
  v        — forward speed             [R, in bias vector slot 0 is repurposed]
  b_gyro   — gyroscope bias (yaw rate) [R]
  b_accel  — accelerometer bias (fwd)  [R]

Error-state dimension: 3 (SE(2)) + 3 (v, b_gyro, b_accel) = 6

Subscribes:
    /imu   (sensor_msgs/Imu)      — 100 Hz
    /scan  (sensor_msgs/LaserScan) — 10 Hz

Publishes:
    /odom_ieskf (nav_msgs/Odometry)
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

from .se2_manifold import Exp, oplus, ominus, Jr_inv
from .ieskf import IESKF
from .scan_matcher import ScanMatcher


def _quat_from_yaw(yaw: float):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class LIO2D(Node):
    """2D LiDAR-Inertial Odometry node."""

    # error-state layout: [δpose(3), δv(1), δb_gyro(1), δb_accel(1)]
    _ERR_DIM = 6

    def __init__(self):
        super().__init__('lio_2d')

        # ── state (nominal) ────────────────────────────────────────────────
        self.pose   = np.eye(3)    # SE(2) matrix
        self.vel    = 0.0          # forward speed [m/s]
        self.b_gyro = 0.0          # gyro bias [rad/s]
        self.b_accel = 0.0         # accel bias [m/s²]

        # ── IESKF (bias_dim=3: v, b_gyro, b_accel) ────────────────────────
        self.kf = IESKF(bias_dim=3)
        self.kf.pose = self.pose
        self.kf.bias = np.array([self.vel, self.b_gyro, self.b_accel])
        self.kf.P    = np.diag([0.01, 0.01, 0.001,   # pose
                                0.1,                  # vel
                                0.01, 0.01])          # biases

        # ── scan matcher ───────────────────────────────────────────────────
        self.matcher = ScanMatcher(k_neighbors=5, max_dist=2.0)

        # ── noise parameters ───────────────────────────────────────────────
        # Process noise: [n_gyro, n_accel, n_b_gyro_rw, n_b_accel_rw]
        self._sigma_gyro   = 0.005   # rad/s / √Hz
        self._sigma_accel  = 0.05    # m/s²  / √Hz
        self._sigma_bg_rw  = 1e-4    # rad/s / √s  (random walk)
        self._sigma_ba_rw  = 1e-3    # m/s²  / √s

        self._sigma_scan   = 0.02    # m (per point measurement noise)
        self._downsample_res = 0.10  # m

        # ── IESKF tuning ───────────────────────────────────────────────────
        self._max_iter = 5
        self._eps      = 1e-4

        # ── timing ─────────────────────────────────────────────────────────
        self._last_imu_stamp: float | None = None

        # ── ROS 2 ──────────────────────────────────────────────────────────
        self.create_subscription(Imu,       '/imu',  self._imu_cb,   10)
        self.create_subscription(LaserScan, '/scan', self._lidar_cb, 10)
        self._odom_pub = self.create_publisher(Odometry, '/odom_ieskf', 10)
        self._tf_br   = TransformBroadcaster(self)

        self.get_logger().info('LIO2D node started.')

    # ── motion model helpers ───────────────────────────────────────────────

    def _f_nominal(self, pose, bias, dt: float):
        """Propagate nominal state. Option A: SE(2) + scalar velocity."""
        v, b_gyro, b_accel = bias

        # Correct IMU readings
        omega = self._last_gyro_z - b_gyro
        ax    = self._last_accel_x - b_accel

        # Velocity update
        v_new = v + ax * dt

        # Pose update via SE(2) Exp
        # In body frame: forward motion = v*dt, rotation = omega*dt
        v_mean = (v + v_new) / 2.0
        tau = np.array([v_mean * dt, 0.0, omega * dt])
        pose_new = oplus(pose, tau)

        bias_new = np.array([v_new, b_gyro, b_accel])
        return pose_new, bias_new

    def _build_F_dx(self, dt: float) -> np.ndarray:
        """Error-state transition Jacobian F = ∂f/∂δx (6×6).

        Linearized around nominal state. Simplified form:
          δpose_{k+1} ≈ Ad(Exp(-ω dt)) δpose_k + B_pose (δv, δb_gyro, δb_accel)
          δv_{k+1}    ≈ δv_k + (−1) δb_accel dt
          δb_{k+1}    ≈ δb_k   (random-walk biases)
        """
        F = np.eye(6)

        omega_dt = (self._last_gyro_z - self.kf.bias[1]) * dt
        c, s = math.cos(omega_dt), math.sin(omega_dt)

        # SE(2) pose block: rotation by −ω dt (adjoint)
        F[0, 0] =  c;  F[0, 1] = s
        F[1, 0] = -s;  F[1, 1] = c

        v_mean = self.kf.bias[0] + 0.5 * (self._last_accel_x - self.kf.bias[2]) * dt

        # ∂δpose / ∂δv
        F[0, 3] = c * dt
        F[1, 3] = s * dt

        # ∂δpose / ∂δb_gyro (heading couples to translation)
        F[0, 4] = -v_mean * s * dt
        F[1, 4] =  v_mean * c * dt
        F[2, 4] = -dt

        # ∂δv / ∂δb_accel
        F[3, 5] = -dt

        return F

    def _build_F_w(self, dt: float) -> np.ndarray:
        """Noise input Jacobian F_w (6×4).

        Noise vector w = [n_gyro, n_accel, n_bg_rw, n_ba_rw].
        """
        v_mean = self.kf.bias[0]
        F_w = np.zeros((6, 4))

        # pose ← gyro noise
        F_w[0, 0] = -v_mean * math.sin(0.0) * dt   # ≈ 0 at small dt
        F_w[2, 0] = -dt
        # pose ← accel noise
        F_w[0, 1] = dt
        # vel ← accel noise
        F_w[3, 1] = dt
        # bias random walks
        F_w[4, 2] = 1.0
        F_w[5, 3] = 1.0

        return F_w

    def _build_Q(self, dt: float) -> np.ndarray:
        """Process noise covariance (4×4)."""
        sg = self._sigma_gyro  * math.sqrt(dt)
        sa = self._sigma_accel * math.sqrt(dt)
        return np.diag([sg**2, sa**2,
                        (self._sigma_bg_rw * math.sqrt(dt))**2,
                        (self._sigma_ba_rw * math.sqrt(dt))**2])

    # ── IMU callback ───────────────────────────────────────────────────────

    def _imu_cb(self, msg: Imu):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Cache latest IMU readings for F_dx / f_nominal
        self._last_gyro_z  = msg.angular_velocity.z
        self._last_accel_x = msg.linear_acceleration.x

        if self._last_imu_stamp is None:
            self._last_imu_stamp = stamp
            return

        dt = stamp - self._last_imu_stamp
        self._last_imu_stamp = stamp

        if dt <= 0.0 or dt > 0.5:
            return

        F_dx = self._build_F_dx(dt)
        F_w  = self._build_F_w(dt)
        Q    = self._build_Q(dt)

        self.kf.predict(self._f_nominal, F_dx, F_w, Q, dt)

    # ── LiDAR callback ─────────────────────────────────────────────────────

    def _lidar_cb(self, msg: LaserScan):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # 1. Convert to robot-frame Cartesian
        n = len(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, n)
        ranges = np.array(msg.ranges, dtype=float)
        scan = ScanMatcher.polar_to_cart(
            ranges, angles, msg.range_min, msg.range_max)

        if len(scan) < 5:
            return

        # 2. Downsample
        scan_ds = ScanMatcher.voxel_downsample(scan, self._downsample_res)

        # 3. First frame: seed map
        if self.matcher.map_points is None:
            world_pts = ScanMatcher.transform_points(scan_ds, self.kf.pose)
            self.matcher.set_map(world_pts)
            self._publish_odom(msg.header.stamp)
            return

        # 4. Build closures for IESKF update
        matcher = self.matcher
        sigma   = self._sigma_scan

        def z_func(pose_j, bias_j):
            z, _, ok = matcher.compute_residuals_and_jacobians(scan_ds, pose_j)
            return z if ok else np.array([])

        def H_func(pose_j, bias_j):
            _, H_se2, ok = matcher.compute_residuals_and_jacobians(scan_ds, pose_j)
            if not ok:
                return np.zeros((0, self._ERR_DIM))
            m = H_se2.shape[0]
            H_full = np.zeros((m, self._ERR_DIM))
            H_full[:, :3] = H_se2   # pose columns; vel+bias columns = 0
            return H_full

        # Measurement noise: estimated per update (needs a probe call)
        z_probe = z_func(self.kf.pose, self.kf.bias)
        if z_probe.size == 0:
            return
        V = np.eye(len(z_probe)) * sigma**2

        # 5. IESKF iterated update
        self.kf.update(z_func, H_func, V, self._max_iter, self._eps)

        # 6. Add frame to map (world frame)
        world_pts = ScanMatcher.transform_points(scan_ds, self.kf.pose)
        self.matcher.add_to_map(world_pts)

        # 7. Publish
        self._publish_odom(msg.header.stamp)

    # ── publishing ─────────────────────────────────────────────────────────

    def _publish_odom(self, stamp):
        pose = self.kf.pose
        x, y = pose[0, 2], pose[1, 2]
        yaw  = math.atan2(pose[1, 0], pose[0, 0])
        qx, qy, qz, qw = _quat_from_yaw(yaw)

        msg = Odometry()
        msg.header.stamp    = stamp
        msg.header.frame_id = 'odom'
        msg.child_frame_id  = 'base_link'

        msg.pose.pose.position.x    = x
        msg.pose.pose.position.y    = y
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        vel = self.kf.bias[0]
        msg.twist.twist.linear.x  = vel
        msg.twist.twist.angular.z = self._last_gyro_z - self.kf.bias[1] \
            if hasattr(self, '_last_gyro_z') else 0.0

        self._odom_pub.publish(msg)

        # TF: odom → base_link
        tf = TransformStamped()
        tf.header         = msg.header
        tf.child_frame_id = 'base_link'
        tf.transform.translation.x = x
        tf.transform.translation.y = y
        tf.transform.rotation.z    = qz
        tf.transform.rotation.w    = qw
        self._tf_br.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = LIO2D()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
