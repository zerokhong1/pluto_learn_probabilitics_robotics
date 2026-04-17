"""
MCL Capstone — Chapter 6 Final Experiment.
Full Monte Carlo Localization using:
  - Odometry motion model (Ch5)
  - Likelihood field measurement model (Ch6)
  - Particle filter (Ch4)
Pluto starts at random position and must self-localize then navigate home.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import PoseStamped, Twist, Point
import json
import math

from pluto_filters.kalman_filters.ekf import normalize_angle as norm_angle


N_PARTICLES = 1000
HOME_POSITION = np.array([1.0, 1.0, 0.0])  # target "home" pose
GOAL_TOLERANCE = 0.15  # [m]


class Particle3D:
    __slots__ = ['x', 'y', 'theta', 'weight']

    def __init__(self, x=0.0, y=0.0, theta=0.0, w=1.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = w


class MCLLocalizationNode(Node):
    """
    Full MCL ROS2 node — capstone for 6 chapters.
    1. Subscribes to /odom for odometry-based motion model.
    2. Subscribes to /scan and /map for likelihood field measurement model.
    3. Publishes mean pose to /pluto/mcl_pose.
    4. Publishes particle cloud to /pluto/mcl_particles.
    5. When converged, publishes navigation goal to /goal_pose.
    """

    def __init__(self):
        super().__init__('mcl_localization')
        self._map: np.ndarray | None = None
        self._map_res = 0.05
        self._map_origin = (0.0, 0.0)
        self._lf_field: np.ndarray | None = None

        self._particles = []
        self._prev_odom = None
        self._converged = False

        self.pub_pose = self.create_publisher(PoseStamped, '/pluto/mcl_pose', 10)
        self.pub_particles = self.create_publisher(MarkerArray, '/pluto/mcl_particles', 10)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)

        self.create_subscription(OccupancyGrid, '/map', self._cb_map, 10)
        self.create_subscription(Odometry, '/odom', self._cb_odom, 10)
        self.create_subscription(LaserScan, '/scan', self._cb_scan, 10)

        self.create_timer(0.5, self._navigation_cb)
        self.get_logger().info('MCL Localization (capstone) ready. Waiting for map...')

    def _cb_map(self, msg: OccupancyGrid):
        w, h = msg.info.width, msg.info.height
        self._map = np.array(msg.data, dtype=np.int8).reshape((h, w))
        self._map_res = msg.info.resolution
        self._map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        self._build_likelihood_field()
        self._initialize_particles_uniform(w, h)
        self.get_logger().info(f'Map received ({w}×{h}), particles initialized.')

    def _build_likelihood_field(self):
        from scipy.ndimage import distance_transform_edt
        obstacle_mask = self._map > 50
        dist = distance_transform_edt(~obstacle_mask) * self._map_res
        sigma = 0.2
        self._lf_field = 0.9 * np.exp(-0.5 * (dist / sigma)**2) + 0.1 / 12.0

    def _initialize_particles_uniform(self, w: int, h: int):
        ox, oy = self._map_origin
        self._particles = []
        count = 0
        attempts = 0
        while count < N_PARTICLES and attempts < N_PARTICLES * 10:
            cx = np.random.randint(0, w)
            cy = np.random.randint(0, h)
            if self._map[cy, cx] < 50:  # free cell
                wx = ox + (cx + 0.5) * self._map_res
                wy = oy + (cy + 0.5) * self._map_res
                theta = np.random.uniform(-np.pi, np.pi)
                self._particles.append(Particle3D(wx, wy, theta, 1.0 / N_PARTICLES))
                count += 1
            attempts += 1

    def _cb_odom(self, msg: Odometry):
        if not self._particles:
            return
        curr = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            self._quat_to_yaw(msg.pose.pose.orientation),
        ])
        if self._prev_odom is None:
            self._prev_odom = curr
            return

        # Odometry motion model — propagate particles
        betas = [0.05, 0.005, 0.005, 0.05]
        prev = self._prev_odom
        from pluto_filters.motion_models.odometry_motion_model import (
            sample_motion_model_odometry
        )
        for p in self._particles:
            new_pose = sample_motion_model_odometry(
                np.array([p.x, p.y, p.theta]),
                tuple(prev), tuple(curr), betas
            )
            p.x, p.y, p.theta = new_pose

        self._prev_odom = curr

    def _cb_scan(self, msg: LaserScan):
        if self._lf_field is None or not self._particles:
            return

        n = len(msg.ranges)
        z = np.clip(np.array(msg.ranges, dtype=float), 0, 12.0)
        angles = np.linspace(msg.angle_min, msg.angle_max, n)

        # Sample rays (use every 10th for speed)
        z_sub = z[::10]
        angles_sub = angles[::10]

        ox, oy = self._map_origin
        h, w = self._lf_field.shape

        for p in self._particles:
            log_w = 0.0
            for zi, angle in zip(z_sub, angles_sub):
                if zi >= 12.0:
                    continue
                bx = p.x + zi * np.cos(p.theta + angle)
                by = p.y + zi * np.sin(p.theta + angle)
                cx = int((bx - ox) / self._map_res)
                cy = int((by - oy) / self._map_res)
                if 0 <= cx < w and 0 <= cy < h:
                    log_w += np.log(max(float(self._lf_field[cy, cx]), 1e-300))
                else:
                    log_w += np.log(0.1 / 12.0)
            p.weight *= np.exp(log_w)

        # Normalize
        total = sum(p.weight for p in self._particles)
        if total < 1e-300:
            self._initialize_particles_uniform(*self._lf_field.shape[::-1])
            return
        for p in self._particles:
            p.weight /= total

        # Resample
        self._resample()
        self._publish_particles()
        self._check_convergence()

    def _resample(self):
        N = len(self._particles)
        weights = np.array([p.weight for p in self._particles])
        cumsum = np.cumsum(weights)
        r = np.random.uniform(0, 1.0 / N)
        positions = r + np.arange(N) / N
        new_particles = []
        j = 0
        for pos in positions:
            while j < N - 1 and cumsum[j] < pos:
                j += 1
            p = self._particles[j]
            new_particles.append(Particle3D(p.x, p.y, p.theta, 1.0 / N))
        self._particles = new_particles

    def _check_convergence(self):
        xs = np.array([p.x for p in self._particles])
        ys = np.array([p.y for p in self._particles])
        x_std = xs.std()
        y_std = ys.std()
        if x_std < 0.3 and y_std < 0.3:
            if not self._converged:
                self.get_logger().info(
                    f'Converged! std_x={x_std:.3f}, std_y={y_std:.3f}. '
                    f'Navigating home...'
                )
                self._converged = True

    def _mean_pose(self) -> tuple[float, float, float]:
        weights = np.array([p.weight for p in self._particles])
        xs = np.array([p.x for p in self._particles])
        ys = np.array([p.y for p in self._particles])
        thetas = np.array([p.theta for p in self._particles])
        x = float(np.average(xs, weights=weights))
        y = float(np.average(ys, weights=weights))
        theta = float(np.arctan2(
            np.average(np.sin(thetas), weights=weights),
            np.average(np.cos(thetas), weights=weights),
        ))
        return x, y, theta

    def _navigation_cb(self):
        if not self._particles:
            return
        x, y, theta = self._mean_pose()

        # Publish mean pose
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = 'map'
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.orientation.z = math.sin(theta / 2)
        pose_msg.pose.orientation.w = math.cos(theta / 2)
        self.pub_pose.publish(pose_msg)

        if self._converged:
            dist = np.sqrt((x - HOME_POSITION[0])**2 + (y - HOME_POSITION[1])**2)
            if dist > GOAL_TOLERANCE:
                # Simple proportional control toward home
                dx = HOME_POSITION[0] - x
                dy = HOME_POSITION[1] - y
                target_angle = np.arctan2(dy, dx)
                angle_err = norm_angle(target_angle - theta)

                cmd = Twist()
                cmd.linear.x = min(0.2, 0.5 * dist)
                cmd.angular.z = 1.5 * angle_err
                self.pub_cmd.publish(cmd)
            else:
                cmd = Twist()
                self.pub_cmd.publish(cmd)
                self.get_logger().info('Pluto arrived home!')

    def _publish_particles(self):
        markers = MarkerArray()
        now = self.get_clock().now().to_msg()
        weights = np.array([p.weight for p in self._particles])
        max_w = float(weights.max()) + 1e-300

        for i, p in enumerate(self._particles):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = now
            m.ns = 'mcl'
            m.id = i
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.pose.position.x = p.x
            m.pose.position.y = p.y
            m.pose.orientation.z = math.sin(p.theta / 2)
            m.pose.orientation.w = math.cos(p.theta / 2)
            m.scale.x = 0.12
            m.scale.y = 0.03
            m.scale.z = 0.03
            alpha = float(p.weight) / max_w
            m.color.r = 0.2
            m.color.g = 1.0
            m.color.b = 0.4
            m.color.a = max(alpha * 0.7, 0.05)
            markers.markers.append(m)

        self.pub_particles.publish(markers)

    @staticmethod
    def _quat_to_yaw(q) -> float:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y**2 + q.z**2)
        return math.atan2(siny_cosp, cosy_cosp)


def main(args=None):
    rclpy.init(args=args)
    node = MCLLocalizationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
