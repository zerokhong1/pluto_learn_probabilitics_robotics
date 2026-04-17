"""
Pluto Hallway Simulator — standalone demo (no Gazebo needed).
Simulates Pluto moving through the hallway, running the Discrete Bayes Filter,
and publishing all topics needed by RViz2.

Published topics:
  /tf                       — odom → base_footprint → base_link → ... (full TF tree)
  /robot_description        — URDF for RobotModel display
  /joint_states             — wheel rotations
  /odom                     — ground-truth odometry
  /scan                     — simulated 2D LiDAR
  /pluto/belief_markers     — Bayes filter histogram (MarkerArray)
  /pluto/belief_1d          — raw belief array
  /pluto/filter_state       — JSON stats (mode, entropy, error)
  /pluto/door_detected      — binary door sensor
  /pluto/eye_marker         — eye color (uncertainty indicator)
  /pluto/gt_path            — ground truth path marker
  /pluto/dead_reckoning     — dead-reckoning path (drifted)
"""

import math
import json
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped, Twist, Point
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan, JointState
from std_msgs.msg import Float64MultiArray, Bool, String, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from pluto_filters.bayes_filter.discrete_bayes_filter import (
    DiscreteBayesFilter1D, CELL_CENTERS, DOOR_POSITIONS, N_CELLS, CELL_SIZE
)

# ─── Hallway geometry ───────────────────────────────────────────────────────
HALLWAY_LENGTH = 10.0
HALLWAY_WIDTH  = 2.0
WALL_Y         = 1.05     # y position of walls
DOOR_WIDTH     = 0.8      # opening width at each door

# ─── Simulation parameters ──────────────────────────────────────────────────
DT             = 0.1      # simulation step [s]
ROBOT_SPEED    = 0.3      # forward velocity [m/s]
LIDAR_RANGE    = 12.0
LIDAR_RAYS     = 360
ENCODER_NOISE  = 0.003    # m/step
IMU_NOISE      = 0.001    # rad/step

# ─── Door sensor model ───────────────────────────────────────────────────────
DOOR_DETECT_RANGE = 0.55   # robot detects door if within this distance [m]
P_HIT_DOOR        = 0.85   # true positive rate
P_FALSE_ALARM     = 0.15   # false positive rate


def _quat_from_yaw(yaw: float):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


class HallwaySimulator(Node):
    """
    Self-contained Pluto hallway simulation + Bayes filter visualization node.
    Auto-drives Pluto forward, detects doors, runs the filter, visualizes everything.
    """

    def __init__(self):
        super().__init__('hallway_simulator')

        # ── Robot state ──────────────────────────────────────────────────────
        self._x     = 0.5    # true position [m]
        self._y     = 0.0
        self._theta = 0.0    # true heading [rad]

        # Dead-reckoning (drifted) state
        self._dr_x     = 0.5
        self._dr_y     = 0.0
        self._dr_theta = 0.0

        self._wheel_angle = 0.0   # cumulative wheel rotation [rad]
        self._t = 0.0
        self._direction = 1.0    # +1 = forward, -1 = backward
        self._turning   = False  # True while doing 180° turn
        self._turn_accum = 0.0   # accumulated turn angle [rad]

        # Mission phases: 0=approach door1, 1=pass door1, 2=approach door2 …
        self._phase = 0
        self._door_passed = [False, False, False]

        # ── Filter ───────────────────────────────────────────────────────────
        self._bf = DiscreteBayesFilter1D()

        # ── ROS publishers ───────────────────────────────────────────────────
        self._tf_br     = TransformBroadcaster(self)
        self._stf_br    = StaticTransformBroadcaster(self)

        self._pub_odom   = self.create_publisher(Odometry,          '/odom',                  10)
        self._pub_scan   = self.create_publisher(LaserScan,         '/scan',                  10)
        self._pub_jstate = self.create_publisher(JointState,        '/joint_states',          10)
        self._pub_belief = self.create_publisher(Float64MultiArray, '/pluto/belief_1d',       10)
        self._pub_bmark  = self.create_publisher(MarkerArray,       '/pluto/belief_markers',  10)
        self._pub_state  = self.create_publisher(String,            '/pluto/filter_state',    10)
        self._pub_door   = self.create_publisher(Bool,              '/pluto/door_detected',   10)
        self._pub_eye    = self.create_publisher(MarkerArray,       '/pluto/eye_marker',      10)
        self._pub_gtpath = self.create_publisher(Marker,            '/pluto/gt_path',         10)
        self._pub_drpath = self.create_publisher(Marker,            '/pluto/dead_reckoning',  10)

        # Path history
        self._gt_path_pts  = []
        self._dr_path_pts  = []

        # Static TF: map → odom (identity for this demo)
        self._publish_static_tfs()

        # Main simulation timer
        self.create_timer(DT, self._step)
        self.get_logger().info('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
        self.get_logger().info(' Pluto Hallway Simulator started!')
        self.get_logger().info(' Driving forward at 0.3 m/s …')
        self.get_logger().info('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')

    # ─────────────────────────────────────────────────────────────────────────
    def _publish_static_tfs(self):
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = 'map'
        tf.child_frame_id  = 'odom'
        tf.transform.rotation.w = 1.0
        self._stf_br.sendTransform([tf])

    # ─────────────────────────────────────────────────────────────────────────
    def _step(self):
        self._t += DT

        # ── Auto-pilot: bounce between walls ─────────────────────────────────
        if self._turning:
            # In-place 180° turn at constant angular velocity
            turn_speed = 1.5   # rad/s
            v     = 0.0
            omega = turn_speed * self._direction   # direction of rotation
            self._turn_accum += abs(omega) * DT
            if self._turn_accum >= math.pi:
                self._turning    = False
                self._turn_accum = 0.0
                self._direction  = -self._direction
                self.get_logger().info(
                    f'━━ Pluto turned around → driving {"forward" if self._direction > 0 else "back"} ━━'
                )
        elif self._direction > 0 and self._x >= HALLWAY_LENGTH - 0.3:
            # Reached far end → start turning
            self._turning = True
            self._turn_accum = 0.0
            self._gt_path_pts.clear()
            self._dr_path_pts.clear()
            self._bf = DiscreteBayesFilter1D()
            v, omega = 0.0, 0.0
        elif self._direction < 0 and self._x <= 0.3:
            # Reached near end → start turning
            self._turning = True
            self._turn_accum = 0.0
            self._gt_path_pts.clear()
            self._dr_path_pts.clear()
            self._bf = DiscreteBayesFilter1D()
            v, omega = 0.0, 0.0
        else:
            v, omega = ROBOT_SPEED, 0.0

        # ── True motion ───────────────────────────────────────────────────────
        self._x     += v * math.cos(self._theta) * DT
        self._y     += v * math.sin(self._theta) * DT
        self._theta += omega * DT
        self._wheel_angle += v / 0.07 * DT   # wheel_angle += v/r * dt

        # ── Dead-reckoning with noise ─────────────────────────────────────────
        v_noise     = v     + np.random.normal(0, ENCODER_NOISE / DT)
        omega_noise = omega + np.random.normal(0, IMU_NOISE / DT)
        self._dr_theta += omega_noise * DT
        self._dr_x     += v_noise * math.cos(self._dr_theta) * DT
        self._dr_y     += v_noise * math.sin(self._dr_theta) * DT

        # ── Door detection ────────────────────────────────────────────────────
        near_door = any(abs(self._x - d) < DOOR_DETECT_RANGE for d in DOOR_POSITIONS)
        if near_door:
            z = np.random.random() < P_HIT_DOOR
        else:
            z = np.random.random() < P_FALSE_ALARM

        # ── Filter update ─────────────────────────────────────────────────────
        # Predict on every step, update only when sensor fires
        self._bf.predict(v * DT)
        if near_door or z:
            self._bf.update(z)

        # ── Publish all ───────────────────────────────────────────────────────
        now = self.get_clock().now().to_msg()
        self._publish_tf(now)
        self._publish_odom(now, v, omega)
        self._publish_joint_states(now)
        self._publish_scan(now)
        self._publish_door_sensor(z)
        self._publish_belief(now)
        self._publish_eye_color(now)
        self._publish_paths(now)

    # ─────────────────────────────────────────────────────────────────────────
    def _publish_tf(self, stamp):
        tfs = []

        # odom → base_footprint
        t = TransformStamped()
        t.header.stamp    = stamp
        t.header.frame_id = 'odom'
        t.child_frame_id  = 'base_footprint'
        t.transform.translation.x = self._x
        t.transform.translation.y = self._y
        qz, qw = math.sin(self._theta / 2), math.cos(self._theta / 2)
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        tfs.append(t)

        # base_footprint → base_link
        t2 = TransformStamped()
        t2.header.stamp    = stamp
        t2.header.frame_id = 'base_footprint'
        t2.child_frame_id  = 'base_link'
        t2.transform.translation.z = 0.0
        t2.transform.rotation.w    = 1.0
        tfs.append(t2)

        # base_link → lidar_link
        t3 = TransformStamped()
        t3.header.stamp    = stamp
        t3.header.frame_id = 'base_link'
        t3.child_frame_id  = 'lidar_link'
        t3.transform.translation.x = -0.05
        t3.transform.translation.y =  0.06
        t3.transform.translation.z =  0.95
        t3.transform.rotation.w    = 1.0
        tfs.append(t3)

        self._tf_br.sendTransform(tfs)

    # ─────────────────────────────────────────────────────────────────────────
    def _publish_odom(self, stamp, v, omega):
        msg = Odometry()
        msg.header.stamp    = stamp
        msg.header.frame_id = 'odom'
        msg.child_frame_id  = 'base_footprint'
        msg.pose.pose.position.x = self._x
        msg.pose.pose.position.y = self._y
        msg.pose.pose.orientation.z = math.sin(self._theta / 2)
        msg.pose.pose.orientation.w = math.cos(self._theta / 2)
        msg.twist.twist.linear.x  = v
        msg.twist.twist.angular.z = omega
        self._pub_odom.publish(msg)

    # ─────────────────────────────────────────────────────────────────────────
    def _publish_joint_states(self, stamp):
        msg = JointState()
        msg.header.stamp = stamp
        msg.name     = ['left_wheel_joint', 'right_wheel_joint']
        msg.position = [self._wheel_angle, self._wheel_angle]
        msg.velocity = [ROBOT_SPEED / 0.07, ROBOT_SPEED / 0.07]
        self._pub_jstate.publish(msg)

    # ─────────────────────────────────────────────────────────────────────────
    def _publish_scan(self, stamp):
        """Simulate 2D LiDAR via ray-casting against hallway geometry."""
        msg = LaserScan()
        msg.header.stamp    = stamp
        msg.header.frame_id = 'lidar_link'
        msg.angle_min    = -math.pi
        msg.angle_max    =  math.pi
        msg.angle_increment = 2 * math.pi / LIDAR_RAYS
        msg.range_min    = 0.12
        msg.range_max    = LIDAR_RANGE
        msg.time_increment = 0.0

        ranges = []
        for i in range(LIDAR_RAYS):
            angle = -math.pi + i * msg.angle_increment + self._theta
            r = self._ray_cast(angle)
            r += np.random.normal(0, 0.01)
            ranges.append(float(np.clip(r, msg.range_min, msg.range_max)))

        msg.ranges = ranges
        self._pub_scan.publish(msg)

    def _ray_cast(self, world_angle: float) -> float:
        """Simple ray-casting against hallway box."""
        cx, cy = self._x, self._y
        dx = math.cos(world_angle)
        dy = math.sin(world_angle)
        tmin = LIDAR_RANGE

        # Left wall  y = +WALL_Y
        if abs(dy) > 1e-9:
            t = (WALL_Y - cy) / dy
            if t > 0:
                hx = cx + t * dx
                if 0 <= hx <= HALLWAY_LENGTH:
                    # check for door opening
                    at_door = any(abs(hx - d) < DOOR_WIDTH / 2 for d in DOOR_POSITIONS)
                    if not at_door:
                        tmin = min(tmin, t)

        # Right wall y = -WALL_Y
        if abs(dy) > 1e-9:
            t = (-WALL_Y - cy) / dy
            if t > 0:
                hx = cx + t * dx
                if 0 <= hx <= HALLWAY_LENGTH:
                    tmin = min(tmin, t)

        # Start cap x = 0
        if abs(dx) > 1e-9:
            t = (0 - cx) / dx
            if t > 0 and abs(cy + t * dy) <= WALL_Y:
                tmin = min(tmin, t)

        # End cap x = 10
        if abs(dx) > 1e-9:
            t = (HALLWAY_LENGTH - cx) / dx
            if t > 0 and abs(cy + t * dy) <= WALL_Y:
                tmin = min(tmin, t)

        return tmin

    # ─────────────────────────────────────────────────────────────────────────
    def _publish_door_sensor(self, z: bool):
        msg = Bool()
        msg.data = z
        self._pub_door.publish(msg)

    # ─────────────────────────────────────────────────────────────────────────
    def _publish_belief(self, stamp):
        belief = self._bf.belief
        mode_x = CELL_CENTERS[np.argmax(belief)]
        entropy = self._bf.uncertainty()
        error = abs(mode_x - self._x)

        # Raw array
        arr = Float64MultiArray()
        arr.data = belief.tolist()
        self._pub_belief.publish(arr)

        # MarkerArray histogram — rendered at y = -1.8 in world frame
        markers = MarkerArray()
        max_val = float(belief.max()) + 1e-10
        for i, (cx, prob) in enumerate(zip(CELL_CENTERS, belief)):
            m = Marker()
            m.header.frame_id = 'odom'
            m.header.stamp    = stamp
            m.ns     = 'belief_1d'
            m.id     = i
            m.type   = Marker.CUBE
            m.action = Marker.ADD
            bar_h = float(prob) / max_val * 1.2
            m.pose.position.x = cx
            m.pose.position.y = -1.8
            m.pose.position.z = bar_h / 2.0
            m.scale.x = CELL_SIZE * 0.9
            m.scale.y = 0.06
            m.scale.z = max(bar_h, 0.01)
            t = float(prob) / max_val
            m.color.r = t
            m.color.g = 0.25
            m.color.b = 1.0 - t
            m.color.a = 0.85
            markers.markers.append(m)

        # Door position markers (vertical red lines)
        for j, d in enumerate(DOOR_POSITIONS):
            dm = Marker()
            dm.header.frame_id = 'odom'
            dm.header.stamp    = stamp
            dm.ns     = 'doors'
            dm.id     = j
            dm.type   = Marker.CUBE
            dm.action = Marker.ADD
            dm.pose.position.x = d
            dm.pose.position.y = -1.8
            dm.pose.position.z = 0.65
            dm.scale.x = 0.04
            dm.scale.y = 0.06
            dm.scale.z = 1.3
            dm.color.r = 0.9
            dm.color.g = 0.1
            dm.color.b = 0.1
            dm.color.a = 0.6
            markers.markers.append(dm)

        # Ground-truth robot position marker on histogram
        gt_m = Marker()
        gt_m.header.frame_id = 'odom'
        gt_m.header.stamp    = stamp
        gt_m.ns     = 'gt_marker'
        gt_m.id     = 0
        gt_m.type   = Marker.ARROW
        gt_m.action = Marker.ADD
        gt_m.pose.position.x = self._x
        gt_m.pose.position.y = -1.8
        gt_m.pose.position.z = 1.4
        gt_m.pose.orientation.y = math.sin(math.pi / 4)
        gt_m.pose.orientation.w = math.cos(-math.pi / 4)
        gt_m.scale.x = 0.3
        gt_m.scale.y = 0.08
        gt_m.scale.z = 0.08
        gt_m.color.r = 0.1
        gt_m.color.g = 0.9
        gt_m.color.b = 0.2
        gt_m.color.a = 1.0
        markers.markers.append(gt_m)

        self._pub_bmark.publish(markers)

        # JSON state
        s = String()
        s.data = json.dumps({
            'mode': float(mode_x),
            'entropy': float(entropy),
            'gt_x': float(self._x),
            'error': float(error),
            't': float(self._t),
        })
        self._pub_state.publish(s)

        if int(self._t * 10) % 20 == 0:   # every 2s
            self.get_logger().info(
                f't={self._t:.1f}s  gt={self._x:.2f}m  '
                f'mode={mode_x:.2f}m  err={error:.2f}m  H={entropy:.3f}'
            )

    # ─────────────────────────────────────────────────────────────────────────
    def _publish_eye_color(self, stamp):
        entropy = self._bf.uncertainty()
        max_entropy = -math.log(1.0 / N_CELLS)  # uniform distribution entropy
        t = min(entropy / max_entropy, 1.0)

        # blue (confident) → yellow → red (uncertain)
        if t < 0.5:
            s = t * 2
            r, g, b = s, s, 1.0 - s
        else:
            s = (t - 0.5) * 2
            r, g, b = 1.0, 1.0 - s, 0.0

        markers = MarkerArray()
        for i, (ey, ez) in enumerate([(0.05, 0.67), (-0.05, 0.67)]):
            m = Marker()
            m.header.frame_id = 'base_link'
            m.header.stamp    = stamp
            m.ns     = 'pluto_eye'
            m.id     = i
            m.type   = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = 0.10
            m.pose.position.y = ey
            m.pose.position.z = ez
            m.scale.x = m.scale.y = m.scale.z = 0.05
            m.color.r = r
            m.color.g = g
            m.color.b = b
            m.color.a = 1.0
            markers.markers.append(m)
        self._pub_eye.publish(markers)

    # ─────────────────────────────────────────────────────────────────────────
    def _publish_paths(self, stamp):
        self._gt_path_pts.append((self._x,    self._y))
        self._dr_path_pts.append((self._dr_x, self._dr_y))
        if len(self._gt_path_pts) > 500:
            self._gt_path_pts.pop(0)
            self._dr_path_pts.pop(0)

        def make_path_marker(pts, ns, r, g, b, id_=0):
            m = Marker()
            m.header.frame_id = 'odom'
            m.header.stamp    = stamp
            m.ns     = ns
            m.id     = id_
            m.type   = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.04
            m.color.r = r
            m.color.g = g
            m.color.b = b
            m.color.a = 0.85
            for px, py in pts:
                p = Point()
                p.x, p.y, p.z = px, py, 0.05
                m.points.append(p)
            return m

        self._pub_gtpath.publish(
            make_path_marker(self._gt_path_pts, 'gt_path', 0.1, 0.9, 0.2))
        self._pub_drpath.publish(
            make_path_marker(self._dr_path_pts, 'dr_path', 0.9, 0.4, 0.1))


def main(args=None):
    rclpy.init(args=args)
    node = HallwaySimulator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
