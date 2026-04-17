"""
Chapter 2 — Discrete Bayes Filter (1D hallway localization).
Implements the classic "robot in a hallway with identical doors" example
from Probabilistic Robotics (Thrun et al.), pp. 27-28.

State space: 1D grid [0, 10m] with 100 cells (resolution 0.1m).
Sensor:      Binary door detector.
Motion:      Commanded velocity with additive Gaussian noise.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Bool, String
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import json


# Known door positions [m] — matches hallway_with_doors.sdf
DOOR_POSITIONS = [2.0, 5.0, 8.0]

# Grid parameters
GRID_MIN = 0.0
GRID_MAX = 10.0
N_CELLS = 100
CELL_SIZE = (GRID_MAX - GRID_MIN) / N_CELLS
CELL_CENTERS = np.linspace(GRID_MIN + CELL_SIZE / 2, GRID_MAX - CELL_SIZE / 2, N_CELLS)

# Sensor model parameters (from book Table 2.1 style)
P_HIT_DOOR = 0.6    # P(sense door | door is there)
P_FALSE_ALARM = 0.2  # P(sense door | no door)

# Motion model parameters
MOTION_SIGMA = 0.05  # std dev of position uncertainty per step [m]


def measurement_model(z: bool) -> np.ndarray:
    """
    p(z | x) for binary door observation.
    Returns likelihood vector over all cells.
    """
    # Distance from each cell to nearest door
    dists = np.array([
        min(abs(c - d) for d in DOOR_POSITIONS) for c in CELL_CENTERS
    ])
    at_door = dists < CELL_SIZE  # boolean mask

    if z:  # robot sees a door
        return np.where(at_door, P_HIT_DOOR, P_FALSE_ALARM)
    else:  # robot does not see a door
        return np.where(at_door, 1.0 - P_HIT_DOOR, 1.0 - P_FALSE_ALARM)


def motion_model_kernel(delta: float) -> np.ndarray:
    """
    Convolution kernel for motion model p(x' | x, u).
    delta: commanded displacement [m].
    Returns N_CELLS x N_CELLS transition matrix row (shift + Gaussian noise).
    """
    # Gaussian kernel centered at delta
    offsets = np.linspace(-3 * MOTION_SIGMA, 3 * MOTION_SIGMA, 21)
    kernel_vals = np.exp(-0.5 * (offsets / MOTION_SIGMA) ** 2)
    kernel_vals /= kernel_vals.sum()

    kernel_shifts = np.round(offsets / CELL_SIZE).astype(int)
    delta_cells = int(round(delta / CELL_SIZE))

    # Apply as 1D convolution with shift
    def apply(belief: np.ndarray) -> np.ndarray:
        new_belief = np.zeros(N_CELLS)
        for shift, weight in zip(kernel_shifts, kernel_vals):
            total_shift = delta_cells + shift
            new_belief += weight * np.roll(belief, total_shift)
        return new_belief

    return apply


class DiscreteBayesFilter1D:
    """Pure-Python implementation of the discrete Bayes filter."""

    def __init__(self):
        self.belief = np.ones(N_CELLS) / N_CELLS  # uniform prior

    def predict(self, delta: float):
        """Prediction step: bel_bar(x) = sum_{x'} p(x | x', u) * bel(x')."""
        apply_motion = motion_model_kernel(delta)
        self.belief = apply_motion(self.belief)
        # Normalize (should be ~1 already, but numerical safety)
        self.belief /= self.belief.sum()

    def update(self, z: bool):
        """Correction step: bel(x) = eta * p(z | x) * bel_bar(x)."""
        likelihood = measurement_model(z)
        self.belief *= likelihood
        total = self.belief.sum()
        if total > 1e-10:
            self.belief /= total
        else:
            # Particle deprivation — reset to uniform
            self.belief = np.ones(N_CELLS) / N_CELLS

    def most_likely_position(self) -> float:
        return CELL_CENTERS[np.argmax(self.belief)]

    def uncertainty(self) -> float:
        """Entropy of the belief distribution."""
        p = self.belief + 1e-300
        return -float(np.sum(p * np.log(p)))


class BayesFilter1DNode(Node):
    """
    ROS2 node wrapping DiscreteBayesFilter1D.

    Subscriptions:
      /pluto/door_detected  (std_msgs/Bool)  — sensor reading
      /pluto/move_cmd       (std_msgs/String) — JSON {"delta": float}
      /odom                 (nav_msgs/Odometry) — ground truth for comparison

    Publications:
      /pluto/belief_1d      (std_msgs/Float64MultiArray) — raw belief vector
      /pluto/belief_markers (visualization_msgs/MarkerArray) — RViz2 histogram
      /pluto/filter_state   (std_msgs/String) — JSON with stats
    """

    def __init__(self):
        super().__init__('bayes_filter_1d')
        self._filter = DiscreteBayesFilter1D()
        self._gt_x = 0.0

        self.pub_belief = self.create_publisher(Float64MultiArray, '/pluto/belief_1d', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/pluto/belief_markers', 10)
        self.pub_state = self.create_publisher(String, '/pluto/filter_state', 10)

        self.create_subscription(Bool, '/pluto/door_detected', self._cb_sensor, 10)
        self.create_subscription(String, '/pluto/move_cmd', self._cb_move, 10)
        self.create_subscription(Odometry, '/odom', self._cb_odom, 10)

        self.create_timer(0.1, self._publish_belief)
        self.get_logger().info('Discrete Bayes Filter 1D ready.')

    def _cb_sensor(self, msg: Bool):
        self._filter.update(msg.data)
        self.get_logger().info(
            f'Sensor update z={msg.data} → '
            f'mode={self._filter.most_likely_position():.2f}m, '
            f'H={self._filter.uncertainty():.3f}'
        )

    def _cb_move(self, msg: String):
        data = json.loads(msg.data)
        delta = float(data.get('delta', 0.0))
        self._filter.predict(delta)
        self.get_logger().info(f'Motion update delta={delta:.2f}m')

    def _cb_odom(self, msg: Odometry):
        self._gt_x = msg.pose.pose.position.x

    def _publish_belief(self):
        # Raw belief array
        arr = Float64MultiArray()
        arr.data = self._filter.belief.tolist()
        self.pub_belief.publish(arr)

        # MarkerArray for RViz2 — bar chart visualization
        markers = MarkerArray()
        max_val = float(np.max(self._filter.belief)) + 1e-10
        for i, (x, prob) in enumerate(zip(CELL_CENTERS, self._filter.belief)):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'belief_1d'
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            bar_height = float(prob) / max_val * 2.0  # scale to max 2m
            m.pose.position.x = x
            m.pose.position.y = -2.0  # offset from hallway
            m.pose.position.z = bar_height / 2.0
            m.scale.x = CELL_SIZE * 0.9
            m.scale.y = 0.05
            m.scale.z = max(bar_height, 0.01)
            # Color: blue (low) → red (high)
            t = float(prob) / max_val
            m.color.r = t
            m.color.g = 0.2
            m.color.b = 1.0 - t
            m.color.a = 0.85
            markers.markers.append(m)
        self.pub_markers.publish(markers)

        # State JSON
        state = String()
        state.data = json.dumps({
            'mode': self._filter.most_likely_position(),
            'entropy': self._filter.uncertainty(),
            'gt_x': self._gt_x,
            'error': abs(self._filter.most_likely_position() - self._gt_x),
        })
        self.pub_state.publish(state)


def main(args=None):
    rclpy.init(args=args)
    node = BayesFilter1DNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
