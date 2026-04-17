"""
Chapter 5 — Velocity Motion Model.
Implements Table 5.3 (sample_motion_model_velocity) and Table 5.1
(motion_model_velocity probability) from Probabilistic Robotics.

Visualization: "Banana distribution" — scatter of 1000 predicted poses.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
import json
import math

from ..kalman_filters.ekf import normalize_angle


# Motion noise parameters α1..α6 (Table 5.3 notation)
# α1,α2: noise in velocity commands
# α3,α4: noise in rotational rate commands
# α5,α6: additional drift noise
DEFAULT_ALPHAS = [0.1, 0.01, 0.01, 0.1, 0.001, 0.001]


def sample_motion_model_velocity(x: np.ndarray, v: float, omega: float,
                                 dt: float, alphas: list = DEFAULT_ALPHAS) -> np.ndarray:
    """
    Table 5.3 from Probabilistic Robotics.
    x = [x_pos, y_pos, theta]
    Returns sampled next pose.
    """
    a1, a2, a3, a4, a5, a6 = alphas
    v_hat = v + np.random.normal(0, np.sqrt(a1 * v**2 + a2 * omega**2))
    omega_hat = omega + np.random.normal(0, np.sqrt(a3 * v**2 + a4 * omega**2))
    gamma_hat = np.random.normal(0, np.sqrt(a5 * v**2 + a6 * omega**2))

    px, py, theta = x
    if abs(omega_hat) < 1e-6:
        new_x = px + v_hat * dt * np.cos(theta)
        new_y = py + v_hat * dt * np.sin(theta)
    else:
        r = v_hat / omega_hat
        new_x = px - r * np.sin(theta) + r * np.sin(theta + omega_hat * dt)
        new_y = py + r * np.cos(theta) - r * np.cos(theta + omega_hat * dt)

    new_theta = normalize_angle(theta + omega_hat * dt + gamma_hat * dt)
    return np.array([new_x, new_y, new_theta])


def motion_model_velocity(x_prime: np.ndarray, x: np.ndarray,
                           v: float, omega: float, dt: float,
                           alphas: list = DEFAULT_ALPHAS) -> float:
    """
    Table 5.1 from Probabilistic Robotics.
    Returns p(x' | u, x) — probability of ending up at x' given motion u from x.
    """
    a1, a2, a3, a4, a5, a6 = alphas
    px, py, theta = x
    px2, py2, theta2 = x_prime

    # Recover commanded motion
    mu_num = (px - px2) * np.cos(theta) + (py - py2) * np.sin(theta)
    mu_den = (py - py2) * np.cos(theta) - (px - px2) * np.sin(theta)
    mu = 0.5 * mu_num / (mu_den + 1e-10)

    x_star = (px + px2) / 2 + mu * np.cos(theta)
    y_star = (py + py2) / 2 + mu * np.sin(theta)

    r_star = np.sqrt((px - x_star)**2 + (py - y_star)**2)
    delta_theta = np.arctan2(py2 - y_star, px2 - x_star) - np.arctan2(py - y_star, px - x_star)
    delta_theta = normalize_angle(delta_theta)

    v_hat = delta_theta / dt * r_star
    omega_hat = delta_theta / dt
    gamma_hat = normalize_angle(theta2 - theta) / dt - omega_hat

    def prob_normal(a, b):
        return np.exp(-0.5 * a**2 / b) / np.sqrt(2 * np.pi * b + 1e-300)

    p1 = prob_normal(v - v_hat, a1 * v**2 + a2 * omega**2)
    p2 = prob_normal(omega - omega_hat, a3 * v**2 + a4 * omega**2)
    p3 = prob_normal(gamma_hat, a5 * v**2 + a6 * omega**2)
    return float(p1 * p2 * p3)


class MotionModelNode(Node):
    """
    ROS2 node demonstrating the "banana distribution".
    Samples 1000 poses from velocity motion model and publishes as point cloud.

    Subscribe to /pluto/motion_cmd (String JSON {"v": float, "omega": float, "dt": float, "alphas": [...] })
    Publishes /pluto/banana_cloud (MarkerArray)
    """

    def __init__(self):
        super().__init__('motion_model_node')
        self._start_pose = np.array([0.0, 0.0, 0.0])
        self._alphas = DEFAULT_ALPHAS.copy()

        self.pub_cloud = self.create_publisher(MarkerArray, '/pluto/banana_cloud', 10)
        self.create_subscription(String, '/pluto/motion_cmd', self._cb_cmd, 10)
        self.get_logger().info('Motion Model Node ready. Publish to /pluto/motion_cmd')

    def _cb_cmd(self, msg: String):
        data = json.loads(msg.data)
        v = float(data.get('v', 0.5))
        omega = float(data.get('omega', 0.05))
        dt = float(data.get('dt', 1.0))
        alphas = data.get('alphas', self._alphas)

        n_samples = 1000
        samples = np.array([
            sample_motion_model_velocity(self._start_pose, v, omega, dt, alphas)
            for _ in range(n_samples)
        ])

        markers = MarkerArray()
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'banana'
        m.id = 0
        m.type = Marker.POINTS
        m.action = Marker.ADD
        m.scale.x = 0.04
        m.scale.y = 0.04
        m.color.r = 0.95
        m.color.g = 0.75
        m.color.b = 0.05
        m.color.a = 0.7

        from geometry_msgs.msg import Point
        from std_msgs.msg import ColorRGBA
        for s in samples:
            p = Point()
            p.x, p.y, p.z = float(s[0]), float(s[1]), 0.02
            m.points.append(p)

        markers.markers.append(m)
        self.pub_cloud.publish(markers)

        self.get_logger().info(
            f'Banana dist: v={v:.2f}, ω={omega:.3f}, '
            f'mean_x={samples[:, 0].mean():.3f}, '
            f'std_x={samples[:, 0].std():.3f}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = MotionModelNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
