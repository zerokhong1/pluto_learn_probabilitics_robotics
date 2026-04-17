"""
Chapter 6 — Beam Model for range sensors.
4-component mixture: p_hit, p_short, p_max, p_rand.
Implements Table 6.1 (beam_range_finder_model) and
Table 6.2 (learn_intrinsic_parameters via EM) from Probabilistic Robotics.
"""

import numpy as np
from dataclasses import dataclass
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
import json


@dataclass
class BeamModelParams:
    z_max: float = 12.0    # max range [m]
    sigma_hit: float = 0.2  # std dev of hit component [m]
    lambda_short: float = 0.5  # exponential rate of short reads
    z_hit: float = 0.7     # mixture weight p_hit
    z_short: float = 0.07  # mixture weight p_short
    z_max_w: float = 0.07  # mixture weight p_max
    z_rand: float = 0.16   # mixture weight p_rand


def p_hit(z: float, z_star: float, params: BeamModelParams) -> float:
    """Gaussian around expected range z_star."""
    if 0 <= z <= params.z_max:
        return np.exp(-0.5 * ((z - z_star) / params.sigma_hit)**2) / (
            params.sigma_hit * np.sqrt(2 * np.pi))
    return 0.0


def p_short(z: float, z_star: float, params: BeamModelParams) -> float:
    """Exponential (unexpected obstacles closer than z_star)."""
    if 0 <= z <= z_star:
        eta = 1.0 / (1.0 - np.exp(-params.lambda_short * z_star))
        return eta * params.lambda_short * np.exp(-params.lambda_short * z)
    return 0.0


def p_max(z: float, params: BeamModelParams) -> float:
    """Point mass at z_max (beam misses everything)."""
    return 1.0 if abs(z - params.z_max) < 0.01 else 0.0


def p_rand(z: float, params: BeamModelParams) -> float:
    """Uniform over [0, z_max] (random noise)."""
    return 1.0 / params.z_max if 0 <= z <= params.z_max else 0.0


def beam_range_finder_model(z_readings: np.ndarray, z_star_readings: np.ndarray,
                              params: BeamModelParams) -> float:
    """
    Table 6.1: p(z_1:K | x, m) — full beam model for K rays.
    Returns log-likelihood for numerical stability.
    """
    log_p = 0.0
    for z, z_star in zip(z_readings, z_star_readings):
        q = (params.z_hit * p_hit(z, z_star, params) +
             params.z_short * p_short(z, z_star, params) +
             params.z_max_w * p_max(z, params) +
             params.z_rand * p_rand(z, params))
        log_p += np.log(max(q, 1e-300))
    return log_p


def learn_beam_model_params(z_data: np.ndarray, z_star_data: np.ndarray,
                             n_iters: int = 100) -> BeamModelParams:
    """
    EM algorithm to learn intrinsic parameters (Table 6.2).
    z_data: (N, K) measured ranges
    z_star_data: (N, K) expected ranges from ray-casting
    Returns fitted BeamModelParams.
    """
    params = BeamModelParams()  # initialize
    N = z_data.size
    z_flat = z_data.flatten()
    zs_flat = z_star_data.flatten()

    for iteration in range(n_iters):
        # E-step: compute responsibilities
        e_hit = np.array([params.z_hit * p_hit(z, zs, params)
                          for z, zs in zip(z_flat, zs_flat)])
        e_short = np.array([params.z_short * p_short(z, zs, params)
                            for z, zs in zip(z_flat, zs_flat)])
        e_max = np.array([params.z_max_w * p_max(z, params) for z in z_flat])
        e_rand = np.array([params.z_rand * p_rand(z, params) for z in z_flat])

        total = e_hit + e_short + e_max + e_rand + 1e-300
        e_hit /= total
        e_short /= total
        e_max /= total
        e_rand /= total

        # M-step: update parameters
        params.z_hit = float(e_hit.mean())
        params.z_short = float(e_short.mean())
        params.z_max_w = float(e_max.mean())
        params.z_rand = float(e_rand.mean())

        # Update sigma_hit
        num = np.sum(e_hit * (z_flat - zs_flat)**2)
        denom = e_hit.sum()
        params.sigma_hit = float(np.sqrt(num / (denom + 1e-300)))

        # Update lambda_short
        num_short = e_short.sum()
        denom_short = np.sum(e_short * z_flat)
        params.lambda_short = float(num_short / (denom_short + 1e-300))

    return params


class BeamModelNode(Node):
    """
    ROS2 node: beam model visualization.
    Color-codes each LiDAR ray by dominant component.
    Green=p_hit, Orange=p_short, Red=p_max, Purple=p_rand.
    """

    def __init__(self):
        super().__init__('beam_model_node')
        self._params = BeamModelParams()
        self._z_data_buffer = []

        self.pub_markers = self.create_publisher(MarkerArray, '/pluto/beam_rays', 10)
        self.pub_params = self.create_publisher(String, '/pluto/beam_params', 10)

        self.create_subscription(LaserScan, '/scan', self._cb_scan, 10)
        self.create_subscription(String, '/pluto/beam_expected', self._cb_expected, 10)

        self._z_star = None
        self.get_logger().info('Beam Model Node ready.')

    def _cb_expected(self, msg: String):
        data = json.loads(msg.data)
        self._z_star = np.array(data['ranges'])

    def _cb_scan(self, msg: LaserScan):
        n = len(msg.ranges)
        z_meas = np.clip(np.array(msg.ranges, dtype=float), 0, self._params.z_max)
        z_star = self._z_star if self._z_star is not None else np.full(n, 5.0)

        # Buffer for EM learning
        self._z_data_buffer.append(z_meas)
        if len(self._z_data_buffer) > 100:
            self._z_data_buffer.pop(0)

        # Color-code beams
        markers = MarkerArray()
        angles = np.linspace(msg.angle_min, msg.angle_max, n)

        for i, (z, zs, angle) in enumerate(zip(z_meas, z_star, angles)):
            ph = self._params.z_hit * p_hit(z, zs, self._params)
            ps = self._params.z_short * p_short(z, zs, self._params)
            pm = self._params.z_max_w * p_max(z, self._params)
            pr = self._params.z_rand * p_rand(z, self._params)
            dominant = np.argmax([ph, ps, pm, pr])

            m = Marker()
            m.header.frame_id = 'lidar_link'
            m.header.stamp = msg.header.stamp
            m.ns = 'beam'
            m.id = i
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.01

            colors = [
                (0.1, 0.9, 0.1, 0.8),   # green: p_hit
                (1.0, 0.5, 0.0, 0.8),   # orange: p_short
                (0.9, 0.1, 0.1, 0.8),   # red: p_max
                (0.6, 0.0, 0.8, 0.8),   # purple: p_rand
            ]
            r, g, b, a = colors[dominant]
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, a

            p_start = Point()
            p_end = Point()
            p_end.x = z * np.cos(angle)
            p_end.y = z * np.sin(angle)
            m.points.extend([p_start, p_end])
            markers.markers.append(m)

        self.pub_markers.publish(markers)

        # Publish params
        p = String()
        p.data = json.dumps({
            'z_hit': self._params.z_hit,
            'z_short': self._params.z_short,
            'z_max': self._params.z_max_w,
            'z_rand': self._params.z_rand,
            'sigma_hit': self._params.sigma_hit,
            'lambda_short': self._params.lambda_short,
        })
        self.pub_params.publish(p)


def main(args=None):
    rclpy.init(args=args)
    node = BeamModelNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
