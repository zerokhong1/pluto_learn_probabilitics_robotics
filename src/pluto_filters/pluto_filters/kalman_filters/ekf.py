"""
Chapter 3 — Extended Kalman Filter (EKF).
State: x = [x, y, θ]^T — 2D differential drive robot.
Measurement: range + bearing to known landmarks.
Implements Algorithm 3.3 from Probabilistic Robotics.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import json


# Known landmark positions in the world [m]
LANDMARKS = {
    0: np.array([3.0, 0.0]),
    1: np.array([-3.0, 2.0]),
    2: np.array([0.0, -4.0]),
    3: np.array([4.0, 4.0]),
    4: np.array([-4.0, -3.0]),
}

# Noise parameters
ALPHA = [0.1, 0.01, 0.01, 0.1]  # motion noise [a1..a4]
SIGMA_R = 0.2   # range noise [m]
SIGMA_PHI = 0.05  # bearing noise [rad]


def normalize_angle(a: float) -> float:
    while a > np.pi:
        a -= 2 * np.pi
    while a < -np.pi:
        a += 2 * np.pi
    return a


class EKF:
    """
    Extended Kalman Filter for 2D robot localization with range-bearing sensors.
    State: μ = [x, y, θ], Σ = 3×3 covariance.
    """

    def __init__(self, mu0: np.ndarray, sigma0: np.ndarray):
        self.mu = mu0.copy()
        self.sigma = sigma0.copy()
        # Measurement noise covariance
        self.Q = np.diag([SIGMA_R**2, SIGMA_PHI**2])

    def predict(self, v: float, omega: float, dt: float):
        """
        EKF prediction step (Algorithm 3.3 lines 2-4).
        Differential drive motion model:
          x' = x + v*cos(θ)*dt
          y' = y + v*sin(θ)*dt
          θ' = θ + ω*dt
        """
        theta = self.mu[2]

        # Motion model g(u, x)
        if abs(omega) < 1e-6:
            dx = v * np.cos(theta) * dt
            dy = v * np.sin(theta) * dt
        else:
            dx = -(v / omega) * np.sin(theta) + (v / omega) * np.sin(theta + omega * dt)
            dy = (v / omega) * np.cos(theta) - (v / omega) * np.cos(theta + omega * dt)
        dtheta = omega * dt

        self.mu = self.mu + np.array([dx, dy, dtheta])
        self.mu[2] = normalize_angle(self.mu[2])

        # Jacobian G = ∂g/∂x
        G = np.eye(3)
        G[0, 2] = -v * np.sin(theta) * dt
        G[1, 2] = v * np.cos(theta) * dt

        # Process noise R (velocity motion model, simplified)
        a1, a2, a3, a4 = ALPHA
        R = np.diag([
            a1 * v**2 + a2 * omega**2,
            a1 * v**2 + a2 * omega**2,
            a3 * v**2 + a4 * omega**2,
        ])

        self.sigma = G @ self.sigma @ G.T + R

    def update(self, landmark_id: int, r_meas: float, phi_meas: float):
        """
        EKF correction step (Algorithm 3.3 lines 5-9).
        Measurement model: h(x) = [sqrt((lx-x)^2+(ly-y)^2), atan2(ly-y, lx-x) - θ]
        """
        if landmark_id not in LANDMARKS:
            return

        lx, ly = LANDMARKS[landmark_id]
        x, y, theta = self.mu

        dx = lx - x
        dy = ly - y
        q = dx**2 + dy**2

        # Expected measurement h(μ)
        r_hat = np.sqrt(q)
        phi_hat = normalize_angle(np.arctan2(dy, dx) - theta)

        # Jacobian H = ∂h/∂x
        H = np.array([
            [-dx / r_hat, -dy / r_hat, 0.0],
            [dy / q, -dx / q, -1.0],
        ])

        S = H @ self.sigma @ H.T + self.Q
        K = self.sigma @ H.T @ np.linalg.inv(S)

        innovation = np.array([r_meas - r_hat, normalize_angle(phi_meas - phi_hat)])
        self.mu = self.mu + K @ innovation
        self.mu[2] = normalize_angle(self.mu[2])
        self.sigma = (np.eye(3) - K @ H) @ self.sigma

    def covariance_ellipse_2d(self, n_points: int = 36) -> np.ndarray:
        """Returns (n_points, 2) array of 2σ covariance ellipse in XY plane."""
        vals, vecs = np.linalg.eigh(self.sigma[:2, :2])
        angles = np.linspace(0, 2 * np.pi, n_points)
        circle = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        ellipse = circle @ np.diag(2 * np.sqrt(np.abs(vals))) @ vecs.T
        return ellipse + self.mu[:2]


class EKFNode(Node):
    """
    ROS2 node for EKF 2D localization.

    Subscriptions:
      /pluto/ekf_control  (String) — JSON {"v": float, "omega": float, "dt": float}
      /pluto/ekf_meas     (String) — JSON {"id": int, "range": float, "bearing": float}

    Publications:
      /pluto/ekf_state    (String) — JSON pose + covariance
      /pluto/ekf_markers  (MarkerArray) — covariance ellipse + ghost trail
    """

    def __init__(self):
        super().__init__('ekf_node')
        mu0 = np.array([0.0, 0.0, 0.0])
        sigma0 = np.diag([0.5, 0.5, 0.1])
        self._ekf = EKF(mu0, sigma0)
        self._ellipse_history = []  # for ghost trail

        self.pub_state = self.create_publisher(String, '/pluto/ekf_state', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/pluto/ekf_markers', 10)

        self.create_subscription(String, '/pluto/ekf_control', self._cb_control, 10)
        self.create_subscription(String, '/pluto/ekf_meas', self._cb_meas, 10)

        self.create_timer(0.1, self._publish_markers)
        self.get_logger().info('EKF Node ready.')

    def _cb_control(self, msg: String):
        data = json.loads(msg.data)
        self._ekf.predict(data['v'], data['omega'], data['dt'])

    def _cb_meas(self, msg: String):
        data = json.loads(msg.data)
        self._ekf.update(data['id'], data['range'], data['bearing'])

    def _publish_markers(self):
        state = String()
        x, y, theta = self._ekf.mu
        state.data = json.dumps({
            'x': x, 'y': y, 'theta': theta,
            'sigma_xx': float(self._ekf.sigma[0, 0]),
            'sigma_yy': float(self._ekf.sigma[1, 1]),
            'sigma_tt': float(self._ekf.sigma[2, 2]),
        })
        self.pub_state.publish(state)

        # Record ellipse for ghost trail
        ellipse = self._ekf.covariance_ellipse_2d()
        self._ellipse_history.append(ellipse.copy())
        if len(self._ellipse_history) > 20:
            self._ellipse_history.pop(0)

        markers = MarkerArray()
        marker_id = 0

        # Current covariance ellipse (LINE_STRIP)
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'ekf_ellipse'
        m.id = marker_id
        marker_id += 1
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.03
        m.color.r = 0.0
        m.color.g = 0.8
        m.color.b = 1.0
        m.color.a = 1.0
        for pt in np.vstack([ellipse, ellipse[0]]):
            p = Point()
            p.x, p.y, p.z = float(pt[0]), float(pt[1]), 0.05
            m.points.append(p)
        markers.markers.append(m)

        # Ghost trail (fading ellipses)
        for i, hist_ellipse in enumerate(self._ellipse_history[:-1]):
            m_ghost = Marker()
            m_ghost.header.frame_id = 'map'
            m_ghost.header.stamp = self.get_clock().now().to_msg()
            m_ghost.ns = 'ekf_trail'
            m_ghost.id = marker_id
            marker_id += 1
            m_ghost.type = Marker.LINE_STRIP
            m_ghost.action = Marker.ADD
            m_ghost.scale.x = 0.01
            alpha = (i + 1) / len(self._ellipse_history) * 0.4
            m_ghost.color.r = 0.5
            m_ghost.color.g = 0.5
            m_ghost.color.b = 1.0
            m_ghost.color.a = alpha
            for pt in np.vstack([hist_ellipse, hist_ellipse[0]]):
                p = Point()
                p.x, p.y, p.z = float(pt[0]), float(pt[1]), 0.05
                m_ghost.points.append(p)
            markers.markers.append(m_ghost)

        self.pub_markers.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = EKFNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
