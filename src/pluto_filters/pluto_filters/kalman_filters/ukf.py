"""
Chapter 3 — Unscented Kalman Filter (UKF).
Same 2D localization scenario as EKF but uses Unscented Transform.
Implements Algorithm 3.4 from Probabilistic Robotics.
Visualization: 2n+1 sigma-point "ghost Plutos" around the robot.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import json

from .ekf import LANDMARKS, ALPHA, SIGMA_R, SIGMA_PHI, normalize_angle


class UKF:
    """
    Unscented Kalman Filter for 2D robot localization.
    Uses UT parameters: α=1e-3, κ=0, β=2 (Gaussian assumption).
    """

    def __init__(self, mu0: np.ndarray, sigma0: np.ndarray,
                 alpha: float = 1e-3, kappa: float = 0.0, beta: float = 2.0):
        self.n = len(mu0)
        self.mu = mu0.copy()
        self.sigma = sigma0.copy()
        self.Q_meas = np.diag([SIGMA_R**2, SIGMA_PHI**2])

        # UT scaling parameters
        self._alpha = alpha
        self._kappa = kappa
        self._beta = beta
        self._lam = alpha**2 * (self.n + kappa) - self.n

        # Weights for mean and covariance
        n, lam = self.n, self._lam
        self.Wm = np.full(2 * n + 1, 0.5 / (n + lam))
        self.Wm[0] = lam / (n + lam)
        self.Wc = self.Wm.copy()
        self.Wc[0] = lam / (n + lam) + (1 - alpha**2 + beta)

    def _sigma_points(self) -> np.ndarray:
        """Compute 2n+1 sigma points."""
        n, lam = self.n, self._lam
        mat = (n + lam) * self.sigma
        # Regularize to ensure positive definiteness
        eps = 1e-6
        mat = mat + eps * np.eye(n)
        try:
            L = np.linalg.cholesky(mat)
        except np.linalg.LinAlgError:
            # Fallback: use eigendecomposition for near-singular matrices
            vals, vecs = np.linalg.eigh(mat)
            vals = np.maximum(vals, eps)
            L = vecs @ np.diag(np.sqrt(vals))
        pts = np.zeros((2 * n + 1, n))
        pts[0] = self.mu
        for i in range(n):
            pts[i + 1] = self.mu + L[:, i]
            pts[n + i + 1] = self.mu - L[:, i]
        return pts

    def _motion_model(self, state: np.ndarray, v: float, omega: float, dt: float) -> np.ndarray:
        x, y, theta = state
        if abs(omega) < 1e-6:
            dx = v * np.cos(theta) * dt
            dy = v * np.sin(theta) * dt
        else:
            dx = -(v / omega) * np.sin(theta) + (v / omega) * np.sin(theta + omega * dt)
            dy = (v / omega) * np.cos(theta) - (v / omega) * np.cos(theta + omega * dt)
        new = np.array([x + dx, y + dy, normalize_angle(theta + omega * dt)])
        return new

    def predict(self, v: float, omega: float, dt: float):
        """UKF prediction step (Algorithm 3.4 lines 2-4)."""
        sigma_pts = self._sigma_points()

        # Propagate through motion model
        propagated = np.array([self._motion_model(pt, v, omega, dt) for pt in sigma_pts])

        # Predicted mean
        mu_bar = np.sum(self.Wm[:, None] * propagated, axis=0)
        mu_bar[2] = normalize_angle(mu_bar[2])

        # Predicted covariance
        a1, a2, a3, a4 = ALPHA
        R = np.diag([a1 * v**2 + a2 * omega**2,
                     a1 * v**2 + a2 * omega**2,
                     a3 * v**2 + a4 * omega**2])

        sigma_bar = R.copy()
        for i, pt in enumerate(propagated):
            diff = pt - mu_bar
            diff[2] = normalize_angle(diff[2])
            sigma_bar += self.Wc[i] * np.outer(diff, diff)

        self.mu = mu_bar
        self.sigma = sigma_bar
        self._last_sigma_pts = propagated  # for visualization

    def update(self, landmark_id: int, r_meas: float, phi_meas: float):
        """UKF correction step (Algorithm 3.4 lines 5-9)."""
        if landmark_id not in LANDMARKS:
            return

        lx, ly = LANDMARKS[landmark_id]
        sigma_pts = self._sigma_points()

        # Measurement predictions at each sigma point
        def h(state):
            x, y, theta = state
            dx, dy = lx - x, ly - y
            return np.array([np.sqrt(dx**2 + dy**2),
                             normalize_angle(np.arctan2(dy, dx) - theta)])

        z_hat_pts = np.array([h(pt) for pt in sigma_pts])
        z_hat = np.sum(self.Wm[:, None] * z_hat_pts, axis=0)

        # Innovation covariance S
        S = self.Q_meas.copy()
        for i, zpt in enumerate(z_hat_pts):
            diff = zpt - z_hat
            diff[1] = normalize_angle(diff[1])
            S += self.Wc[i] * np.outer(diff, diff)

        # Cross-covariance T
        T = np.zeros((self.n, 2))
        for i, (xpt, zpt) in enumerate(zip(sigma_pts, z_hat_pts)):
            dx = xpt - self.mu
            dx[2] = normalize_angle(dx[2])
            dz = zpt - z_hat
            dz[1] = normalize_angle(dz[1])
            T += self.Wc[i] * np.outer(dx, dz)

        K = T @ np.linalg.inv(S)
        innovation = np.array([r_meas - z_hat[0], normalize_angle(phi_meas - z_hat[1])])
        self.mu = self.mu + K @ innovation
        self.mu[2] = normalize_angle(self.mu[2])
        self.sigma = self.sigma - K @ S @ K.T

    def get_sigma_points(self) -> np.ndarray:
        return self._sigma_points()


class UKFNode(Node):
    """
    ROS2 node for UKF 2D localization.
    Renders 2n+1 sigma points as small ghost markers in RViz2.
    """

    def __init__(self):
        super().__init__('ukf_node')
        mu0 = np.array([0.0, 0.0, 0.0])
        sigma0 = np.diag([0.5, 0.5, 0.1])
        self._ukf = UKF(mu0, sigma0)

        self.pub_state = self.create_publisher(String, '/pluto/ukf_state', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/pluto/ukf_markers', 10)

        self.create_subscription(String, '/pluto/ukf_control', self._cb_control, 10)
        self.create_subscription(String, '/pluto/ukf_meas', self._cb_meas, 10)

        self.create_timer(0.1, self._publish)
        self.get_logger().info('UKF Node ready (2n+1 sigma-point visualization).')

    def _cb_control(self, msg: String):
        data = json.loads(msg.data)
        self._ukf.predict(data['v'], data['omega'], data['dt'])

    def _cb_meas(self, msg: String):
        data = json.loads(msg.data)
        self._ukf.update(data['id'], data['range'], data['bearing'])

    def _publish(self):
        x, y, theta = self._ukf.mu
        state = String()
        state.data = json.dumps({'x': x, 'y': y, 'theta': theta})
        self.pub_state.publish(state)

        # Sigma point ghost markers
        sigma_pts = self._ukf.get_sigma_points()
        markers = MarkerArray()
        for i, pt in enumerate(sigma_pts):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'ukf_sigma'
            m.id = i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(pt[0])
            m.pose.position.y = float(pt[1])
            m.pose.position.z = 0.1
            m.scale.x = 0.12
            m.scale.y = 0.12
            m.scale.z = 0.05

            if i == 0:  # Central sigma point = mean
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 1.0, 0.0, 0.9
            elif i <= self._ukf.n:  # + direction
                m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.8, 1.0, 0.6
            else:  # - direction
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.4, 0.1, 0.6
            markers.markers.append(m)

        self.pub_markers.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = UKFNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
