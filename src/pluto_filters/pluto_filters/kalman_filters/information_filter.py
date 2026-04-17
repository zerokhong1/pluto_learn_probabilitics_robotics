"""
Chapter 3 — Information Filter (dual form of Kalman Filter).
Represents belief as (Ω, ξ) where Ω = Σ⁻¹ (information matrix)
and ξ = Ω μ (information vector).
Implements Algorithm 3.5 from Probabilistic Robotics.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json


class InformationFilter:
    """
    Linear Information Filter.
    Dual of the Kalman Filter — works with information matrix Ω = Σ⁻¹.
    Useful for multi-robot / GraphSLAM (preview of Chapter 11).
    """

    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray,
                 R: np.ndarray, Q: np.ndarray,
                 Omega0: np.ndarray, xi0: np.ndarray):
        self.A = A
        self.B = B
        self.C = C
        self.R = R          # Process noise covariance
        self.Q = Q          # Measurement noise covariance
        self.Q_inv = np.linalg.inv(Q)
        self.Omega = Omega0.copy()   # Information matrix
        self.xi = xi0.copy()         # Information vector

    @property
    def mu(self) -> np.ndarray:
        return np.linalg.solve(self.Omega, self.xi)

    @property
    def sigma(self) -> np.ndarray:
        return np.linalg.inv(self.Omega)

    def predict(self, u: np.ndarray):
        """
        Information Filter prediction (Algorithm 3.5 lines 2-3).
        Note: prediction is more expensive than KF because we need Σ.
        """
        sigma = self.sigma
        Phi = np.linalg.inv(self.A @ sigma @ self.A.T + self.R)
        self.Omega = Phi
        self.xi = Phi @ (self.A @ self.mu + self.B @ u)

    def update(self, z: np.ndarray):
        """
        Information Filter correction (Algorithm 3.5 lines 4-5).
        Update is simple in information space: just add C^T Q^-1 C.
        """
        self.Omega = self.Omega + self.C.T @ self.Q_inv @ self.C
        self.xi = self.xi + self.C.T @ self.Q_inv @ z

    def information_gain(self) -> float:
        return float(np.trace(self.Omega))


class InformationFilterNode(Node):
    """
    ROS2 node for Information Filter (1D, same scenario as KF for comparison).
    Publishes both Σ and Ω traces to illustrate duality.
    """

    def __init__(self):
        super().__init__('information_filter')
        dt = 0.1

        A = np.array([[1, dt], [0, 1]])
        B = np.array([[0.5 * dt**2], [dt]])
        C = np.array([[1, 0]])
        R = np.diag([0.01, 0.1])
        Q = np.array([[0.5]])

        # Start with low information (high uncertainty)
        sigma0 = np.diag([100.0, 100.0])
        Omega0 = np.linalg.inv(sigma0)
        xi0 = Omega0 @ np.array([0.0, 0.0])

        self._if = InformationFilter(A, B, C, R, Q, Omega0, xi0)

        self.pub_state = self.create_publisher(String, '/pluto/if_state', 10)
        self.create_subscription(String, '/pluto/if_control', self._cb_control, 10)
        self.create_subscription(String, '/pluto/if_meas', self._cb_meas, 10)
        self.create_timer(dt, self._publish)
        self.get_logger().info('Information Filter ready.')

    def _cb_control(self, msg: String):
        data = json.loads(msg.data)
        u = np.array([float(data['accel'])])
        self._if.predict(u)

    def _cb_meas(self, msg: String):
        data = json.loads(msg.data)
        z = np.array([float(data['pos'])])
        self._if.update(z)

    def _publish(self):
        state = String()
        state.data = json.dumps({
            'pos': float(self._if.mu[0]),
            'vel': float(self._if.mu[1]),
            'trace_sigma': float(np.trace(self._if.sigma)),
            'trace_omega': float(np.trace(self._if.Omega)),
            'info_gain': self._if.information_gain(),
        })
        self.pub_state.publish(state)


def main(args=None):
    rclpy.init(args=args)
    node = InformationFilterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
