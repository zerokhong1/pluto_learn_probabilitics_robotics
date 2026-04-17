"""
Chapter 3 — Linear Kalman Filter.
State: x = [position, velocity]^T (1D rail scenario).
Implements Algorithm 3.1 from Probabilistic Robotics.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import PoseWithCovarianceStamped
from visualization_msgs.msg import Marker
import json


class KalmanFilter:
    """
    Linear Kalman Filter.

    State: x ∈ R^n
    Control: u ∈ R^l
    Measurement: z ∈ R^k

    Model:
        x_t = A x_{t-1} + B u_t + ε    ε ~ N(0, R)
        z_t = C x_t + δ                 δ ~ N(0, Q)
    """

    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray,
                 R: np.ndarray, Q: np.ndarray,
                 mu0: np.ndarray, sigma0: np.ndarray):
        self.A = A        # State transition matrix (n×n)
        self.B = B        # Control matrix (n×l)
        self.C = C        # Observation matrix (k×n)
        self.R = R        # Process noise covariance (n×n)
        self.Q = Q        # Measurement noise covariance (k×k)
        self.mu = mu0.copy()        # Mean (n,)
        self.sigma = sigma0.copy()  # Covariance (n×n)

    def predict(self, u: np.ndarray):
        """Algorithm 3.1 lines 2-3: prediction step."""
        self.mu = self.A @ self.mu + self.B @ u
        self.sigma = self.A @ self.sigma @ self.A.T + self.R

    def update(self, z: np.ndarray):
        """Algorithm 3.1 lines 4-6: correction step."""
        S = self.C @ self.sigma @ self.C.T + self.Q
        K = self.sigma @ self.C.T @ np.linalg.inv(S)
        innovation = z - self.C @ self.mu
        self.mu = self.mu + K @ innovation
        I = np.eye(len(self.mu))
        self.sigma = (I - K @ self.C) @ self.sigma

    def trace_sigma(self) -> float:
        return float(np.trace(self.sigma))


class KalmanFilter1DNode(Node):
    """
    ROS2 node: 1D rail localization with linear Kalman Filter.
    Publishes covariance ellipse as RViz2 marker (sphere scaled by std dev).
    """

    def __init__(self):
        super().__init__('kalman_filter_1d')
        dt = 0.1  # 10Hz

        # Model: x = [pos, vel], control = acceleration
        A = np.array([[1, dt], [0, 1]])
        B = np.array([[0.5 * dt**2], [dt]])
        C = np.array([[1, 0]])  # only position is observed
        R = np.diag([0.01, 0.1])  # process noise
        Q = np.array([[0.5]])      # GPS-like measurement noise (0.5m std)

        mu0 = np.array([0.0, 0.0])
        sigma0 = np.diag([1.0, 1.0])

        self._kf = KalmanFilter(A, B, C, R, Q, mu0, sigma0)

        self.pub_state = self.create_publisher(String, '/pluto/kf_state', 10)
        self.pub_marker = self.create_publisher(Marker, '/pluto/kf_uncertainty', 10)
        self.pub_pose = self.create_publisher(
            PoseWithCovarianceStamped, '/pluto/kf_pose', 10)

        self.create_subscription(String, '/pluto/kf_control', self._cb_control, 10)
        self.create_subscription(Float64MultiArray, '/pluto/gps_meas', self._cb_meas, 10)

        self.create_timer(dt, self._timer_cb)
        self._t = 0.0
        self.get_logger().info('Kalman Filter 1D ready.')

    def _cb_control(self, msg: String):
        data = json.loads(msg.data)
        u = np.array([float(data['accel'])])
        self._kf.predict(u)

    def _cb_meas(self, msg: Float64MultiArray):
        z = np.array(msg.data)
        self._kf.update(z)

    def _timer_cb(self):
        self._t += 0.1
        self._publish_state()

    def _publish_state(self):
        state = String()
        state.data = json.dumps({
            'pos': self._kf.mu[0],
            'vel': self._kf.mu[1],
            'sigma_pos': float(self._kf.sigma[0, 0]),
            'sigma_vel': float(self._kf.sigma[1, 1]),
            'trace': self._kf.trace_sigma(),
        })
        self.pub_state.publish(state)

        # Publish uncertainty ellipsoid marker (sphere scaled by 2σ)
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'kf_uncertainty'
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = self._kf.mu[0]
        m.pose.position.y = 0.0
        m.pose.position.z = 0.4
        two_sigma = 2.0 * float(np.sqrt(self._kf.sigma[0, 0]))
        m.scale.x = max(two_sigma, 0.05)
        m.scale.y = 0.3
        m.scale.z = 0.3

        # Eye color: red→yellow→green as uncertainty decreases
        trace = self._kf.trace_sigma()
        t = min(trace / 2.0, 1.0)  # normalize to [0,1]
        m.color.r = t
        m.color.g = 1.0 - t
        m.color.b = 0.0
        m.color.a = 0.7
        self.pub_marker.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = KalmanFilter1DNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
