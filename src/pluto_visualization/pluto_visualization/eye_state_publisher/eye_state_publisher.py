"""
Pluto Eye State Publisher.
Reads filter uncertainty (trace of Σ) and changes Pluto's eye color:
  - Blue:  uncertainty low  (confident)
  - Yellow: medium uncertainty (re-localizing)
  - Red:   uncertainty high (lost)
Publishes colored sphere markers to /pluto/eye_marker.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
import json
import math


UNCERTAINTY_LOW = 0.1   # trace(Σ) threshold for "confident" (blue)
UNCERTAINTY_HIGH = 2.0  # trace(Σ) threshold for "lost" (red)


def uncertainty_to_color(trace: float) -> tuple[float, float, float]:
    """Maps trace(Σ) to RGB. Low→blue, mid→yellow, high→red."""
    t = min(max((trace - UNCERTAINTY_LOW) / (UNCERTAINTY_HIGH - UNCERTAINTY_LOW), 0.0), 1.0)
    if t < 0.5:
        # blue → yellow
        s = t * 2
        return s, s, 1.0 - s
    else:
        # yellow → red
        s = (t - 0.5) * 2
        return 1.0, 1.0 - s, 0.0


class EyeStatePublisher(Node):
    """
    Subscribes to filter state topics and publishes eye color markers.
    Merges signals from KF, EKF, UKF, PF — takes the minimum (most optimistic).
    """

    def __init__(self):
        super().__init__('eye_state_publisher')
        self._uncertainty = {'kf': 1.0, 'ekf': 1.0, 'ukf': 1.0, 'pf': 1.0}

        self.pub_eyes = self.create_publisher(MarkerArray, '/pluto/eye_marker', 10)

        self.create_subscription(String, '/pluto/kf_state', self._cb_kf, 10)
        self.create_subscription(String, '/pluto/ekf_state', self._cb_ekf, 10)
        self.create_subscription(String, '/pluto/ukf_state', self._cb_ukf, 10)
        self.create_subscription(String, '/pluto/pf_state', self._cb_pf, 10)

        self.create_timer(0.1, self._publish_eyes)
        self.get_logger().info('Eye State Publisher ready.')

    def _cb_kf(self, msg: String):
        d = json.loads(msg.data)
        self._uncertainty['kf'] = d.get('trace', 1.0)

    def _cb_ekf(self, msg: String):
        d = json.loads(msg.data)
        sigma_sum = d.get('sigma_xx', 1.0) + d.get('sigma_yy', 1.0) + d.get('sigma_tt', 0.1)
        self._uncertainty['ekf'] = sigma_sum

    def _cb_ukf(self, msg: String):
        self._uncertainty['ukf'] = self._uncertainty['ekf']  # same scenario

    def _cb_pf(self, msg: String):
        d = json.loads(msg.data)
        ess = d.get('ess', 1.0)
        # Map ESS → uncertainty: low ESS = high uncertainty
        self._uncertainty['pf'] = max(0.0, 2.0 - ess / 250.0)

    def _publish_eyes(self):
        # Use minimum uncertainty across active filters
        u = min(self._uncertainty.values())
        r, g, b = uncertainty_to_color(u)

        markers = MarkerArray()
        now = self.get_clock().now().to_msg()

        for i, (x, y) in enumerate([(0.10, 0.05), (0.10, -0.05)]):
            m = Marker()
            m.header.frame_id = 'head_link'
            m.header.stamp = now
            m.ns = 'pluto_eye'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = 0.07
            m.scale.x = m.scale.y = m.scale.z = 0.05
            m.color.r = r
            m.color.g = g
            m.color.b = b
            m.color.a = 1.0
            markers.markers.append(m)

        self.pub_eyes.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = EyeStatePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
