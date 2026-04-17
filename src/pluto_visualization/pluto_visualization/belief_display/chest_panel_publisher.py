"""
Chest Panel Publisher — real-time belief histogram on Pluto's chest.
Subscribes to /pluto/belief_1d (Float64MultiArray) and renders it
as a bar chart MarkerArray at the chest panel position.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray

CHEST_X = 0.16   # offset from base_link to chest panel
CHEST_Z_BASE = 0.2
PANEL_HEIGHT = 0.18
N_BARS = 100


class ChestPanelPublisher(Node):
    def __init__(self):
        super().__init__('chest_panel_publisher')
        self._belief = np.ones(N_BARS) / N_BARS

        self.pub_panel = self.create_publisher(MarkerArray, '/pluto/chest_panel', 10)
        self.create_subscription(Float64MultiArray, '/pluto/belief_1d', self._cb_belief, 10)
        self.create_timer(0.1, self._publish)
        self.get_logger().info('Chest Panel Publisher ready.')

    def _cb_belief(self, msg: Float64MultiArray):
        arr = np.array(msg.data)
        if len(arr) == N_BARS:
            self._belief = arr

    def _publish(self):
        markers = MarkerArray()
        now = self.get_clock().now().to_msg()
        max_p = float(self._belief.max()) + 1e-10

        for i, p in enumerate(self._belief):
            m = Marker()
            m.header.frame_id = 'base_link'
            m.header.stamp = now
            m.ns = 'chest_bar'
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD

            # Lay out bars across the chest panel
            bar_w = 0.20 / N_BARS
            bar_h = float(p) / max_p * PANEL_HEIGHT

            m.pose.position.x = CHEST_X + 0.005
            m.pose.position.y = (i - N_BARS / 2) * bar_w
            m.pose.position.z = CHEST_Z_BASE + bar_h / 2

            m.scale.x = 0.005
            m.scale.y = bar_w * 0.9
            m.scale.z = max(bar_h, 0.002)

            t = float(p) / max_p
            m.color.r = t
            m.color.g = 0.3 * (1 - t)
            m.color.b = 1.0 - t
            m.color.a = 0.9
            markers.markers.append(m)

        self.pub_panel.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = ChestPanelPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
