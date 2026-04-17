"""
Auto-drive node for Gazebo demo.
Publishes /cmd_vel so Pluto drives forward and bounces back automatically.
Fixes Issue #2 from TROUBLESHOOTING.md: no cmd_vel publisher → robot sits still.
"""
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


FORWARD_SPEED = 0.3   # m/s
TURN_SPEED    = 1.5   # rad/s
X_FAR         = 9.5   # m — turn around near far wall
X_NEAR        = 0.3   # m — turn around near start


class AutoDrive(Node):
    def __init__(self):
        super().__init__('auto_drive')

        self._pub  = self.create_publisher(Twist, '/cmd_vel', 10)
        self._sub  = self.create_subscription(Odometry, '/odom', self._odom_cb, 10)

        self._x         = 0.0
        self._direction = 1.0    # +1 forward, -1 backward
        self._turning   = False
        self._turn_accum = 0.0
        self._dt        = 0.1

        self.create_timer(self._dt, self._tick)
        self.get_logger().info('Auto-drive started — Pluto will drive forward automatically.')

    def _odom_cb(self, msg: Odometry):
        self._x = msg.pose.pose.position.x

    def _tick(self):
        twist = Twist()

        if self._turning:
            twist.angular.z = TURN_SPEED * self._direction
            self._turn_accum += abs(twist.angular.z) * self._dt
            if self._turn_accum >= math.pi:
                self._turning    = False
                self._turn_accum = 0.0
                self._direction  = -self._direction
                self.get_logger().info(
                    f'Pluto turned around → driving {"forward" if self._direction > 0 else "back"}'
                )
        elif self._direction > 0 and self._x >= X_FAR:
            self._turning    = True
            self._turn_accum = 0.0
        elif self._direction < 0 and self._x <= X_NEAR:
            self._turning    = True
            self._turn_accum = 0.0
        else:
            twist.linear.x = FORWARD_SPEED

        self._pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = AutoDrive()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
