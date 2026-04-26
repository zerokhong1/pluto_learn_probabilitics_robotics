#!/usr/bin/env python3
"""Subscribe to /limoncello/state (Odometry) and write TUM file."""
import sys, rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

class TumWriter(Node):
    def __init__(self, path):
        super().__init__('tum_writer')
        self.f = open(path, 'w')
        self.create_subscription(Odometry, '/limoncello/state', self.cb, 100)
        self.get_logger().info(f'Writing TUM to {path}')

    def cb(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.f.write(f'{t:.9f} {p.x:.6f} {p.y:.6f} {p.z:.6f} '
                     f'{q.x:.6f} {q.y:.6f} {q.z:.6f} {q.w:.6f}\n')
        self.f.flush()

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/traj.tum'
    rclpy.init()
    node = TumWriter(path)
    rclpy.spin(node)
    node.f.close()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
