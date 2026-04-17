"""
Chapter 6 — Likelihood Field Measurement Model.
Precomputes distance transform of occupancy map, then evaluates
p(z | x, m) as Gaussian over distance to nearest obstacle.
~10x faster than beam model at runtime.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import json


SIGMA_HIT = 0.2   # std dev for distance Gaussian [m]
Z_HIT = 0.9
Z_RAND = 0.1
Z_MAX = 12.0


class LikelihoodField:
    """
    Precomputed likelihood field from occupancy grid.
    p(z | x, m) ≈ z_hit * N(dist; 0, σ) + z_rand / z_max
    """

    def __init__(self, occupancy_grid: np.ndarray, resolution: float, origin: tuple):
        self.resolution = resolution  # [m/cell]
        self.origin_x, self.origin_y = origin
        self._build_field(occupancy_grid)

    def _build_field(self, occ: np.ndarray):
        """Build distance transform from obstacle cells."""
        obstacle_mask = occ > 50  # cells with >50% occupancy
        dist_cells = distance_transform_edt(~obstacle_mask)
        self.dist_m = dist_cells * self.resolution

        # Likelihood field: Gaussian over distance to nearest obstacle
        self.likelihood = (
            Z_HIT * np.exp(-0.5 * (self.dist_m / SIGMA_HIT)**2) +
            Z_RAND / Z_MAX
        )

    def world_to_cell(self, wx: float, wy: float) -> tuple[int, int]:
        cx = int((wx - self.origin_x) / self.resolution)
        cy = int((wy - self.origin_y) / self.resolution)
        return cx, cy

    def query(self, wx: float, wy: float) -> float:
        """Returns likelihood at world coordinate (wx, wy)."""
        cx, cy = self.world_to_cell(wx, wy)
        h, w = self.likelihood.shape
        if 0 <= cx < w and 0 <= cy < h:
            return float(self.likelihood[cy, cx])
        return float(Z_RAND / Z_MAX)

    def measurement_log_likelihood(self, robot_pose: np.ndarray,
                                    scan_ranges: np.ndarray,
                                    scan_angles: np.ndarray) -> float:
        """
        Compute log p(z | x, m) for full LaserScan.
        robot_pose = [x, y, theta]
        """
        rx, ry, rtheta = robot_pose
        log_p = 0.0
        for z, angle in zip(scan_ranges, scan_angles):
            if z >= Z_MAX or z <= 0.0:
                continue
            bx = rx + z * np.cos(rtheta + angle)
            by = ry + z * np.sin(rtheta + angle)
            log_p += np.log(max(self.query(bx, by), 1e-300))
        return log_p


class LikelihoodFieldNode(Node):
    """
    ROS2 node: likelihood field visualization + scan evaluation.
    Subscribes to occupancy map, builds likelihood field,
    evaluates particle weights via likelihood field model.
    """

    def __init__(self):
        super().__init__('likelihood_field_node')
        self._lf: LikelihoodField | None = None

        self.pub_markers = self.create_publisher(MarkerArray, '/pluto/lf_scan_markers', 10)
        self.pub_score = self.create_publisher(String, '/pluto/lf_score', 10)

        self.create_subscription(OccupancyGrid, '/map', self._cb_map, 10)
        self.create_subscription(LaserScan, '/scan', self._cb_scan, 10)
        self.create_subscription(String, '/pluto/robot_pose', self._cb_pose, 10)

        self._pose = np.array([0.0, 0.0, 0.0])
        self.get_logger().info('Likelihood Field Node ready.')

    def _cb_map(self, msg: OccupancyGrid):
        w, h = msg.info.width, msg.info.height
        occ = np.array(msg.data, dtype=np.int8).reshape((h, w))
        res = msg.info.resolution
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y
        self._lf = LikelihoodField(occ, res, (ox, oy))
        self.get_logger().info(f'Likelihood field built ({w}×{h} cells, res={res:.3f}m)')

    def _cb_pose(self, msg: String):
        data = json.loads(msg.data)
        self._pose = np.array([data['x'], data['y'], data['theta']])

    def _cb_scan(self, msg: LaserScan):
        if self._lf is None:
            return

        n = len(msg.ranges)
        z = np.clip(np.array(msg.ranges, dtype=float), 0, Z_MAX)
        angles = np.linspace(msg.angle_min, msg.angle_max, n)

        log_ll = self._lf.measurement_log_likelihood(self._pose, z, angles)

        score = String()
        score.data = json.dumps({'log_likelihood': log_ll})
        self.pub_score.publish(score)

        # Visualize: each beam endpoint colored by likelihood
        rx, ry, rtheta = self._pose
        markers = MarkerArray()
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = msg.header.stamp
        m.ns = 'lf_dots'
        m.id = 0
        m.type = Marker.POINTS
        m.action = Marker.ADD
        m.scale.x = 0.06
        m.scale.y = 0.06

        from std_msgs.msg import ColorRGBA
        for zi, angle in zip(z, angles):
            if zi >= Z_MAX:
                continue
            bx = rx + zi * np.cos(rtheta + angle)
            by = ry + zi * np.sin(rtheta + angle)
            lk = self._lf.query(bx, by)

            p = Point()
            p.x, p.y, p.z = bx, by, 0.05
            m.points.append(p)

            c = ColorRGBA()
            # Hot colormap: low=blue, high=red
            c.r = float(lk)
            c.g = 0.2
            c.b = float(1.0 - lk)
            c.a = 0.8
            m.colors.append(c)

        markers.markers.append(m)
        self.pub_markers.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = LikelihoodFieldNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
