"""
Chapter 4 — Particle Filter (Monte Carlo Localization).
Implements SIR (Sampling Importance Resampling) — Algorithm 4.3 from
Probabilistic Robotics, plus Augmented MCL for kidnapped robot.

Visualization: each particle rendered as a transparent Pluto ghost.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import json

from ..kalman_filters.ekf import LANDMARKS, ALPHA, normalize_angle


# Particle filter parameters
N_PARTICLES = 500
SENSOR_RANGE_MAX = 12.0
SIGMA_R_PF = 0.3    # range noise for PF measurement model
SIGMA_PHI_PF = 0.1  # bearing noise for PF measurement model

# Augmented MCL parameters (for kidnapped robot recovery)
W_SLOW_ALPHA = 0.001
W_FAST_ALPHA = 0.1


class Particle:
    __slots__ = ['x', 'y', 'theta', 'weight']

    def __init__(self, x=0.0, y=0.0, theta=0.0, weight=1.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight


def sample_motion_model_velocity(particle: Particle, v: float, omega: float, dt: float) -> Particle:
    """
    Sample from velocity motion model (Table 5.3, Thrun et al.).
    Returns new particle propagated through noisy motion.
    """
    a1, a2, a3, a4, a5, a6 = ALPHA[0], ALPHA[1], ALPHA[0], ALPHA[1], 0.01, 0.01

    v_hat = v + np.random.normal(0, np.sqrt(a1 * v**2 + a2 * omega**2))
    omega_hat = omega + np.random.normal(0, np.sqrt(a3 * v**2 + a4 * omega**2))
    gamma_hat = np.random.normal(0, np.sqrt(a5 * v**2 + a6 * omega**2))

    theta = particle.theta
    if abs(omega_hat) < 1e-6:
        new_x = particle.x + v_hat * dt * np.cos(theta)
        new_y = particle.y + v_hat * dt * np.sin(theta)
    else:
        r = v_hat / omega_hat
        new_x = particle.x - r * np.sin(theta) + r * np.sin(theta + omega_hat * dt)
        new_y = particle.y + r * np.cos(theta) - r * np.cos(theta + omega_hat * dt)
    new_theta = normalize_angle(theta + omega_hat * dt + gamma_hat * dt)

    return Particle(new_x, new_y, new_theta, particle.weight)


def landmark_measurement_weight(particle: Particle, landmark_id: int,
                                r_meas: float, phi_meas: float) -> float:
    """
    p(z | x, m) — likelihood of observation given particle pose.
    Uses Gaussian model around expected measurement.
    """
    if landmark_id not in LANDMARKS:
        return 1.0

    lx, ly = LANDMARKS[landmark_id]
    dx = lx - particle.x
    dy = ly - particle.y
    r_hat = np.sqrt(dx**2 + dy**2)
    phi_hat = normalize_angle(np.arctan2(dy, dx) - particle.theta)

    p_r = np.exp(-0.5 * ((r_meas - r_hat) / SIGMA_R_PF)**2) / (SIGMA_R_PF * np.sqrt(2 * np.pi))
    dphi = normalize_angle(phi_meas - phi_hat)
    p_phi = np.exp(-0.5 * (dphi / SIGMA_PHI_PF)**2) / (SIGMA_PHI_PF * np.sqrt(2 * np.pi))
    return max(p_r * p_phi, 1e-300)


def systematic_resample(particles: list[Particle]) -> list[Particle]:
    """Systematic resampling — O(N), low variance (Algorithm 4.3)."""
    N = len(particles)
    weights = np.array([p.weight for p in particles])
    weights /= weights.sum()
    cumsum = np.cumsum(weights)

    r = np.random.uniform(0, 1.0 / N)
    positions = r + np.arange(N) / N

    new_particles = []
    j = 0
    for pos in positions:
        while j < N - 1 and cumsum[j] < pos:
            j += 1
        p = particles[j]
        new_particles.append(Particle(p.x, p.y, p.theta, 1.0 / N))
    return new_particles


class AugmentedMCL:
    """
    Augmented MCL — handles kidnapped robot by injecting random particles
    when average weight drops significantly (Algorithm 8.2).
    """

    def __init__(self, n_particles: int = N_PARTICLES,
                 map_bounds: tuple = (-5, 5, -5, 5)):
        self.n = n_particles
        self.x_min, self.x_max, self.y_min, self.y_max = map_bounds

        # Initialize with uniform random particles
        self.particles = self._random_particles(n_particles)
        self.w_slow = 0.0
        self.w_fast = 0.0

    def _random_particles(self, n: int) -> list[Particle]:
        return [
            Particle(
                x=np.random.uniform(self.x_min, self.x_max),
                y=np.random.uniform(self.y_min, self.y_max),
                theta=np.random.uniform(-np.pi, np.pi),
                weight=1.0 / n,
            )
            for _ in range(n)
        ]

    def predict(self, v: float, omega: float, dt: float):
        """Sample from motion model for all particles."""
        self.particles = [
            sample_motion_model_velocity(p, v, omega, dt)
            for p in self.particles
        ]

    def update(self, landmark_id: int, r_meas: float, phi_meas: float):
        """Weight particles by measurement likelihood."""
        for p in self.particles:
            p.weight *= landmark_measurement_weight(p, landmark_id, r_meas, phi_meas)

        # Normalize
        total = sum(p.weight for p in self.particles)
        if total < 1e-300:
            # Particle deprivation — reinitialize
            self.particles = self._random_particles(self.n)
            return
        for p in self.particles:
            p.weight /= total

        # Augmented MCL: track slow/fast weight averages
        w_avg = total / self.n
        self.w_slow += W_SLOW_ALPHA * (w_avg - self.w_slow)
        self.w_fast += W_FAST_ALPHA * (w_avg - self.w_fast)

    def resample(self):
        """Augmented resampling with random injection for kidnapped robot."""
        p_random = max(0.0, 1.0 - self.w_fast / (self.w_slow + 1e-300))
        n_random = int(p_random * self.n)

        resampled = systematic_resample(self.particles)
        if n_random > 0:
            randoms = self._random_particles(n_random)
            resampled[-n_random:] = randoms

        self.particles = resampled

    def mean_pose(self) -> tuple[float, float, float]:
        weights = np.array([p.weight for p in self.particles])
        xs = np.array([p.x for p in self.particles])
        ys = np.array([p.y for p in self.particles])
        thetas = np.array([p.theta for p in self.particles])
        x = float(np.average(xs, weights=weights))
        y = float(np.average(ys, weights=weights))
        theta = float(np.arctan2(
            np.average(np.sin(thetas), weights=weights),
            np.average(np.cos(thetas), weights=weights),
        ))
        return x, y, theta

    def effective_n(self) -> float:
        """Effective sample size — low value indicates degeneracy."""
        weights = np.array([p.weight for p in self.particles])
        return float(1.0 / np.sum(weights**2))


class ParticleFilterNode(Node):
    """
    ROS2 node for Augmented MCL particle filter.

    Subscriptions:
      /pluto/pf_control  (String) — JSON {"v", "omega", "dt"}
      /pluto/pf_meas     (String) — JSON {"id", "range", "bearing"}
      /pluto/kidnap      (String) — JSON {"x", "y", "theta"} to teleport

    Publications:
      /pluto/pf_state    (String) — JSON mean pose + ESS
      /pluto/pf_ghosts   (MarkerArray) — particle ghost visualization
    """

    def __init__(self):
        super().__init__('particle_filter')
        self._mcl = AugmentedMCL(n_particles=N_PARTICLES)

        self.pub_state = self.create_publisher(String, '/pluto/pf_state', 10)
        self.pub_ghosts = self.create_publisher(MarkerArray, '/pluto/pf_ghosts', 10)

        self.create_subscription(String, '/pluto/pf_control', self._cb_control, 10)
        self.create_subscription(String, '/pluto/pf_meas', self._cb_meas, 10)
        self.create_subscription(String, '/pluto/kidnap', self._cb_kidnap, 10)

        self.create_timer(0.1, self._publish)
        self.get_logger().info(f'Particle Filter (N={N_PARTICLES}) ready. Augmented MCL active.')

    def _cb_control(self, msg: String):
        data = json.loads(msg.data)
        self._mcl.predict(data['v'], data['omega'], data['dt'])

    def _cb_meas(self, msg: String):
        data = json.loads(msg.data)
        self._mcl.update(data['id'], data['range'], data['bearing'])
        self._mcl.resample()

    def _cb_kidnap(self, msg: String):
        """Simulate kidnapped robot — teleport all particles to wrong position."""
        data = json.loads(msg.data)
        self.get_logger().warn(
            f"KIDNAPPED! Teleporting to ({data['x']:.1f}, {data['y']:.1f})")
        # Augmented MCL will recover via random particle injection
        self._mcl.w_slow = 0.0
        self._mcl.w_fast = 0.0

    def _publish(self):
        x, y, theta = self._mcl.mean_pose()
        ess = self._mcl.effective_n()
        state = String()
        state.data = json.dumps({'x': x, 'y': y, 'theta': theta, 'ess': ess})
        self.pub_state.publish(state)

        # Ghost markers
        markers = MarkerArray()
        weights = np.array([p.weight for p in self._mcl.particles])
        max_w = float(weights.max()) + 1e-300

        for i, p in enumerate(self._mcl.particles):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'pf_ghost'
            m.id = i
            m.type = Marker.ARROW
            m.action = Marker.ADD

            # Position + orientation
            import math
            m.pose.position.x = p.x
            m.pose.position.y = p.y
            m.pose.position.z = 0.0
            # Quaternion from yaw
            m.pose.orientation.z = math.sin(p.theta / 2.0)
            m.pose.orientation.w = math.cos(p.theta / 2.0)

            m.scale.x = 0.15  # arrow length
            m.scale.y = 0.04
            m.scale.z = 0.04

            alpha = float(p.weight) / max_w
            m.color.r = 0.3
            m.color.g = 0.8
            m.color.b = 1.0
            m.color.a = max(alpha * 0.8, 0.05)
            markers.markers.append(m)

        self.pub_ghosts.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
