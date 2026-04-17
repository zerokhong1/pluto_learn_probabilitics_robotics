"""
Chapter 5 — Odometry Motion Model.
Implements Table 5.6 (sample_motion_model_odometry) from Probabilistic Robotics.
Uses measured wheel encoder deltas rather than commanded velocities.
More accurate in practice because it uses actual (noisy) measurements.
"""

import numpy as np
from ..kalman_filters.ekf import normalize_angle


# Odometry noise parameters β1..β4
DEFAULT_BETAS = [0.05, 0.005, 0.005, 0.05]


def sample_motion_model_odometry(x: np.ndarray,
                                  odometry_bar: tuple[float, float, float],
                                  odometry: tuple[float, float, float],
                                  betas: list = DEFAULT_BETAS) -> np.ndarray:
    """
    Table 5.6 from Probabilistic Robotics.
    odometry_bar = (x_bar, y_bar, theta_bar) — previous raw odometry
    odometry     = (x', y', theta')          — current raw odometry
    Returns sampled next robot pose.
    """
    b1, b2, b3, b4 = betas
    x_bar, y_bar, theta_bar = odometry_bar
    x2, y2, theta2 = odometry

    delta_rot1 = normalize_angle(np.arctan2(y2 - y_bar, x2 - x_bar) - theta_bar)
    delta_trans = np.sqrt((x2 - x_bar)**2 + (y2 - y_bar)**2)
    delta_rot2 = normalize_angle(theta2 - theta_bar - delta_rot1)

    # Sample with noise
    delta_rot1_hat = delta_rot1 - np.random.normal(
        0, np.sqrt(b1 * delta_rot1**2 + b2 * delta_trans**2))
    delta_trans_hat = delta_trans - np.random.normal(
        0, np.sqrt(b3 * delta_trans**2 + b4 * (delta_rot1**2 + delta_rot2**2)))
    delta_rot2_hat = delta_rot2 - np.random.normal(
        0, np.sqrt(b1 * delta_rot2**2 + b2 * delta_trans**2))

    px, py, theta = x
    new_x = px + delta_trans_hat * np.cos(theta + delta_rot1_hat)
    new_y = py + delta_trans_hat * np.sin(theta + delta_rot1_hat)
    new_theta = normalize_angle(theta + delta_rot1_hat + delta_rot2_hat)
    return np.array([new_x, new_y, new_theta])


def motion_model_odometry(x_prime: np.ndarray, x: np.ndarray,
                           odometry_bar: tuple, odometry: tuple,
                           betas: list = DEFAULT_BETAS) -> float:
    """
    Returns p(x' | u_odom, x) — Table 5.5 style probability.
    """
    from scipy.stats import norm

    b1, b2, b3, b4 = betas
    x_bar, y_bar, theta_bar = odometry_bar
    x2, y2, theta2 = odometry

    delta_rot1 = normalize_angle(np.arctan2(y2 - y_bar, x2 - x_bar) - theta_bar)
    delta_trans = np.sqrt((x2 - x_bar)**2 + (y2 - y_bar)**2)
    delta_rot2 = normalize_angle(theta2 - theta_bar - delta_rot1)

    px, py, theta = x
    px2, py2, theta2_end = x_prime

    delta_rot1_hat = normalize_angle(np.arctan2(py2 - py, px2 - px) - theta)
    delta_trans_hat = np.sqrt((px2 - px)**2 + (py2 - py)**2)
    delta_rot2_hat = normalize_angle(theta2_end - theta - delta_rot1_hat)

    p1 = norm.pdf(delta_rot1 - delta_rot1_hat,
                  scale=np.sqrt(b1 * delta_rot1_hat**2 + b2 * delta_trans_hat**2))
    p2 = norm.pdf(delta_trans - delta_trans_hat,
                  scale=np.sqrt(b3 * delta_trans_hat**2 + b4 * (delta_rot1_hat**2 + delta_rot2_hat**2)))
    p3 = norm.pdf(delta_rot2 - delta_rot2_hat,
                  scale=np.sqrt(b1 * delta_rot2_hat**2 + b2 * delta_trans_hat**2))
    return float(p1 * p2 * p3)
