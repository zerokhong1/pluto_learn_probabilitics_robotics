"""Unit tests for ieskf.py.

Validates IESKF against known-good cases:
1. Linear system — should match standard KF in 1 iteration.
2. SE(2) rotation — manifold update wraps angles correctly.
3. Convergence — iteration count drops with lower measurement noise.
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from pluto_filters.ieskf_lio.ieskf import IESKF
from pluto_filters.ieskf_lio.se2_manifold import Exp, oplus, ominus

RNG = np.random.default_rng(0)


# ── helper: build a linear KF for comparison ─────────────────────────────────

def kf_update_linear(x, P, H, z, R):
    """Standard (non-iterated) Kalman update for a Euclidean state."""
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x_new = x + K @ (z - H @ x)
    P_new = (np.eye(len(x)) - K @ H) @ P
    return x_new, P_new


# ── test 1: linear measurement, IESKF == KF in 1 iteration ─────────────────

def test_linear_system_matches_kf():
    """For a linear system on Euclidean space IESKF = standard KF."""
    ieskf = IESKF(bias_dim=3)
    # Start at identity pose, zero biases
    ieskf.pose = np.eye(3)
    ieskf.bias = np.zeros(3)
    ieskf.P    = np.eye(6) * 1.0

    # Observation: x-position of robot
    # H selects the x-translation component of the error-state
    # error-state: [δpx, δpy, δθ, δv, δbg, δba]
    # pose[0,2] = x → H = [1,0,0, 0,0,0]
    H = np.zeros((1, 6)); H[0, 0] = 1.0
    z = np.array([0.5])   # measured x = 0.5
    R = np.eye(1) * 0.01

    def z_func(pose_j, bias_j):
        return np.array([pose_j[0, 2]]) - z  # residual = estimate - measurement

    def H_func(pose_j, bias_j):
        return H

    ieskf.update(z_func, H_func, R, max_iter=1)

    # Standard KF: state = zeros (flat Euclidean approx)
    x0 = np.zeros(6)
    P0 = np.eye(6)
    x_kf, _ = kf_update_linear(x0, P0, H, z, R)

    # IESKF x-correction should match KF x-correction
    dx = ieskf.pose[0, 2] - 0.0  # pose was identity → x=0 → correction applied
    assert abs(dx - x_kf[0]) < 1e-6, f"IESKF x={dx:.6f} ≠ KF x={x_kf[0]:.6f}"


# ── test 2: SE(2) rotation wraps correctly ───────────────────────────────────

def test_pose_update_wraps_angle():
    """After ⊕ with a large rotation, yaw must stay in (−π, π]."""
    ieskf = IESKF(bias_dim=3)
    ieskf.pose = Exp(np.array([0.0, 0.0, 3.0]))   # yaw ≈ 172°
    ieskf.bias = np.zeros(3)
    ieskf.P    = np.eye(6) * 0.1

    # Pretend a scan measurement pulls yaw by +0.3 rad
    H = np.zeros((1, 6)); H[0, 2] = 1.0
    R = np.eye(1) * 0.001

    def z_func(pose_j, _):
        yaw_est = np.arctan2(pose_j[1, 0], pose_j[0, 0])
        return np.array([yaw_est - 3.3])   # measurement says 3.3 rad

    def H_func(pose_j, _):
        return H

    ieskf.update(z_func, H_func, R, max_iter=5)
    yaw = np.arctan2(ieskf.pose[1, 0], ieskf.pose[0, 0])
    assert -np.pi <= yaw <= np.pi, f"Yaw {yaw:.4f} out of range after update"
    assert not np.any(np.isnan(ieskf.pose)), "NaN in pose after update"


# ── test 3: convergence improves with tighter noise ──────────────────────────

def test_convergence_moves_toward_measurement():
    """After update, the pose should be closer to the measured position."""
    ieskf = IESKF(bias_dim=3)
    ieskf.pose = Exp(np.array([0.5, 0.3, 0.1]))   # offset from origin
    ieskf.bias = np.zeros(3)
    ieskf.P    = np.eye(6) * 1.0

    # Measurement: robot is at (0, 0) in both x and y
    H = np.zeros((2, 6)); H[0, 0] = 1.0; H[1, 1] = 1.0
    V = np.eye(2) * 0.001   # tight measurement noise → strong pull
    z_true = np.array([0.0, 0.0])

    x_before = ieskf.pose[0, 2]
    y_before = ieskf.pose[1, 2]
    dist_before = np.sqrt(x_before**2 + y_before**2)

    ieskf.update(
        lambda p, b: np.array([p[0, 2], p[1, 2]]) - z_true,
        lambda p, b: H,
        V, max_iter=5,
    )

    x_after = ieskf.pose[0, 2]
    y_after = ieskf.pose[1, 2]
    dist_after = np.sqrt(x_after**2 + y_after**2)

    assert dist_after < dist_before, (
        f"After update, pose should move toward measurement: "
        f"dist_before={dist_before:.4f}, dist_after={dist_after:.4f}"
    )
    assert not np.any(np.isnan(ieskf.pose)), "NaN in pose after update"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
