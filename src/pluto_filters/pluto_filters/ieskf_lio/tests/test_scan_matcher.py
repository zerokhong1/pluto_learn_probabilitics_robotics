"""Unit tests for scan_matcher.py."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from pluto_filters.ieskf_lio.scan_matcher import ScanMatcher
from pluto_filters.ieskf_lio.se2_manifold import Exp

RNG = np.random.default_rng(7)


def _wall_map(length=10.0, n=200, y=1.0) -> np.ndarray:
    """Horizontal wall at y=±wall_y."""
    xs = np.linspace(0, length, n)
    top    = np.stack([xs, np.full(n,  y)], axis=1)
    bottom = np.stack([xs, np.full(n, -y)], axis=1)
    return np.vstack([top, bottom])


def _scan_from_pose(pose: np.ndarray, wall_y=1.0,
                    n_rays=36, range_max=5.0) -> np.ndarray:
    """Simulate a perfect 2D scan (rays perpendicular to walls only for simplicity)."""
    R = pose[:2, :2]
    t = pose[:2, 2]
    angles = np.linspace(-np.pi, np.pi, n_rays, endpoint=False)
    pts = []
    for a in angles:
        d = np.array([np.cos(a), np.sin(a)])
        # Intersect with top wall y = wall_y
        # t[1] + r*d[1] = wall_y  →  r = (wall_y - t[1]) / d[1]
        for sign in [1.0, -1.0]:
            wy = sign * wall_y
            if abs(d[1]) > 1e-6:
                r = (wy - t[1]) / d[1]
                if 0 < r < range_max:
                    # point in world frame
                    wp = t + r * d
                    # back-transform to robot frame
                    p_robot = R.T @ (wp - t)
                    pts.append(p_robot)
                    break
    return np.array(pts) if pts else np.zeros((0, 2))


# ── test 1: perfect alignment → residuals ≈ 0 ────────────────────────────────

def test_perfect_alignment():
    sm = ScanMatcher(k_neighbors=5, max_dist=3.0)
    pose = np.eye(3)
    wall = _wall_map()
    sm.set_map(wall)

    scan = _scan_from_pose(pose)
    z, H, ok = sm.compute_residuals_and_jacobians(scan, pose)

    assert ok, "Should find correspondences for perfect alignment"
    assert np.max(np.abs(z)) < 0.05, f"Max residual={np.max(np.abs(z)):.4f} for perfect alignment"


# ── test 2: known x-offset → residuals reflect shift ─────────────────────────

def test_known_offset():
    sm = ScanMatcher(k_neighbors=5, max_dist=3.0)
    wall = _wall_map()
    sm.set_map(wall)

    # y-offset is OBSERVABLE in a hallway (perpendicular to walls).
    # x-offset alone is NOT observable (walls are horizontal → normals all ±y).
    pose_true = Exp(np.array([0.0, 0.05, 0.0]))   # shifted 5cm in y
    pose_est  = np.eye(3)

    scan = _scan_from_pose(pose_true)
    z, H, ok = sm.compute_residuals_and_jacobians(scan, pose_est)

    assert ok, "Should find correspondences"
    assert np.max(np.abs(z)) > 1e-4, "Residuals should be non-zero for y-offset pose"


# ── test 3: known rotation → residuals reflect angular error ─────────────────

def test_known_rotation():
    sm = ScanMatcher(k_neighbors=5, max_dist=3.0)
    wall = _wall_map()
    sm.set_map(wall)

    angle_deg = 5.0
    pose_true = Exp(np.array([0.0, 0.0, np.deg2rad(angle_deg)]))
    pose_est  = np.eye(3)

    scan = _scan_from_pose(pose_true)
    if len(scan) == 0:
        pytest.skip("Scan empty for this geometry")
    z, H, ok = sm.compute_residuals_and_jacobians(scan, pose_est)

    assert ok, "Should find correspondences"
    assert np.max(np.abs(z)) > 1e-4, "Residuals should be non-zero for rotated scan"


# ── test 4: straight corridor → only perpendicular DoF constrained ────────────

def test_degenerate_corridor():
    """Long straight corridor: normal is always ±y → x (along-corridor) is unobservable."""
    sm = ScanMatcher(k_neighbors=5, max_dist=3.0)
    # Pure horizontal walls, no features
    xs = np.linspace(-25, 25, 500)
    top    = np.stack([xs, np.full(500,  1.5)], axis=1)
    bottom = np.stack([xs, np.full(500, -1.5)], axis=1)
    sm.set_map(np.vstack([top, bottom]))

    pose = np.eye(3)
    scan = _scan_from_pose(pose, wall_y=1.5)
    if len(scan) == 0:
        pytest.skip("Scan empty")
    z, H, ok = sm.compute_residuals_and_jacobians(scan, pose)

    if not ok:
        pytest.skip("Not enough correspondences for corridor test")

    # In a pure corridor, all normals point in ±y direction.
    # H[:, 0] = n_x ≈ 0, so along-corridor (x) column is near zero.
    assert np.max(np.abs(H[:, 0])) < 0.1, (
        f"x-column of H should be near 0 in a corridor, got max={np.max(np.abs(H[:,0])):.4f}"
    )


# ── test 5: voxel downsample reduces point count ─────────────────────────────

def test_voxel_downsample():
    pts = RNG.uniform(-5, 5, (1000, 2))
    ds  = ScanMatcher.voxel_downsample(pts, res=0.5)
    assert len(ds) < len(pts), "Downsampling should reduce point count"
    assert len(ds) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
