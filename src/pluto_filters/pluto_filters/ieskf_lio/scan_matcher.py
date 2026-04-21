"""
Point-to-line scan matching for 2D LiDAR.

2D analog of LIMOncello's point-to-plane residual (paper eq. 17).
In 2D, planes become lines; normals are 2D vectors.

Reference: LIMOncello paper eq. (17)-(18), adapted to SE(2).
"""

import numpy as np
from scipy.spatial import KDTree


class ScanMatcher:
    """Point-to-line ICP-style scan matcher.

    Residual for point pᵢ (robot frame), pose T = (R, t):
        h_i(T) = nᵢᵀ (R pᵢ + t − qᵢ)

    where qᵢ is the centroid of k nearest map neighbors,
    nᵢ is the line normal (eigenvector of smallest eigenvalue).

    Jacobian w.r.t. error state [δx, δy, δθ]:
        ∂hᵢ/∂δ = nᵢᵀ [I₂ | d(R pᵢ)/dθ]
               = [nᵢ[0], nᵢ[1], nᵢᵀ [−(Rpᵢ)y, (Rpᵢ)x]]
    """

    def __init__(self, k_neighbors: int = 5, max_dist: float = 2.0,
                 planarity_thresh: float = 0.3):
        """
        Args:
            k_neighbors: number of nearest neighbors for line fitting (≥2).
            max_dist: maximum correspondence distance; reject farther points.
            planarity_thresh: reject if λ_min/λ_max > this (not a clean line).
        """
        self.k = k_neighbors
        self.max_dist = max_dist
        self.planarity_thresh = planarity_thresh

        self.map_points: np.ndarray | None = None
        self.map_tree:   KDTree | None = None

    # ── map management ──────────────────────────────────────────────────────

    def set_map(self, points: np.ndarray):
        """Set map from (N, 2) world-frame points."""
        self.map_points = points.copy()
        self.map_tree = KDTree(self.map_points)

    def add_to_map(self, points: np.ndarray):
        """Append (M, 2) world-frame points and rebuild KD-tree."""
        if self.map_points is None:
            self.set_map(points)
        else:
            self.map_points = np.vstack([self.map_points, points])
            self.map_tree = KDTree(self.map_points)

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def polar_to_cart(ranges: np.ndarray, angles: np.ndarray,
                      range_min: float = 0.1,
                      range_max: float = 30.0) -> np.ndarray:
        """Convert polar LaserScan arrays to (M, 2) robot-frame points."""
        valid = (ranges > range_min) & (ranges < range_max) & np.isfinite(ranges)
        r = ranges[valid]
        a = angles[valid]
        return np.stack([r * np.cos(a), r * np.sin(a)], axis=1)

    @staticmethod
    def voxel_downsample(points: np.ndarray, res: float) -> np.ndarray:
        """Voxel grid downsample. Keeps one point per res×res cell."""
        if len(points) == 0:
            return points
        keys = np.floor(points / res).astype(int)
        _, idx = np.unique(keys, axis=0, return_index=True)
        return points[np.sort(idx)]

    @staticmethod
    def transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """Apply SE(2) pose matrix to (M, 2) points → world frame."""
        R = pose[:2, :2]
        t = pose[:2, 2]
        return (R @ points.T).T + t

    # ── core residual computation ────────────────────────────────────────────

    def compute_residuals_and_jacobians(
        self,
        scan: np.ndarray,
        pose_matrix: np.ndarray,
    ):
        """Compute point-to-line residuals and Jacobians.

        Args:
            scan: (M, 2) points in robot frame.
            pose_matrix: 3×3 SE(2) matrix (current state estimate).

        Returns:
            z:     (K,)    residual vector (K ≤ M valid correspondences).
            H:     (K, 3)  Jacobian w.r.t. [δx, δy, δθ].
            valid: bool — True if K ≥ 3 (enough for update).
        """
        if self.map_tree is None or len(scan) == 0:
            return np.array([]), np.zeros((0, 3)), False

        R = pose_matrix[:2, :2]
        t = pose_matrix[:2, 2]
        world_pts = (R @ scan.T).T + t  # (M, 2)

        residuals = []
        jacobians = []

        for i, wp in enumerate(world_pts):
            # k-NN in map
            dists, idxs = self.map_tree.query(wp, k=min(self.k, len(self.map_points)))

            # scalar dist when k=1
            if np.isscalar(dists):
                dists = np.array([dists])
                idxs  = np.array([idxs])

            if dists[0] > self.max_dist:
                continue

            neighbors = self.map_points[idxs]       # (k, 2)
            if len(neighbors) < 2:
                continue

            centroid = neighbors.mean(axis=0)
            cov = np.cov((neighbors - centroid).T)
            if cov.ndim == 0:
                cov = np.array([[cov, 0.0], [0.0, 0.0]])

            eigvals, eigvecs = np.linalg.eigh(cov)
            # eigvals sorted ascending → eigvecs[:,0] is normal direction
            normal = eigvecs[:, 0]

            # Planarity check: well-defined line → λ_min << λ_max
            if eigvals[1] > 1e-10 and eigvals[0] / eigvals[1] > self.planarity_thresh:
                continue

            # Residual: signed point-to-line distance
            r = float(normal @ (wp - centroid))

            # Jacobian: ∂r/∂[δx, δy, δθ]
            Rp = R @ scan[i]                            # 2-vector in world frame
            dRp_dth = np.array([-Rp[1], Rp[0]])        # d(Rp)/dθ
            H_i = np.array([normal[0], normal[1], float(normal @ dRp_dth)])

            residuals.append(r)
            jacobians.append(H_i)

        if len(residuals) < 3:
            return np.array([]), np.zeros((0, 3)), False

        return np.array(residuals), np.array(jacobians), True
