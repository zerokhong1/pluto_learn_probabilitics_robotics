"""
Iterated Error-State Kalman Filter (IESKF) on manifolds.

Generic implementation independent of any specific system model.
State lives on a manifold; error-state lives in tangent space (Rⁿ).

Reference: LIMOncello paper Section II-B, equations (15), (19), (20).
           IKFoM: https://github.com/hku-mars/IKFoM
"""

import numpy as np
from .se2_manifold import oplus, ominus, Jr_inv


class IESKF:
    """Iterated Error-State Kalman Filter on SE(2) × Rᵏ manifolds.

    The state is split into:
      - pose: 3×3 SE(2) matrix
      - bias: R^bias_dim vector (gyro bias + accel bias)

    Error-state dimension = 3 (SE(2) tangent) + bias_dim.
    """

    def __init__(self, bias_dim: int = 2):
        """
        Args:
            bias_dim: dimension of bias vector (default 2: b_gyro, b_accel)
        """
        self.error_dim = 3 + bias_dim
        self.bias_dim = bias_dim

        self.pose = np.eye(3)               # SE(2) nominal pose
        self.bias = np.zeros(bias_dim)      # nominal biases
        self.P = np.eye(self.error_dim) * 0.01  # error-state covariance

    # ── helpers ────────────────────────────────────────────────────────────

    def _state_oplus(self, pose, bias, delta: np.ndarray):
        """Composite manifold update: (SE(2) × Rᵏ) ⊕ δ."""
        new_pose = oplus(pose, delta[:3])
        new_bias = bias + delta[3:]
        return new_pose, new_bias

    def _state_ominus(self, pose_j, bias_j, pose_hat, bias_hat) -> np.ndarray:
        """Composite manifold difference: (x_j ⊖ x_hat) ∈ R^error_dim."""
        d_pose = ominus(pose_j, pose_hat)       # 3
        d_bias = bias_j - bias_hat              # bias_dim
        return np.concatenate([d_pose, d_bias])

    # ── prediction ─────────────────────────────────────────────────────────

    def predict(self, f_nominal, F_dx: np.ndarray, F_w: np.ndarray,
                Q: np.ndarray, dt: float):
        """Prediction step. Paper eq. (15a-b).

        Args:
            f_nominal: callable(pose, bias, dt) -> (pose_new, bias_new).
                       Propagates nominal state using motion model.
            F_dx: (error_dim × error_dim) error-state transition Jacobian.
            F_w:  (error_dim × noise_dim) noise input Jacobian.
            Q:    (noise_dim × noise_dim) process noise covariance.
            dt:   time step [s].
        """
        self.pose, self.bias = f_nominal(self.pose, self.bias, dt)
        self.P = F_dx @ self.P @ F_dx.T + F_w @ Q @ F_w.T

    # ── iterated update ────────────────────────────────────────────────────

    def update(self, z_func, H_func, V: np.ndarray,
               max_iter: int = 5, eps: float = 1e-4):
        """Iterated update step. Paper eq. (20a-b).

        Args:
            z_func: callable(pose_j, bias_j) -> np.ndarray (m,).
                    Computes residual vector at current linearization point.
            H_func: callable(pose_j, bias_j) -> np.ndarray (m, error_dim).
                    Measurement Jacobian at current linearization point.
            V:      (m × m) measurement noise covariance.
            max_iter: maximum IESKF iterations.
            eps:    convergence threshold on ‖x_{j+1} ⊖ x_j‖.

        Key difference from standard EKF:
          - Multiple re-linearizations
          - Prior transported to current linearization point via Jr_inv
          - Update uses manifold ⊕/⊖ instead of vector addition
        """
        # Save prior (x̂)
        pose_hat = self.pose.copy()
        bias_hat = self.bias.copy()

        # Working estimate starts at prior
        pose_j = self.pose.copy()
        bias_j = self.bias.copy()

        K_j = None  # keep for final covariance update

        for _ in range(max_iter):
            z_j = z_func(pose_j, bias_j)
            if z_j.size == 0:
                break
            H_j = H_func(pose_j, bias_j)

            # Transport Jacobian J_j = Jr_inv(x_j ⊖ x̂)  — SE(2) part only
            # For the composite manifold, extend to full error_dim
            dx = self._state_ominus(pose_j, bias_j, pose_hat, bias_hat)
            J_j = np.eye(self.error_dim)
            J_j[:3, :3] = Jr_inv(dx[:3])   # SE(2) block; bias block stays I

            J_inv = np.linalg.inv(J_j)
            P_j = J_inv @ self.P @ J_inv.T

            # Kalman gain (eq. 20a)
            S = H_j @ P_j @ H_j.T + V
            K_j = P_j @ H_j.T @ np.linalg.solve(S, np.eye(S.shape[0]))

            # State update (eq. 20b)
            correction = (
                -K_j @ z_j
                - (np.eye(self.error_dim) - K_j @ H_j) @ J_inv @ dx
            )
            pose_new, bias_new = self._state_oplus(pose_j, bias_j, correction)

            # Convergence check
            step = self._state_ominus(pose_new, bias_new, pose_j, bias_j)
            pose_j, bias_j = pose_new, bias_new
            if np.linalg.norm(step) < eps:
                break

        self.pose = pose_j
        self.bias = bias_j

        # Covariance update (Joseph form for numerical stability)
        if K_j is not None and z_j.size > 0:
            IKH = np.eye(self.error_dim) - K_j @ H_j
            self.P = IKH @ self.P @ IKH.T + K_j @ V @ K_j.T
