"""
SE(2) Lie group operations for 2D rigid body transformations.

SE(2) elements are 3x3 matrices:
    X = | cos(θ)  -sin(θ)  x |
        | sin(θ)   cos(θ)  y |
        |   0        0     1 |

Lie algebra vector: τ = [ρx, ρy, θ]ᵀ ∈ R³

All formulas verified numerically against finite-difference Jacobians.

Reference: Solà et al. "A micro Lie theory for state estimation in robotics"
           https://arxiv.org/abs/1812.01537
           LIMOncello paper Section II-A.
"""

import numpy as np

_EPS = 1e-8


def hat(tau: np.ndarray) -> np.ndarray:
    """Wedge operator: R³ → se(2) (3×3 Lie algebra matrix).
    τ = [ρx, ρy, θ]
    """
    rx, ry, th = tau
    return np.array([
        [0.0, -th,  rx],
        [th,   0.0, ry],
        [0.0,  0.0, 0.0],
    ])


def vee(tau_hat: np.ndarray) -> np.ndarray:
    """Vee operator: se(2) matrix → R³. Inverse of hat."""
    return np.array([tau_hat[0, 2], tau_hat[1, 2], tau_hat[1, 0]])


def Exp(tau: np.ndarray) -> np.ndarray:
    """Exponential map: R³ → SE(2). Closed-form formula."""
    rx, ry, th = tau
    c, s = np.cos(th), np.sin(th)

    if abs(th) > _EPS:
        V = np.array([
            [s,           -(1.0 - c)],
            [1.0 - c,      s        ],
        ]) / th
    else:
        V = np.array([
            [1.0,      -th / 2.0],
            [th / 2.0,  1.0     ],
        ])

    t = V @ np.array([rx, ry])
    return np.array([
        [c,  -s,  t[0]],
        [s,   c,  t[1]],
        [0.0, 0.0, 1.0],
    ])


def Log(X: np.ndarray) -> np.ndarray:
    """Logarithmic map: SE(2) → R³. Inverse of Exp."""
    c, s = X[0, 0], X[1, 0]
    th = np.arctan2(s, c)
    tx, ty = X[0, 2], X[1, 2]

    if abs(th) > _EPS:
        a = s / th
        b = (1.0 - c) / th
        denom = a * a + b * b
        V_inv = np.array([
            [ a,  b],
            [-b,  a],
        ]) / denom
    else:
        V_inv = np.array([
            [1.0,       th / 2.0],
            [-th / 2.0, 1.0     ],
        ])

    rho = V_inv @ np.array([tx, ty])
    return np.array([rho[0], rho[1], th])


def oplus(X: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Right-plus: X ⊕ τ = X @ Exp(τ).  Paper eq. (1a)."""
    return X @ Exp(tau)


def ominus(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Right-minus: Y ⊖ X = Log(X⁻¹ @ Y).  Paper eq. (1b)."""
    return Log(np.linalg.inv(X) @ Y)


def Adjoint(X: np.ndarray) -> np.ndarray:
    """Adjoint matrix of X ∈ SE(2). Returns 3×3 matrix.

    Verified: X Exp(τ) X⁻¹ = Exp(Ad(X) τ) for any τ.

    Ad(X) = | R    [ty, -tx]ᵀ |
             | 0        1     |
    """
    R  = X[:2, :2]
    tx = X[0, 2]
    ty = X[1, 2]
    Ad = np.zeros((3, 3))
    Ad[:2, :2] = R
    Ad[0, 2]   =  ty
    Ad[1, 2]   = -tx
    Ad[2, 2]   =  1.0
    return Ad


def Jr(tau: np.ndarray) -> np.ndarray:
    """Right Jacobian of SE(2). τ ∈ R³ → 3×3 matrix.

    Derived from Jr = Σ (-1)^n/(n+1)! ad(τ)^n and verified numerically.

    Jr = | A·I₂ + B·J_s   Q·ρ |
         |      0           1  |

    where A = sin θ/θ,  B = (1-cos θ)/θ,
          J_s = [[0,1],[-1,0]]  (the skew part),
          Q = [[(θ-sin θ)/θ², -(1-cos θ)/θ²],
               [(1-cos θ)/θ²,  (θ-sin θ)/θ²]]
    """
    rx, ry, th = tau
    c, s = np.cos(th), np.sin(th)

    if abs(th) > _EPS:
        A  = s / th
        B  = (1.0 - c) / th
        C  = (th - s) / (th * th)
        D  = (1.0 - c) / (th * th)
    else:
        A  = 1.0 - th**2 / 6.0
        B  = th / 2.0 - th**3 / 24.0
        C  = th / 6.0 - th**3 / 120.0
        D  = 0.5 - th**2 / 24.0

    # Upper-left 2×2: A·I + B·[[0,1],[-1,0]]
    UL = np.array([[A,  B],
                   [-B, A]])

    # Upper-right 2×1: Q @ [ρx, ρy]
    UR = np.array([C * rx - D * ry,
                   D * rx + C * ry])

    J = np.zeros((3, 3))
    J[:2, :2] = UL
    J[:2, 2]  = UR
    J[2, 2]   = 1.0
    return J


def Jr_inv(tau: np.ndarray) -> np.ndarray:
    """Inverse of right Jacobian of SE(2).

    Analytically inverted from Jr's block structure:
    Jr_inv = | M⁻¹   -M⁻¹ v |
              |  0       1   |
    where M is the 2×2 upper-left and v is the upper-right column.
    """
    rx, ry, th = tau
    c, s = np.cos(th), np.sin(th)

    if abs(th) > _EPS:
        # M = A·I + B·J_s with det(M) = A²+B²
        A  = s / th
        B  = (1.0 - c) / th
        det = A * A + B * B
        # M⁻¹ = [[A,-B],[B,A]] / det
        M_inv = np.array([[A, -B],
                          [B,  A]]) / det
        C  = (th - s) / (th * th)
        D  = (1.0 - c) / (th * th)
    else:
        M_inv = np.array([[1.0, -th / 2.0],
                          [th / 2.0, 1.0]])
        C  = th / 6.0
        D  = 0.5

    v = np.array([C * rx - D * ry,
                  D * rx + C * ry])

    J = np.zeros((3, 3))
    J[:2, :2] =  M_inv
    J[:2,  2] = -M_inv @ v
    J[2,   2] =  1.0
    return J
