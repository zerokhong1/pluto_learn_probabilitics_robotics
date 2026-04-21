"""Unit tests for se2_manifold.py.

All identities from Solà et al. "A micro Lie theory" and the LIMOncello paper.
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from pluto_filters.ieskf_lio.se2_manifold import (
    Exp, Log, hat, vee, oplus, ominus, Adjoint, Jr, Jr_inv,
)

RNG = np.random.default_rng(42)
TOL = 1e-9


def random_se2():
    """Random SE(2) matrix."""
    theta = RNG.uniform(-np.pi, np.pi)
    x, y  = RNG.uniform(-5, 5, 2)
    return Exp(np.array([x, y, theta]))


def random_tau():
    """Random Lie algebra vector [ρx, ρy, θ] with θ ≠ 0."""
    return np.array([RNG.uniform(-2, 2), RNG.uniform(-2, 2),
                     RNG.uniform(-np.pi / 2, np.pi / 2)])


# ── hat / vee ────────────────────────────────────────────────────────────────

def test_vee_hat_roundtrip():
    for _ in range(20):
        tau = random_tau()
        assert np.allclose(vee(hat(tau)), tau, atol=TOL)


# ── Exp / Log ────────────────────────────────────────────────────────────────

def test_exp_identity():
    X = Exp(np.zeros(3))
    assert np.allclose(X, np.eye(3), atol=TOL)


def test_exp_log_roundtrip():
    for _ in range(20):
        tau = random_tau()
        assert np.allclose(Log(Exp(tau)), tau, atol=TOL), f"Log(Exp(τ)) ≠ τ for τ={tau}"


def test_log_exp_roundtrip():
    for _ in range(20):
        X = random_se2()
        assert np.allclose(Exp(Log(X)), X, atol=TOL), "Exp(Log(X)) ≠ X"


def test_exp_small_angle():
    tau = np.array([0.1, 0.2, 1e-12])
    X = Exp(tau)
    assert not np.any(np.isnan(X)), "Exp produced NaN for small angle"
    assert np.allclose(Log(X), tau, atol=1e-8)


# ── oplus / ominus ───────────────────────────────────────────────────────────

def test_oplus_identity():
    for _ in range(20):
        X = random_se2()
        assert np.allclose(oplus(X, np.zeros(3)), X, atol=TOL)


def test_oplus_ominus_roundtrip():
    for _ in range(20):
        X = random_se2()
        tau = random_tau()
        assert np.allclose(ominus(oplus(X, tau), X), tau, atol=TOL)


def test_ominus_oplus_roundtrip():
    for _ in range(20):
        X = random_se2()
        Y = random_se2()
        assert np.allclose(oplus(X, ominus(Y, X)), Y, atol=TOL)


# ── Adjoint ──────────────────────────────────────────────────────────────────

def test_adjoint_identity():
    Ad = Adjoint(np.eye(3))
    assert np.allclose(Ad, np.eye(3), atol=TOL)


def test_adjoint_equivariance():
    """Ad(X) Exp(τ) X⁻¹ should equal X Exp(τ) X⁻¹ via adjoint property."""
    for _ in range(10):
        X   = random_se2()
        tau = random_tau()
        lhs = X @ Exp(tau) @ np.linalg.inv(X)
        rhs = Exp(Adjoint(X) @ tau)
        assert np.allclose(lhs, rhs, atol=1e-8), "Adjoint equivariance failed"


# ── Jr / Jr_inv ───────────────────────────────────────────────────────────────

def test_jr_jrinv_inverse():
    for _ in range(20):
        tau = random_tau()
        J   = Jr(tau)
        Ji  = Jr_inv(tau)
        assert np.allclose(J @ Ji, np.eye(3), atol=1e-7), "Jr @ Jr_inv ≠ I"
        assert np.allclose(Ji @ J, np.eye(3), atol=1e-7), "Jr_inv @ Jr ≠ I"


def test_jr_small_angle():
    tau = np.array([0.3, -0.1, 1e-11])
    J  = Jr(tau)
    Ji = Jr_inv(tau)
    assert not np.any(np.isnan(J)),  "Jr NaN for small angle"
    assert not np.any(np.isnan(Ji)), "Jr_inv NaN for small angle"
    assert np.allclose(J @ Ji, np.eye(3), atol=1e-7)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
