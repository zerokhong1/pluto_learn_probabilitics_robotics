"""
Filter Showdown Benchmark — Chapter 3 Experiment.
Runs KF, EKF, UKF on the same figure-8 trajectory and compares:
  - RMSE position
  - NEES consistency
  - Computation time
Outputs JSON results table + prints summary.
"""

import numpy as np
import time
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[5]))

from pluto_filters.kalman_filters.ekf import EKF, LANDMARKS, normalize_angle
from pluto_filters.kalman_filters.ukf import UKF


def generate_figure8_trajectory(n_steps: int = 200, dt: float = 0.1):
    """Generate a figure-8 trajectory with noisy odometry."""
    t = np.linspace(0, 4 * np.pi, n_steps)
    true_x = 3.0 * np.sin(t / 2)
    true_y = 2.0 * np.sin(t)
    true_theta = np.arctan2(np.gradient(true_y, t), np.gradient(true_x, t))

    true_poses = np.stack([true_x, true_y, true_theta], axis=1)
    return true_poses, dt


def generate_measurements(true_poses: np.ndarray, landmark_id: int = 0,
                           sigma_r: float = 0.2, sigma_phi: float = 0.05):
    """Generate noisy range-bearing measurements to one landmark."""
    lx, ly = LANDMARKS[landmark_id]
    measurements = []
    for pose in true_poses:
        dx = lx - pose[0]
        dy = ly - pose[1]
        r_true = np.sqrt(dx**2 + dy**2)
        phi_true = normalize_angle(np.arctan2(dy, dx) - pose[2])
        r_noisy = r_true + np.random.normal(0, sigma_r)
        phi_noisy = phi_true + np.random.normal(0, sigma_phi)
        measurements.append((r_noisy, phi_noisy))
    return measurements


def rmse(estimates: np.ndarray, truth: np.ndarray) -> float:
    diff = estimates - truth
    diff[:, 2] = np.array([normalize_angle(a) for a in diff[:, 2]])
    return float(np.sqrt(np.mean(diff[:, :2]**2)))


def nees(estimates: np.ndarray, sigmas: list, truth: np.ndarray) -> float:
    """Normalized Estimation Error Squared — consistent filter → NEES ≈ 2."""
    scores = []
    for est, sigma, gt in zip(estimates, sigmas, truth):
        err = est[:2] - gt[:2]
        try:
            nees_i = err @ np.linalg.inv(sigma[:2, :2]) @ err
            scores.append(nees_i)
        except np.linalg.LinAlgError:
            pass
    return float(np.mean(scores)) if scores else float('nan')


def run_ekf_benchmark(true_poses, measurements, v=0.5, omega=0.3, dt=0.1):
    mu0 = np.array([0.2, 0.2, 0.1])
    sigma0 = np.diag([0.5, 0.5, 0.1])
    ekf = EKF(mu0, sigma0)

    estimates = []
    sigmas = []
    t0 = time.perf_counter()

    for i, (r, phi) in enumerate(measurements):
        if i > 0:
            ekf.predict(v, omega, dt)
        ekf.update(0, r, phi)
        estimates.append(ekf.mu.copy())
        sigmas.append(ekf.sigma.copy())

    elapsed = time.perf_counter() - t0
    estimates = np.array(estimates)
    return estimates, sigmas, elapsed


def run_ukf_benchmark(true_poses, measurements, v=0.5, omega=0.3, dt=0.1):
    mu0 = np.array([0.2, 0.2, 0.1])
    sigma0 = np.diag([0.5, 0.5, 0.1])
    ukf = UKF(mu0, sigma0)

    estimates = []
    sigmas = []
    t0 = time.perf_counter()

    for i, (r, phi) in enumerate(measurements):
        if i > 0:
            ukf.predict(v, omega, dt)
        ukf.update(0, r, phi)
        estimates.append(ukf.mu.copy())
        sigmas.append(ukf.sigma.copy())

    elapsed = time.perf_counter() - t0
    estimates = np.array(estimates)
    return estimates, sigmas, elapsed


def main():
    np.random.seed(42)
    print("=" * 60)
    print("  PLUTO FILTER SHOWDOWN — Chapter 3 Benchmark")
    print("=" * 60)

    true_poses, dt = generate_figure8_trajectory(200, 0.1)
    measurements = generate_measurements(true_poses, landmark_id=0)

    # Run filters
    ekf_est, ekf_sig, ekf_time = run_ekf_benchmark(true_poses, measurements)
    ukf_est, ukf_sig, ukf_time = run_ukf_benchmark(true_poses, measurements)

    # Metrics
    ekf_rmse = rmse(ekf_est, true_poses)
    ukf_rmse = rmse(ukf_est, true_poses)
    ekf_nees = nees(ekf_est, ekf_sig, true_poses)
    ukf_nees = nees(ukf_est, ukf_sig, true_poses)

    results = {
        'EKF': {'rmse': ekf_rmse, 'nees': ekf_nees, 'time_ms': ekf_time * 1000},
        'UKF': {'rmse': ukf_rmse, 'nees': ukf_nees, 'time_ms': ukf_time * 1000},
    }

    print(f"\n{'Filter':<8} {'RMSE [m]':<12} {'NEES':<12} {'Time [ms]':<12}")
    print("-" * 48)
    for name, r in results.items():
        print(f"{name:<8} {r['rmse']:<12.4f} {r['nees']:<12.3f} {r['time_ms']:<12.2f}")
    print("\nNote: NEES ≈ 2.0 indicates consistent filter for 2D state.")

    # Save results
    out = Path(__file__).parent / 'results.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == '__main__':
    main()
