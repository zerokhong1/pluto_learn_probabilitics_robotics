"""
Hallway comparison experiment: EKF vs IESKF on the standard Pluto hallway.

Runs both filters offline against a simulated trajectory,
computes Absolute Pose Error (APE), and plots trajectories side-by-side.

Usage (standalone, no ROS 2 needed):
    python3 hallway_comparison.py

For ROS 2 live comparison see the corresponding launch file.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', '..', '..', 'pluto_filters'))

from pluto_filters.ieskf_lio.se2_manifold import Exp, oplus, ominus
from pluto_filters.ieskf_lio.ieskf import IESKF
from pluto_filters.ieskf_lio.scan_matcher import ScanMatcher

# ── Simulation parameters ─────────────────────────────────────────────────────

RNG          = np.random.default_rng(42)
DT_IMU       = 0.01     # 100 Hz
DT_LIDAR     = 0.1      # 10 Hz
TOTAL_TIME   = 30.0     # seconds
SPEED        = 0.3      # m/s forward
SIGMA_GYRO   = 0.005    # rad/s noise std
SIGMA_ACCEL  = 0.02     # m/s² noise std
SIGMA_SCAN   = 0.01     # m per LiDAR point

# Hallway geometry (matches pluto_hallway.sdf)
WALL_Y       = 1.05
HALL_LENGTH  = 10.0
DOOR_POS     = [2.0, 5.0, 8.0]
DOOR_WIDTH   = 0.4


def _build_hallway_map(n_pts=400):
    """Build (N, 2) world-frame map points for the hallway walls."""
    xs = np.linspace(0, HALL_LENGTH, n_pts // 2)
    top    = np.stack([xs, np.full(len(xs),  WALL_Y)], axis=1)
    bottom = np.stack([xs, np.full(len(xs), -WALL_Y)], axis=1)
    # Cap walls
    ys = np.linspace(-WALL_Y, WALL_Y, 20)
    left  = np.stack([np.zeros(20),              ys], axis=1)
    right = np.stack([np.full(20, HALL_LENGTH),  ys], axis=1)
    return np.vstack([top, bottom, left, right])


def _simulate_imu(pose_gt, dt, omega_true, ax_true):
    """Sample noisy IMU readings."""
    omega_meas = omega_true + RNG.normal(0, SIGMA_GYRO)
    ax_meas    = ax_true    + RNG.normal(0, SIGMA_ACCEL)
    return omega_meas, ax_meas


def _simulate_scan(pose_gt: np.ndarray, map_pts: np.ndarray,
                   n_rays=360, range_max=12.0) -> np.ndarray:
    """Ray-cast against map points to get a noisy 2D scan (robot frame)."""
    R_gt = pose_gt[:2, :2]
    t_gt = pose_gt[:2, 2]
    angles = np.linspace(-np.pi, np.pi, n_rays, endpoint=False)
    pts = []
    for a in angles:
        d = R_gt @ np.array([np.cos(a), np.sin(a)])
        # Find closest map point along ray direction
        vecs = map_pts - t_gt
        projs = vecs @ d
        dists_perp = np.linalg.norm(vecs - np.outer(projs, d), axis=1)
        mask = (projs > 0.1) & (projs < range_max) & (dists_perp < 0.05)
        if not mask.any():
            continue
        r = projs[mask].min() + RNG.normal(0, SIGMA_SCAN)
        p_robot = np.array([r * np.cos(a), r * np.sin(a)])
        pts.append(p_robot)
    return np.array(pts) if pts else np.zeros((0, 2))


def _ape(traj_est, traj_gt):
    """Absolute Pose Error (translation only) [m]."""
    errs = [np.linalg.norm(e[:2, 2] - g[:2, 2])
            for e, g in zip(traj_est, traj_gt)]
    return np.array(errs)


def run_ieskf(map_pts, gt_poses, imu_seq, lidar_seq):
    """Run IESKF and return estimated trajectory."""
    kf = IESKF(bias_dim=3)
    kf.pose = np.eye(3)
    kf.bias = np.zeros(3)  # [v, b_gyro, b_accel]
    kf.P    = np.diag([0.01]*3 + [0.1, 0.01, 0.01])

    sm = ScanMatcher(k_neighbors=5, max_dist=2.0)
    sm.set_map(map_pts)

    traj = [kf.pose.copy()]
    last_t = 0.0

    for (t, omega_m, ax_m) in imu_seq:
        dt = t - last_t
        last_t = t

        def f_nom(pose, bias, dt):
            v, bg, ba = bias
            omega = omega_m - bg
            ax    = ax_m - ba
            v_new = v + ax * dt
            tau   = np.array([(v + v_new) / 2 * dt, 0.0, omega * dt])
            return oplus(pose, tau), np.array([v_new, bg, ba])

        F_dx = np.eye(6)
        F_dx[3, 5] = -dt
        F_w = np.zeros((6, 4))
        F_w[2, 0] = -dt; F_w[0, 1] = dt; F_w[3, 1] = dt
        F_w[4, 2] = 1.0; F_w[5, 3] = 1.0
        sg = SIGMA_GYRO * np.sqrt(dt)
        sa = SIGMA_ACCEL * np.sqrt(dt)
        Q  = np.diag([sg**2, sa**2, (1e-4*np.sqrt(dt))**2, (1e-3*np.sqrt(dt))**2])
        kf.predict(f_nom, F_dx, F_w, Q, dt)

        # Check if LiDAR frame at this time
        if lidar_seq and abs(lidar_seq[0][0] - t) < DT_IMU / 2:
            _, scan = lidar_seq.pop(0)

            def zh_fn(pj, _, _s=scan):
                z, H3, ok = sm.compute_residuals_and_jacobians(_s, pj)
                if not ok:
                    return np.array([]), np.zeros((0, 6))
                H6 = np.zeros((H3.shape[0], 6)); H6[:, :3] = H3
                return z, H6

            z_p, _ = zh_fn(kf.pose, kf.bias)
            if z_p.size > 0:
                kf.update(zh_fn, np.eye(1) * SIGMA_SCAN**2, max_iter=5)

            world_pts = ScanMatcher.transform_points(scan, kf.pose)
            sm.add_to_map(world_pts)
            traj.append(kf.pose.copy())

    return traj


def run_ekf_euclidean(map_pts, gt_poses, imu_seq, lidar_seq):
    """
    EKF with flat Euclidean state [x, y, θ, v, b_gyro, b_accel].
    Standard (non-iterated) update for comparison.
    """
    x  = np.zeros(6)   # [px, py, th, v, bg, ba]
    P  = np.diag([0.01]*3 + [0.1, 0.01, 0.01])

    sm = ScanMatcher(k_neighbors=5, max_dist=2.0)
    sm.set_map(map_pts)

    traj = [Exp(np.array([x[0], x[1], x[2]]))]
    last_t = 0.0

    for (t, omega_m, ax_m) in imu_seq:
        dt = t - last_t; last_t = t

        th = x[2]; v = x[3]; bg = x[4]; ba = x[5]
        omega = omega_m - bg
        ax    = ax_m - ba
        v_new = v + ax * dt
        v_m   = (v + v_new) / 2

        # Predict
        x[0] += v_m * np.cos(th) * dt
        x[1] += v_m * np.sin(th) * dt
        x[2] += omega * dt
        x[3]  = v_new

        F = np.eye(6)
        F[0, 2] = -v_m * np.sin(th) * dt
        F[0, 3] =  np.cos(th) * dt
        F[0, 5] = -np.cos(th) * dt * dt / 2
        F[1, 2] =  v_m * np.cos(th) * dt
        F[1, 3] =  np.sin(th) * dt
        F[1, 5] = -np.sin(th) * dt * dt / 2
        F[2, 4] = -dt; F[3, 5] = -dt

        sg = SIGMA_GYRO * np.sqrt(dt)
        sa = SIGMA_ACCEL * np.sqrt(dt)
        F_w = np.zeros((6, 4))
        F_w[2, 0] = -dt; F_w[0, 1] = dt; F_w[3, 1] = dt
        F_w[4, 2] = 1.0; F_w[5, 3] = 1.0
        Q  = np.diag([sg**2, sa**2, (1e-4*np.sqrt(dt))**2, (1e-3*np.sqrt(dt))**2])
        P  = F @ P @ F.T + F_w @ Q @ F_w.T

        if lidar_seq and abs(lidar_seq[0][0] - t) < DT_IMU / 2:
            _, scan = lidar_seq.pop(0)
            pose_est = Exp(np.array([x[0], x[1], x[2]]))
            z, H3, ok = sm.compute_residuals_and_jacobians(scan, pose_est)
            if ok:
                H6 = np.zeros((len(z), 6)); H6[:, :3] = H3
                V  = np.eye(len(z)) * SIGMA_SCAN**2
                S  = H6 @ P @ H6.T + V
                K  = P @ H6.T @ np.linalg.solve(S, np.eye(S.shape[0]))
                x  = x + K @ z
                P  = (np.eye(6) - K @ H6) @ P

            world_pts = ScanMatcher.transform_points(scan, Exp(np.array([x[0], x[1], x[2]])))
            sm.add_to_map(world_pts)
            traj.append(Exp(np.array([x[0], x[1], x[2]])))

    return traj


def main():
    map_pts = _build_hallway_map()

    # Ground-truth trajectory (straight line at SPEED m/s)
    steps    = int(TOTAL_TIME / DT_IMU)
    t_arr    = np.arange(steps) * DT_IMU
    gt_poses = [Exp(np.array([SPEED * t, 0.0, 0.0])) for t in t_arr]

    # IMU sequence: (t, omega_meas, ax_meas)
    imu_seq = []
    for i, t in enumerate(t_arr):
        om, ax = _simulate_imu(gt_poses[i], DT_IMU, 0.0, 0.0)
        imu_seq.append((t, om, ax))

    # LiDAR sequence: every DT_LIDAR
    lidar_seq = []
    for t in np.arange(DT_LIDAR, TOTAL_TIME, DT_LIDAR):
        idx  = int(t / DT_IMU)
        scan = _simulate_scan(gt_poses[idx], map_pts)
        if len(scan) >= 5:
            lidar_seq.append((t, scan))

    import copy
    imu_ieskf  = copy.deepcopy(imu_seq)
    lidar_ieskf = copy.deepcopy(lidar_seq)
    imu_ekf    = copy.deepcopy(imu_seq)
    lidar_ekf  = copy.deepcopy(lidar_seq)

    traj_ieskf = run_ieskf(map_pts, gt_poses, imu_ieskf, lidar_ieskf)
    traj_ekf   = run_ekf_euclidean(map_pts, gt_poses, imu_ekf, lidar_ekf)

    # Align lengths
    n = min(len(traj_ieskf), len(traj_ekf), len(lidar_seq) + 1)
    gt_traj = [Exp(np.array([SPEED * t, 0.0, 0.0]))
               for t in np.arange(0, TOTAL_TIME, DT_LIDAR)[:n]]

    ape_ieskf = _ape(traj_ieskf[:n], gt_traj)
    ape_ekf   = _ape(traj_ekf[:n],   gt_traj)

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Ch05 — IESKF vs EKF: Hallway Comparison', fontsize=13)

    ax = axes[0]
    xs_gt = [p[0, 2] for p in gt_traj]
    ys_gt = [p[1, 2] for p in gt_traj]
    xs_ie = [p[0, 2] for p in traj_ieskf[:n]]
    ys_ie = [p[1, 2] for p in traj_ieskf[:n]]
    xs_ek = [p[0, 2] for p in traj_ekf[:n]]
    ys_ek = [p[1, 2] for p in traj_ekf[:n]]

    ax.plot(xs_gt, ys_gt, 'g-',  lw=2.5, label='Ground truth')
    ax.plot(xs_ie, ys_ie, 'r--', lw=1.5, label='IESKF')
    ax.plot(xs_ek, ys_ek, 'b:',  lw=1.5, label='EKF')
    ax.axhline( WALL_Y, color='k', lw=1)
    ax.axhline(-WALL_Y, color='k', lw=1)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title('Trajectories'); ax.legend(); ax.set_aspect('equal')

    ax = axes[1]
    t_axis = np.arange(n) * DT_LIDAR
    ax.plot(t_axis, ape_ieskf, 'r-',  lw=1.5, label=f'IESKF  RMSE={np.sqrt(np.mean(ape_ieskf**2)):.3f}m')
    ax.plot(t_axis, ape_ekf,   'b--', lw=1.5, label=f'EKF    RMSE={np.sqrt(np.mean(ape_ekf**2)):.3f}m')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('APE [m]')
    ax.set_title('Absolute Pose Error'); ax.legend()

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'hallway_comparison.png')
    plt.savefig(out, dpi=120)
    print(f"Saved {out}")
    print(f"IESKF RMSE: {np.sqrt(np.mean(ape_ieskf**2)):.4f} m")
    print(f"EKF   RMSE: {np.sqrt(np.mean(ape_ekf**2)):.4f} m")


if __name__ == '__main__':
    main()
