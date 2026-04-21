"""
Degenerate corridor experiment: IESKF vs EKF in a featureless 50m corridor.

2D analog of LIMOncello City02 tunnel experiment.

In a straight corridor, LiDAR normals always point ±y (perpendicular to walls).
The along-corridor direction (x) is weakly observable — only from wall endpoints
and corridor entry/exit, not from mid-corridor wall returns.

Expected result:
  - Both filters accumulate x-drift in the featureless zone (0–40m)
  - IESKF (SE(2) manifold, iterated update) drifts less than EKF (Euclidean)
  - After the feature zone (40–50m), IESKF recovers tighter

Usage:
    python3 degenerate_corridor.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os, copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', '..', '..', 'pluto_filters'))

from pluto_filters.ieskf_lio.se2_manifold import Exp, oplus
from pluto_filters.ieskf_lio.ieskf import IESKF
from pluto_filters.ieskf_lio.scan_matcher import ScanMatcher

# ── Parameters ────────────────────────────────────────────────────────────────

RNG          = np.random.default_rng(13)
DT_IMU       = 0.01      # 100 Hz
DT_LIDAR     = 0.1       # 10 Hz
TOTAL_TIME   = 50.0 / 0.3   # time to traverse 50 m at 0.3 m/s ≈ 167 s
SPEED        = 0.3        # m/s
SIGMA_GYRO   = 0.005
SIGMA_ACCEL  = 0.05       # higher accel noise to stress-test
SIGMA_SCAN   = 0.01

CORRIDOR_LENGTH = 50.0
CORRIDOR_WIDTH  = 1.5     # half-width


def _build_corridor_map(n_along=800):
    """Build map for the degenerate corridor (pure straight walls)."""
    xs = np.linspace(0, CORRIDOR_LENGTH, n_along)
    top    = np.stack([xs, np.full(n_along,  CORRIDOR_WIDTH)], axis=1)
    bottom = np.stack([xs, np.full(n_along, -CORRIDOR_WIDTH)], axis=1)
    # End caps only (no entry cap to make approach featureless)
    ys  = np.linspace(-CORRIDOR_WIDTH, CORRIDOR_WIDTH, 30)
    cap = np.stack([np.full(30, CORRIDOR_LENGTH), ys], axis=1)
    # Three pillars in feature zone (x = 41, 44, 47)
    pillar_pts = []
    for cx, cy, r in [(41, 0, 0.15), (44, 0.8, 0.15), (47, -0.8, 0.15)]:
        angs = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        pillar_pts.append(
            np.stack([cx + r * np.cos(angs), cy + r * np.sin(angs)], axis=1)
        )
    return np.vstack([top, bottom, cap] + pillar_pts)


def _simulate_scan(pose_gt, map_pts, n_rays=360, range_max=12.0):
    """Ray-cast scan in robot frame."""
    R_gt = pose_gt[:2, :2]
    t_gt = pose_gt[:2, 2]
    angles = np.linspace(-np.pi, np.pi, n_rays, endpoint=False)
    pts = []
    for a in angles:
        d = R_gt @ np.array([np.cos(a), np.sin(a)])
        vecs = map_pts - t_gt
        projs = vecs @ d
        perp  = np.linalg.norm(vecs - np.outer(projs, d), axis=1)
        mask  = (projs > 0.1) & (projs < range_max) & (perp < 0.05)
        if not mask.any():
            continue
        r = projs[mask].min() + RNG.normal(0, SIGMA_SCAN)
        pts.append(np.array([r * np.cos(a), r * np.sin(a)]))
    return np.array(pts) if pts else np.zeros((0, 2))


def _build_sequences(map_pts):
    steps   = int(TOTAL_TIME / DT_IMU)
    t_arr   = np.arange(steps) * DT_IMU
    gt_poses = [Exp(np.array([SPEED * t, 0.0, 0.0])) for t in t_arr]

    imu_seq = []
    for i, t in enumerate(t_arr):
        om = RNG.normal(0, SIGMA_GYRO)
        ax = RNG.normal(0, SIGMA_ACCEL)
        imu_seq.append((t, om, ax))

    lidar_seq = []
    for t in np.arange(DT_LIDAR, TOTAL_TIME, DT_LIDAR):
        idx  = int(t / DT_IMU)
        scan = _simulate_scan(gt_poses[idx], map_pts)
        if len(scan) >= 5:
            lidar_seq.append((t, scan))

    return gt_poses, imu_seq, lidar_seq


def run_ieskf(map_pts, imu_seq, lidar_seq):
    kf = IESKF(bias_dim=3)
    kf.pose = np.eye(3)
    kf.bias = np.zeros(3)
    kf.P    = np.diag([0.01]*3 + [0.1, 0.01, 0.01])

    sm = ScanMatcher(k_neighbors=5, max_dist=3.0)
    sm.set_map(map_pts)

    traj = [kf.pose.copy()]
    last_t = 0.0

    for (t, om, ax) in imu_seq:
        dt = t - last_t; last_t = t

        def f_nom(pose, bias, dt, _om=om, _ax=ax):
            v, bg, ba = bias
            omega = _om - bg
            a     = _ax - ba
            v_new = v + a * dt
            tau   = np.array([(v + v_new) / 2 * dt, 0.0, omega * dt])
            return oplus(pose, tau), np.array([v_new, bg, ba])

        F_dx = np.eye(6); F_dx[3, 5] = -dt; F_dx[2, 4] = -dt
        F_w  = np.zeros((6, 4))
        F_w[2, 0] = -dt; F_w[0, 1] = dt; F_w[3, 1] = dt
        F_w[4, 2] = 1.0; F_w[5, 3] = 1.0
        sg = SIGMA_GYRO * np.sqrt(dt); sa = SIGMA_ACCEL * np.sqrt(dt)
        Q  = np.diag([sg**2, sa**2, (1e-4*np.sqrt(dt))**2, (1e-3*np.sqrt(dt))**2])
        kf.predict(f_nom, F_dx, F_w, Q, dt)

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

            sm.add_to_map(ScanMatcher.transform_points(scan, kf.pose))
            traj.append(kf.pose.copy())

    return traj


def run_ekf(map_pts, imu_seq, lidar_seq):
    x  = np.zeros(6)
    P  = np.diag([0.01]*3 + [0.1, 0.01, 0.01])

    sm = ScanMatcher(k_neighbors=5, max_dist=3.0)
    sm.set_map(map_pts)

    traj = [Exp(np.array([x[0], x[1], x[2]]))]
    last_t = 0.0

    for (t, om, ax) in imu_seq:
        dt = t - last_t; last_t = t
        th = x[2]; v = x[3]; bg = x[4]; ba = x[5]
        omega = om - bg; a = ax - ba
        v_new = v + a * dt; v_m = (v + v_new) / 2
        x[0] += v_m * np.cos(th) * dt
        x[1] += v_m * np.sin(th) * dt
        x[2] += omega * dt; x[3] = v_new

        F = np.eye(6)
        F[0, 2] = -v_m * np.sin(th) * dt; F[0, 3] = np.cos(th) * dt
        F[1, 2] =  v_m * np.cos(th) * dt; F[1, 3] = np.sin(th) * dt
        F[2, 4] = -dt; F[3, 5] = -dt
        F_w = np.zeros((6, 4))
        F_w[2, 0] = -dt; F_w[0, 1] = dt; F_w[3, 1] = dt
        F_w[4, 2] = 1.0; F_w[5, 3] = 1.0
        sg = SIGMA_GYRO * np.sqrt(dt); sa = SIGMA_ACCEL * np.sqrt(dt)
        Q  = np.diag([sg**2, sa**2, (1e-4*np.sqrt(dt))**2, (1e-3*np.sqrt(dt))**2])
        P  = F @ P @ F.T + F_w @ Q @ F_w.T

        if lidar_seq and abs(lidar_seq[0][0] - t) < DT_IMU / 2:
            _, scan = lidar_seq.pop(0)
            pose_e = Exp(np.array([x[0], x[1], x[2]]))
            z, H3, ok = sm.compute_residuals_and_jacobians(scan, pose_e)
            if ok:
                H6 = np.zeros((len(z), 6)); H6[:, :3] = H3
                V  = np.eye(len(z)) * SIGMA_SCAN**2
                S  = H6 @ P @ H6.T + V
                K  = P @ H6.T @ np.linalg.solve(S, np.eye(S.shape[0]))
                x  = x + K @ z
                P  = (np.eye(6) - K @ H6) @ P
            sm.add_to_map(ScanMatcher.transform_points(scan, Exp(np.array([x[0], x[1], x[2]]))))
            traj.append(Exp(np.array([x[0], x[1], x[2]])))

    return traj


def _ape_along_corridor(traj_est, traj_gt):
    """APE separated into along-corridor (x) and perpendicular (y) components."""
    n = min(len(traj_est), len(traj_gt))
    ex = np.array([abs(traj_est[i][0, 2] - traj_gt[i][0, 2]) for i in range(n)])
    ey = np.array([abs(traj_est[i][1, 2] - traj_gt[i][1, 2]) for i in range(n)])
    return ex, ey


def main():
    map_pts = _build_corridor_map()
    gt_poses, imu_seq, lidar_seq = _build_sequences(map_pts)

    traj_ieskf = run_ieskf(map_pts, copy.deepcopy(imu_seq), copy.deepcopy(lidar_seq))
    traj_ekf   = run_ekf  (map_pts, copy.deepcopy(imu_seq), copy.deepcopy(lidar_seq))

    n = min(len(traj_ieskf), len(traj_ekf))
    gt_sampled = [Exp(np.array([SPEED * t, 0.0, 0.0]))
                  for t in np.arange(0, TOTAL_TIME, DT_LIDAR)[:n]]

    ex_ie, ey_ie = _ape_along_corridor(traj_ieskf[:n], gt_sampled)
    ex_ek, ey_ek = _ape_along_corridor(traj_ekf[:n],   gt_sampled)

    t_axis = np.arange(n) * DT_LIDAR

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Ch05 — IESKF vs EKF: Degenerate Corridor (50 m)\n'
                 '2D analog of LIMOncello City02 tunnel experiment', fontsize=12)

    # Top-view trajectories
    ax = axes[0, 0]
    xs_gt = [p[0, 2] for p in gt_sampled]
    ax.plot(xs_gt, [0]*n, 'g-', lw=2.5, label='Ground truth')
    ax.plot([p[0, 2] for p in traj_ieskf[:n]],
            [p[1, 2] for p in traj_ieskf[:n]], 'r--', lw=1.5, label='IESKF')
    ax.plot([p[0, 2] for p in traj_ekf[:n]],
            [p[1, 2] for p in traj_ekf[:n]],   'b:',  lw=1.5, label='EKF')
    ax.axhline( CORRIDOR_WIDTH, color='k', lw=1); ax.axhline(-CORRIDOR_WIDTH, color='k', lw=1)
    ax.axvline(40, color='gray', ls='--', lw=0.8, label='Feature zone start')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title('Top-view trajectories'); ax.legend(fontsize=8); ax.set_aspect('equal')

    # Along-corridor error (weakly observable)
    ax = axes[0, 1]
    ax.plot(t_axis, ex_ie, 'r-',  lw=1.5, label=f'IESKF  RMSE={np.sqrt(np.mean(ex_ie**2)):.3f}m')
    ax.plot(t_axis, ex_ek, 'b--', lw=1.5, label=f'EKF    RMSE={np.sqrt(np.mean(ex_ek**2)):.3f}m')
    ax.axvline(40/SPEED, color='gray', ls='--', lw=0.8, label='Feature zone')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Along-corridor error [m]')
    ax.set_title('x-error (weakly observable DoF)'); ax.legend(fontsize=8)

    # Perpendicular error (strongly observable)
    ax = axes[1, 0]
    ax.plot(t_axis, ey_ie, 'r-',  lw=1.5, label=f'IESKF  RMSE={np.sqrt(np.mean(ey_ie**2)):.3f}m')
    ax.plot(t_axis, ey_ek, 'b--', lw=1.5, label=f'EKF    RMSE={np.sqrt(np.mean(ey_ek**2)):.3f}m')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Perp-corridor error [m]')
    ax.set_title('y-error (strongly observable DoF)'); ax.legend(fontsize=8)

    # Combined APE
    ax = axes[1, 1]
    ape_ie = np.sqrt(ex_ie**2 + ey_ie**2)
    ape_ek = np.sqrt(ex_ek**2 + ey_ek**2)
    ax.plot(t_axis, ape_ie, 'r-',  lw=1.5, label=f'IESKF  RMSE={np.sqrt(np.mean(ape_ie**2)):.3f}m')
    ax.plot(t_axis, ape_ek, 'b--', lw=1.5, label=f'EKF    RMSE={np.sqrt(np.mean(ape_ek**2)):.3f}m')
    ax.axvline(40/SPEED, color='gray', ls='--', lw=0.8, label='Feature zone')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('APE [m]')
    ax.set_title('Total Absolute Pose Error'); ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'degenerate_corridor.png')
    plt.savefig(out, dpi=120)
    print(f"Saved {out}")
    print(f"\n=== Results ===")
    print(f"IESKF along-corridor RMSE : {np.sqrt(np.mean(ex_ie**2)):.4f} m")
    print(f"EKF   along-corridor RMSE : {np.sqrt(np.mean(ex_ek**2)):.4f} m")
    print(f"IESKF total APE RMSE      : {np.sqrt(np.mean(ape_ie**2)):.4f} m")
    print(f"EKF   total APE RMSE      : {np.sqrt(np.mean(ape_ek**2)):.4f} m")


if __name__ == '__main__':
    main()
