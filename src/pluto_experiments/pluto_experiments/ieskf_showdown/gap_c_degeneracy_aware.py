"""
Gap C — Degeneracy-Aware IESKF (DA-IESKF) vs IESKF vs EKF.

Contribution:
    Standard IESKF blindly applies the Kalman update in ALL pose directions,
    including directions that are NOT observable from the current scan
    (e.g. along-corridor x in a featureless tunnel).  Numerical noise in H
    causes the filter to over-constrain those directions, shrinking P_xx
    faster than warranted.  When features later appear (pillars, room end),
    the filter has insufficient uncertainty to correct accumulated x-drift.

DA-IESKF fix:
    Before each IESKF update, compute SVD of the 3-column pose-block of H.
    Any singular value below thresh * σ_max corresponds to a direction in
    SE(2) state space that is poorly observed.  Zero those singular values
    before reconstructing H.  The update then only moves the estimate in
    genuinely observable directions; P grows correctly in degenerate ones.

Experiment: 50m featureless corridor (0-40m) + pillar feature zone (40-50m).
    - Featureless zone: x is weakly observable (wall normals ≈ ±y only)
    - Feature zone: 3 pillars at x=41,44,47 → x becomes observable

Expected result:
    DA-IESKF accumulates slightly more x-uncertainty in the corridor, then
    makes a larger, better-founded correction when pillars appear.
    Net effect: lower along-corridor APE at the end.

Usage:
    python3 gap_c_degeneracy_aware.py
"""

import numpy as np
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', '..', '..', 'pluto_filters'))

from pluto_filters.ieskf_lio.se2_manifold import Exp, oplus
from pluto_filters.ieskf_lio.ieskf import IESKF
from pluto_filters.ieskf_lio.scan_matcher import ScanMatcher

# ── Parameters (same as degenerate_corridor.py) ───────────────────────────────

RNG             = np.random.default_rng(13)
DT_IMU          = 0.01
DT_LIDAR        = 0.1
TOTAL_TIME      = 50.0 / 0.3
SPEED           = 0.3
SIGMA_GYRO      = 0.005
SIGMA_ACCEL     = 0.05
SIGMA_SCAN      = 0.01
CORRIDOR_LENGTH = 50.0
CORRIDOR_WIDTH  = 1.5

# DA-IESKF degeneracy threshold: singular values below this fraction of σ_max
# are treated as unobservable and zeroed out.
DEGENERACY_THRESH = 0.08


# ── Environment ───────────────────────────────────────────────────────────────

def _build_corridor_map(n_along=800):
    xs = np.linspace(0, CORRIDOR_LENGTH, n_along)
    top    = np.stack([xs, np.full(n_along,  CORRIDOR_WIDTH)], axis=1)
    bottom = np.stack([xs, np.full(n_along, -CORRIDOR_WIDTH)], axis=1)
    ys  = np.linspace(-CORRIDOR_WIDTH, CORRIDOR_WIDTH, 30)
    cap = np.stack([np.full(30, CORRIDOR_LENGTH), ys], axis=1)
    pillar_pts = []
    for cx, cy, r in [(41, 0, 0.15), (44, 0.8, 0.15), (47, -0.8, 0.15)]:
        angs = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        pillar_pts.append(
            np.stack([cx + r * np.cos(angs), cy + r * np.sin(angs)], axis=1))
    return np.vstack([top, bottom, cap] + pillar_pts)


def _simulate_scan(pose_gt, map_pts, n_rays=360, range_max=12.0):
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
    imu_seq = [(t, RNG.normal(0, SIGMA_GYRO), RNG.normal(0, SIGMA_ACCEL))
               for t in t_arr]
    lidar_seq = []
    for t in np.arange(DT_LIDAR, TOTAL_TIME, DT_LIDAR):
        idx  = int(t / DT_IMU)
        scan = _simulate_scan(gt_poses[idx], map_pts)
        if len(scan) >= 5:
            lidar_seq.append((t, scan))
    return gt_poses, imu_seq, lidar_seq


# ── Degeneracy-aware H filter ─────────────────────────────────────────────────

def _svd_filter_H(H6, thresh=DEGENERACY_THRESH):
    """
    Project H6's pose block (columns 0:3) to remove unobservable directions.

    Uses truncated SVD: singular values below thresh * σ_max are zeroed.
    Returns (H6_filtered, sigma_ratio) where sigma_ratio = σ_min/σ_max
    (0 = fully degenerate, 1 = fully observable in all directions).
    """
    H3 = H6[:, :3]
    if H3.shape[0] < 3:
        return H6, 1.0

    U, S, Vt = np.linalg.svd(H3, full_matrices=False)
    s_max = S.max()
    if s_max < 1e-12:
        return H6, 0.0

    sigma_ratio = S.min() / s_max

    # Zero out degenerate singular values
    S_filt = np.where(S >= thresh * s_max, S, 0.0)

    H3_filt = U @ np.diag(S_filt) @ Vt
    H6_filt = H6.copy()
    H6_filt[:, :3] = H3_filt
    return H6_filt, sigma_ratio


# ── IMU prediction helper (shared across all filters) ─────────────────────────

def _imu_jacobians(om, ax, dt):
    """Return (f_nom, F_dx, F_w, Q) for one IMU step."""
    sg = SIGMA_GYRO  * np.sqrt(dt)
    sa = SIGMA_ACCEL * np.sqrt(dt)
    Q  = np.diag([sg**2, sa**2,
                  (1e-4 * np.sqrt(dt))**2, (1e-3 * np.sqrt(dt))**2])

    def f_nom(pose, bias, dt, _om=om, _ax=ax):
        v, bg, ba = bias
        omega = _om - bg
        a     = _ax - ba
        v_new = v + a * dt
        tau   = np.array([(v + v_new) / 2 * dt, 0.0, omega * dt])
        return oplus(pose, tau), np.array([v_new, bg, ba])

    F_dx = np.eye(6)
    F_dx[3, 5] = -dt
    F_dx[2, 4] = -dt
    F_w = np.zeros((6, 4))
    F_w[2, 0] = -dt;  F_w[0, 1] = dt;  F_w[3, 1] = dt
    F_w[4, 2] = 1.0;  F_w[5, 3] = 1.0
    return f_nom, F_dx, F_w, Q


# ── Run: DA-IESKF ─────────────────────────────────────────────────────────────

def run_da_ieskf(map_pts, imu_seq, lidar_seq):
    kf = IESKF(bias_dim=3)
    kf.pose = np.eye(3)
    kf.bias = np.zeros(3)
    kf.P    = np.diag([0.01]*3 + [0.1, 0.01, 0.01])

    sm = ScanMatcher(k_neighbors=5, max_dist=3.0)
    sm.set_map(map_pts)

    traj         = [kf.pose.copy()]
    p_xx         = [kf.P[0, 0]]   # x-direction variance
    sigma_ratios = [1.0]           # degeneracy indicator
    last_t = 0.0

    for (t, om, ax) in imu_seq:
        dt = t - last_t; last_t = t
        f_nom, F_dx, F_w, Q = _imu_jacobians(om, ax, dt)
        kf.predict(f_nom, F_dx, F_w, Q, dt)

        if lidar_seq and abs(lidar_seq[0][0] - t) < DT_IMU / 2:
            _, scan = lidar_seq.pop(0)

            ratio_this = [1.0]  # capture inside closure

            def zh_fn(pj, _, _s=scan, _r=ratio_this):
                z, H3, ok = sm.compute_residuals_and_jacobians(_s, pj)
                if not ok:
                    return np.array([]), np.zeros((0, 6))
                H6 = np.zeros((H3.shape[0], 6))
                H6[:, :3] = H3
                H6_filt, ratio = _svd_filter_H(H6)
                _r[0] = ratio
                return z, H6_filt

            z_p, _ = zh_fn(kf.pose, kf.bias)
            if z_p.size > 0:
                kf.update(zh_fn, np.eye(1) * SIGMA_SCAN**2, max_iter=5)

            sm.add_to_map(ScanMatcher.transform_points(scan, kf.pose))
            traj.append(kf.pose.copy())
            p_xx.append(kf.P[0, 0])
            sigma_ratios.append(ratio_this[0])

    return traj, np.array(p_xx), np.array(sigma_ratios)


# ── Run: standard IESKF ───────────────────────────────────────────────────────

def run_ieskf(map_pts, imu_seq, lidar_seq):
    kf = IESKF(bias_dim=3)
    kf.pose = np.eye(3)
    kf.bias = np.zeros(3)
    kf.P    = np.diag([0.01]*3 + [0.1, 0.01, 0.01])

    sm = ScanMatcher(k_neighbors=5, max_dist=3.0)
    sm.set_map(map_pts)

    traj = [kf.pose.copy()]
    p_xx = [kf.P[0, 0]]
    last_t = 0.0

    for (t, om, ax) in imu_seq:
        dt = t - last_t; last_t = t
        f_nom, F_dx, F_w, Q = _imu_jacobians(om, ax, dt)
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
            p_xx.append(kf.P[0, 0])

    return traj, np.array(p_xx)


# ── Run: EKF baseline ─────────────────────────────────────────────────────────

def run_ekf(map_pts, imu_seq, lidar_seq):
    x = np.zeros(6)
    P = np.diag([0.01]*3 + [0.1, 0.01, 0.01])

    sm = ScanMatcher(k_neighbors=5, max_dist=3.0)
    sm.set_map(map_pts)

    traj = [Exp(np.array([x[0], x[1], x[2]]))]
    p_xx = [P[0, 0]]
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
        sg = SIGMA_GYRO * np.sqrt(dt); sa = SIGMA_ACCEL * np.sqrt(dt)
        F_w = np.zeros((6, 4))
        F_w[2, 0] = -dt; F_w[0, 1] = dt; F_w[3, 1] = dt
        F_w[4, 2] = 1.0; F_w[5, 3] = 1.0
        Q = np.diag([sg**2, sa**2,
                     (1e-4*np.sqrt(dt))**2, (1e-3*np.sqrt(dt))**2])
        P = F @ P @ F.T + F_w @ Q @ F_w.T

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
            sm.add_to_map(ScanMatcher.transform_points(
                scan, Exp(np.array([x[0], x[1], x[2]]))))
            traj.append(Exp(np.array([x[0], x[1], x[2]])))
            p_xx.append(P[0, 0])

    return traj, np.array(p_xx)


# ── Metrics ───────────────────────────────────────────────────────────────────

def _ape_components(traj_est, traj_gt):
    n = min(len(traj_est), len(traj_gt))
    ex = np.array([abs(traj_est[i][0, 2] - traj_gt[i][0, 2]) for i in range(n)])
    ey = np.array([abs(traj_est[i][1, 2] - traj_gt[i][1, 2]) for i in range(n)])
    return ex, ey


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    map_pts = _build_corridor_map()
    gt_poses, imu_seq, lidar_seq = _build_sequences(map_pts)

    print("Running DA-IESKF ...")
    traj_da, pxx_da, sigma_r = run_da_ieskf(
        map_pts, copy.deepcopy(imu_seq), copy.deepcopy(lidar_seq))

    print("Running IESKF ...")
    traj_ie, pxx_ie = run_ieskf(
        map_pts, copy.deepcopy(imu_seq), copy.deepcopy(lidar_seq))

    print("Running EKF ...")
    traj_ek, pxx_ek = run_ekf(
        map_pts, copy.deepcopy(imu_seq), copy.deepcopy(lidar_seq))

    n = min(len(traj_da), len(traj_ie), len(traj_ek))
    gt_sampled = [Exp(np.array([SPEED * t, 0.0, 0.0]))
                  for t in np.arange(0, TOTAL_TIME, DT_LIDAR)[:n]]

    ex_da, ey_da = _ape_components(traj_da[:n], gt_sampled)
    ex_ie, ey_ie = _ape_components(traj_ie[:n], gt_sampled)
    ex_ek, ey_ek = _ape_components(traj_ek[:n], gt_sampled)

    t_axis    = np.arange(n) * DT_LIDAR
    n_pxx = min(len(pxx_da), len(pxx_ie), len(pxx_ek), n)

    rmse = lambda e: np.sqrt(np.mean(e**2))

    print("\n=== Results ===")
    print(f"{'Filter':<12} {'x-RMSE':>10} {'y-RMSE':>10} {'APE-RMSE':>10}")
    for name, ex, ey in [('DA-IESKF', ex_da, ey_da),
                          ('IESKF',    ex_ie, ey_ie),
                          ('EKF',      ex_ek, ey_ek)]:
        ape = np.sqrt(ex**2 + ey**2)
        print(f"{name:<12} {rmse(ex):>10.3f} {rmse(ey):>10.3f} {rmse(ape):>10.3f}")

    # ── Feature zone benefit (last 20% of trajectory) ────────────────────────
    feat_start = int(0.8 * n)
    print(f"\n--- Feature zone (t > {t_axis[feat_start]:.0f}s) ---")
    for name, ex in [('DA-IESKF', ex_da), ('IESKF', ex_ie), ('EKF', ex_ek)]:
        print(f"  {name:<12} x-RMSE = {rmse(ex[feat_start:]):.3f} m")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        'Gap C — DA-IESKF vs IESKF vs EKF: Degeneracy-Aware Update\n'
        'Featureless 50m corridor, pillars at x=41,44,47 (feature zone)',
        fontsize=11)

    # Top-view
    ax = axes[0, 0]
    xs_gt = [p[0, 2] for p in gt_sampled]
    ax.plot(xs_gt, [0]*n, 'g-', lw=2.5, label='Ground truth')
    ax.plot([p[0, 2] for p in traj_da[:n]], [p[1, 2] for p in traj_da[:n]],
            'b-',  lw=2,   label=f'DA-IESKF')
    ax.plot([p[0, 2] for p in traj_ie[:n]], [p[1, 2] for p in traj_ie[:n]],
            'r--', lw=1.5, label=f'IESKF')
    ax.plot([p[0, 2] for p in traj_ek[:n]], [p[1, 2] for p in traj_ek[:n]],
            'k:',  lw=1.5, label=f'EKF')
    ax.axhline( CORRIDOR_WIDTH, color='gray', lw=1)
    ax.axhline(-CORRIDOR_WIDTH, color='gray', lw=1)
    ax.axvline(40, color='orange', ls='--', lw=1, label='Feature zone')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title('Top-view trajectories'); ax.legend(fontsize=8)
    ax.set_aspect('equal')

    # Along-corridor error
    ax = axes[0, 1]
    ax.plot(t_axis, ex_da, 'b-',  lw=2,   label=f'DA-IESKF RMSE={rmse(ex_da):.2f}m')
    ax.plot(t_axis, ex_ie, 'r--', lw=1.5, label=f'IESKF    RMSE={rmse(ex_ie):.2f}m')
    ax.plot(t_axis, ex_ek, 'k:',  lw=1.5, label=f'EKF      RMSE={rmse(ex_ek):.2f}m')
    ax.axvline(40/SPEED, color='orange', ls='--', lw=1, label='Feature zone')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Along-corridor error [m]')
    ax.set_title('x-error (weakly observable DoF)'); ax.legend(fontsize=8)

    # x-covariance P[0,0]
    ax = axes[1, 0]
    ax.semilogy(t_axis[:n_pxx], pxx_da[:n_pxx], 'b-',  lw=2,
                label='DA-IESKF P[x,x]')
    ax.semilogy(t_axis[:n_pxx], pxx_ie[:n_pxx], 'r--', lw=1.5,
                label='IESKF    P[x,x]')
    ax.semilogy(t_axis[:n_pxx], pxx_ek[:n_pxx], 'k:',  lw=1.5,
                label='EKF      P[x,x]')
    ax.axvline(40/SPEED, color='orange', ls='--', lw=1, label='Feature zone')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('P[x,x] (log scale)')
    ax.set_title('x-direction covariance\n(larger = more honest uncertainty)')
    ax.legend(fontsize=8)

    # Degeneracy indicator
    ax = axes[1, 1]
    t_sigma = t_axis[:len(sigma_r)]
    ax.plot(t_sigma, sigma_r, 'b-', lw=1.5, label='σ_min / σ_max of H')
    ax.axhline(DEGENERACY_THRESH, color='red', ls='--', lw=1,
               label=f'Threshold = {DEGENERACY_THRESH}')
    ax.axvline(40/SPEED, color='orange', ls='--', lw=1, label='Feature zone')
    ax.fill_between(t_sigma, 0, sigma_r,
                    where=(sigma_r < DEGENERACY_THRESH),
                    alpha=0.25, color='red', label='Degenerate')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('σ_min / σ_max')
    ax.set_title('Degeneracy indicator\n(red shaded = DA filter active)')
    ax.legend(fontsize=8); ax.set_ylim(0, 1)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'gap_c_degeneracy_aware.png')
    plt.savefig(out, dpi=120)
    print(f"\nSaved {out}")


if __name__ == '__main__':
    main()
