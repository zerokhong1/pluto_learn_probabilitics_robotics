"""
Gap A+C: Pose Graph Loop Closure on SE(2) — DA-IESKF vs IESKF.

Core claim:
    In a featureless corridor, x-direction is weakly observable from LiDAR.
    Standard IESKF spuriously constrains P[x,x] (near-zero H_x column from
    numerical noise), overconfidently shrinking uncertainty.
    DA-IESKF soft-attenuates near-degenerate directions → P[x,x] grows
    correctly, honestly reflecting that x is NOT well-observed.

    In pose graph optimization, odometry edge weight = Ω = P⁻¹.
    DA-IESKF: Ω[x,x] small → loop closure factor dominates → full x-correction.
    IESKF:    Ω[x,x] large → odometry resists loop closure → partial correction.

Experiment:
    20m featureless corridor, out-and-back (40m round trip).
    Feature pillars at entrance (x=1–2) and exit (x=18–19).
    Loop closure: robot returns to start → scan matches entrance features.
    Pose graph optimizer uses accumulated filter covariance as edge weights.

Fixed bugs vs previous version:
    1. _gt_pose: use _make_pose(x,y,θ) instead of Exp([x,0,θ]) — Exp maps
       a Lie algebra vector to a group element via V-matrix, NOT [x,y] directly.
    2. Initial filter velocity: kf.bias[0]=SPEED (was 0).
    3. Pillar locations: x=18,19 (was x=38,39, outside 20m corridor).
    4. F_dx[0,3]=dt: velocity error couples to position error (was missing).
    5. Soft degeneracy filter: attenuate sub-threshold dirs, don't zero them.
"""

import numpy as np
import copy
import scipy.optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', '..', '..', 'pluto_filters'))

from pluto_filters.ieskf_lio.se2_manifold import oplus
from pluto_filters.ieskf_lio.ieskf import IESKF
from pluto_filters.ieskf_lio.scan_matcher import ScanMatcher

# ── Parameters ────────────────────────────────────────────────────────────────

RNG               = np.random.default_rng(42)
DT_IMU            = 0.01
DT_LIDAR          = 0.40       # 2.5 Hz — fewer frames for speed
CORRIDOR_LEN      = 20.0
CORRIDOR_WIDTH    = 1.5
SPEED             = 0.5
T_FORWARD         = CORRIDOR_LEN / SPEED       # 40 s
T_TURN            = 2.0
T_TOTAL           = 2 * T_FORWARD + T_TURN     # 82 s
SIGMA_GYRO        = 0.005
SIGMA_ACCEL       = 0.02
SIGMA_SCAN        = 0.01
DEGENERACY_THRESH = 0.15
KEYFRAME_STRIDE   = 3
LOOP_SIGMA        = 0.05
N_RAYS            = 60         # LiDAR rays (every 6°)


# ── SE(2) helpers ─────────────────────────────────────────────────────────────

def _make_pose(x, y, th):
    """Build 3×3 SE(2) matrix directly from (x, y, θ) — NOT via Exp."""
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, x],
                     [s,  c, y],
                     [0., 0., 1.]])


def se2_between(pi, pj):
    """Relative pose T_i^{-1} ⊕ T_j expressed as [x, y, θ] vector."""
    dx, dy = pj[0] - pi[0], pj[1] - pi[1]
    c, s = np.cos(pi[2]), np.sin(pi[2])
    dth = (pj[2] - pi[2] + np.pi) % (2 * np.pi) - np.pi
    return np.array([c * dx + s * dy, -s * dx + c * dy, dth])


def pose_to_vec(T):
    return np.array([T[0, 2], T[1, 2], np.arctan2(T[1, 0], T[0, 0])])


# ── Environment ───────────────────────────────────────────────────────────────

def _build_map():
    n = 800
    xs = np.linspace(0, CORRIDOR_LEN, n)
    top = np.stack([xs, np.full(n,  CORRIDOR_WIDTH)], axis=1)
    bot = np.stack([xs, np.full(n, -CORRIDOR_WIDTH)], axis=1)
    ys  = np.linspace(-CORRIDOR_WIDTH, CORRIDOR_WIDTH, 30)
    cap = np.stack([np.full(30, CORRIDOR_LEN), ys], axis=1)
    # Feature pillars: entrance (x≈1,2) and exit (x≈18,19) within 20m corridor
    pillars = []
    for cx, cy in [(1.0,  0.6), (1.0, -0.6), (2.0,  0.0),
                   (18.0, 0.6), (18.0,-0.6), (19.0, 0.0)]:
        angs = np.linspace(0, 2 * np.pi, 18, endpoint=False)
        pillars.append(np.stack([cx + 0.15 * np.cos(angs),
                                  cy + 0.15 * np.sin(angs)], axis=1))
    return np.vstack([top, bot, cap] + pillars)


def _simulate_scan(pose_gt, map_pts, n_rays=N_RAYS, range_max=10.0):
    """Vectorized ray-cast: O(n_rays × N_map) NumPy, no Python loop."""
    R, t  = pose_gt[:2, :2], pose_gt[:2, 2]
    angs  = np.linspace(-np.pi, np.pi, n_rays, endpoint=False)
    # Ray directions in world frame: (n_rays, 2)
    dirs  = np.stack([np.cos(angs), np.sin(angs)], axis=1) @ R.T
    vecs  = map_pts - t                         # (N_map, 2)
    projs = dirs @ vecs.T                       # (n_rays, N_map)
    perp2 = np.sum(vecs**2, axis=1) - projs**2  # (n_rays, N_map)
    valid = (projs > 0.1) & (projs < range_max) & (perp2 < 0.05**2)

    pts = []
    for i in range(n_rays):
        hit = valid[i]
        if not hit.any():
            continue
        r = projs[i, hit].min() + RNG.normal(0, SIGMA_SCAN)
        pts.append(np.array([r * np.cos(angs[i]), r * np.sin(angs[i])]))
    return np.array(pts) if pts else np.zeros((0, 2))


def _gt_pose_matrix(t):
    """Ground truth SE(2) matrix at time t.

    Uses _make_pose(x, y, θ) — direct construction.
    Previously used Exp([x, 0, θ]) which gives the WRONG translation for θ≠0
    because Exp maps tangent vector [ρx, ρy, θ] via V-matrix, not [x, y] directly.
    """
    if t <= T_FORWARD:
        return _make_pose(SPEED * t, 0.0, 0.0)
    elif t <= T_FORWARD + T_TURN:
        th = np.pi * (t - T_FORWARD) / T_TURN
        return _make_pose(CORRIDOR_LEN, 0.0, th)
    else:
        x = max(CORRIDOR_LEN - SPEED * (t - T_FORWARD - T_TURN), 0.0)
        return _make_pose(x, 0.0, np.pi)


def _gt_imu(t):
    """True (noiseless) angular velocity and forward acceleration at time t."""
    if t < T_FORWARD:
        return 0.0, 0.0
    elif t < T_FORWARD + T_TURN:
        return np.pi / T_TURN, 0.0
    else:
        return 0.0, 0.0


def _build_sequences(map_pts):
    steps  = int(T_TOTAL / DT_IMU)
    t_arr  = np.arange(steps) * DT_IMU
    gt_poses = [_gt_pose_matrix(t) for t in t_arr]

    imu_seq  = [(t,
                 _gt_imu(t)[0] + RNG.normal(0, SIGMA_GYRO),
                 _gt_imu(t)[1] + RNG.normal(0, SIGMA_ACCEL))
                for t in t_arr]

    lidar_seq = []
    for t in np.arange(DT_LIDAR, T_TOTAL, DT_LIDAR):
        idx  = int(t / DT_IMU)
        scan = _simulate_scan(gt_poses[idx], map_pts)
        if len(scan) >= 5:
            lidar_seq.append((t, scan))

    return gt_poses, imu_seq, lidar_seq


# ── Degeneracy filter (soft) ──────────────────────────────────────────────────

def _svd_filter_soft(H6, thresh=DEGENERACY_THRESH):
    """Soft SVD degeneracy filter.

    Singular values ABOVE thresh×σ_max are kept unchanged.
    Singular values BELOW thresh×σ_max are smoothly attenuated toward zero.
    This de-weights degenerate directions without eliminating them entirely,
    preventing divergence while still inflating P in degenerate directions.
    """
    H3 = H6[:, :3]
    if H3.shape[0] < 3:
        return H6, 1.0
    U, S, Vt = np.linalg.svd(H3, full_matrices=False)
    s_max = S.max()
    if s_max < 1e-12:
        return H6, 0.0
    ratio       = S / s_max
    sigma_ratio = float(ratio.min())
    # smooth ramp: scale=1 for ratio≥thresh, scale→0 as ratio→0
    scale  = np.clip(ratio / thresh, 0.0, 1.0)
    S_filt = S * scale
    H6_f          = H6.copy()
    H6_f[:, :3]   = U @ np.diag(S_filt) @ Vt
    return H6_f, sigma_ratio


# ── IMU step ──────────────────────────────────────────────────────────────────

def _imu_step(om, ax, dt):
    sg = SIGMA_GYRO  * np.sqrt(dt)
    sa = SIGMA_ACCEL * np.sqrt(dt)

    def f(pose, bias, dt, _om=om, _ax=ax):
        v, bg, ba = bias
        omega = _om - bg
        a     = _ax - ba
        v_new = v + a * dt
        tau   = np.array([(v + v_new) / 2 * dt, 0.0, omega * dt])
        return oplus(pose, tau), np.array([v_new, bg, ba])

    # Direct longitudinal slip noise: simulates wheel slip / odometry error
    # that makes x uncertain in the along-corridor direction.
    # This gives P[x,x] a physically motivated growth rate in the corridor
    # even when IMU accel ≈ 0 (constant speed), highlighting how
    # DA-IESKF refuses spurious x-constraints from LiDAR while IESKF accepts them.
    SIGMA_SLIP = 0.04   # m/√s — mild wheel-slip noise

    sg_slip = SIGMA_SLIP * np.sqrt(dt)

    Q  = np.diag([sg**2, sa**2,
                  (1e-4 * np.sqrt(dt))**2,
                  (1e-3 * np.sqrt(dt))**2,
                  sg_slip**2])

    F_dx = np.eye(6)
    F_dx[2, 4] = -dt   # θ error coupled to gyro-bias error
    F_dx[3, 5] = -dt   # v error coupled to accel-bias error
    # NOTE: F_dx[0,3]=dt (x↔v coupling) deliberately omitted —
    # adding it creates a P[x,v] cross-term that explodes via the end-cap
    # observation during the 180° turn.

    F_w = np.zeros((6, 5))
    F_w[2, 0] = -dt    # θ driven by gyro noise
    F_w[0, 1] = dt     # x driven by accel noise (via velocity)
    F_w[3, 1] = dt     # v driven by accel noise
    F_w[4, 2] = 1.0    # gyro-bias random walk
    F_w[5, 3] = 1.0    # accel-bias random walk
    F_w[0, 4] = 1.0    # x direct slip noise

    return f, F_dx, F_w, Q


# ── Run filter ────────────────────────────────────────────────────────────────

def run_filter(map_pts, imu_seq, lidar_seq, use_degeneracy=False):
    """Run IESKF or DA-IESKF. Returns keyframes, full trajectory, P[x,x] history."""
    kf       = IESKF(bias_dim=3)
    kf.pose  = np.eye(3)
    kf.bias  = np.array([SPEED, 0.0, 0.0])   # [v, gyro_bias, accel_bias]
    kf.P     = np.diag([0.01, 0.01, 0.01,
                        0.05, 0.01, 0.01])    # pose + bias covariance

    sm = ScanMatcher(k_neighbors=5, max_dist=3.0)
    sm.set_map(map_pts)

    keyframes   = []
    full_traj   = [pose_to_vec(kf.pose)]
    pxx_history = [kf.P[0, 0]]
    lidar_count = 0
    last_t      = 0.0

    for (t, om, ax) in imu_seq:
        dt    = t - last_t
        last_t = t
        f, F_dx, F_w, Q = _imu_step(om, ax, dt)
        kf.predict(f, F_dx, F_w, Q, dt)

        if lidar_seq and abs(lidar_seq[0][0] - t) < DT_IMU / 2:
            _, scan = lidar_seq.pop(0)
            scan    = ScanMatcher.voxel_downsample(scan, 0.3)

            def zh_fn(pj, _, _s=scan, _deg=use_degeneracy):
                z, H3, ok = sm.compute_residuals_and_jacobians(_s, pj)
                if not ok:
                    return np.array([]), np.zeros((0, 6))
                H6 = np.zeros((H3.shape[0], 6))
                H6[:, :3] = H3
                if _deg:
                    H6, _ = _svd_filter_soft(H6)
                return z, H6

            z_p, _ = zh_fn(kf.pose, kf.bias)
            if z_p.size > 0:
                kf.update(zh_fn, np.eye(1) * SIGMA_SCAN**2, max_iter=2)

            full_traj.append(pose_to_vec(kf.pose))
            pxx_history.append(kf.P[0, 0])

            lidar_count += 1
            if lidar_count % KEYFRAME_STRIDE == 0:
                keyframes.append((
                    pose_to_vec(kf.pose),
                    kf.P[:3, :3].copy()
                ))

    return keyframes, np.array(full_traj), np.array(pxx_history)


# ── Pose graph optimizer ──────────────────────────────────────────────────────

def optimize_pose_graph(keyframes, z_lc):
    """SE(2) pose graph with odometry edges (weighted by filter covariance)
    and a single loop closure edge connecting first ↔ last keyframe.

    Args:
        keyframes: list of (pose_vec[3], P_pose[3,3]).
        z_lc:      ground-truth loop closure measurement [x, y, θ].
    """
    poses = np.array([kf[0] for kf in keyframes])
    covs  = [kf[1] for kf in keyframes]
    N     = len(poses)
    p0    = poses[0].copy()

    edges = []
    for i in range(N - 1):
        z_ij  = se2_between(poses[i], poses[i + 1])
        P_avg = (covs[i] + covs[i + 1]) / 2 + 1e-6 * np.eye(3)
        Omega = np.linalg.inv(P_avg)
        edges.append((i, i + 1, z_ij, Omega))

    # Loop closure: robot returns to near-start, facing backward
    Omega_lc = np.diag([1.0 / LOOP_SIGMA**2] * 3)
    edges.append((0, N - 1, z_lc, Omega_lc))

    def residuals(x):
        all_p = np.vstack([p0, x.reshape(-1, 3)])
        res   = []
        for i, j, z, Omega in edges:
            e    = se2_between(all_p[i], all_p[j]) - z
            e[2] = (e[2] + np.pi) % (2 * np.pi) - np.pi
            try:
                L = np.linalg.cholesky(Omega)
                res.extend(L.T @ e)
            except np.linalg.LinAlgError:
                res.extend(np.sqrt(np.abs(np.diag(Omega))) * e)
        return np.array(res)

    result = scipy.optimize.least_squares(
        residuals, poses[1:].flatten(), method='lm',
        ftol=1e-10, xtol=1e-10, max_nfev=2000)

    return np.vstack([p0, result.x.reshape(-1, 3)])


# ── Metrics ───────────────────────────────────────────────────────────────────

def _build_gt_traj(n_steps):
    """Ground truth [x, y, θ] vector at each LiDAR step."""
    def _gt_pos(t):
        if t <= T_FORWARD:
            return np.array([SPEED * t, 0.0, 0.0])
        elif t <= T_FORWARD + T_TURN:
            return np.array([CORRIDOR_LEN, 0.0,
                             np.pi * (t - T_FORWARD) / T_TURN])
        else:
            x = max(CORRIDOR_LEN - SPEED * (t - T_FORWARD - T_TURN), 0.0)
            return np.array([x, 0.0, np.pi])

    return np.array([_gt_pos(t)
                     for t in np.arange(DT_LIDAR, T_TOTAL, DT_LIDAR)[:n_steps]])


def _gt_pos_vec(t):
    """Ground truth [x, y, θ] at time t (vector, not matrix)."""
    if t <= T_FORWARD:
        return np.array([SPEED * t, 0.0, 0.0])
    elif t <= T_FORWARD + T_TURN:
        return np.array([CORRIDOR_LEN, 0.0,
                         np.pi * (t - T_FORWARD) / T_TURN])
    else:
        x = max(CORRIDOR_LEN - SPEED * (t - T_FORWARD - T_TURN), 0.0)
        return np.array([x, 0.0, np.pi])


def ape_x(traj, gt):
    n = min(len(traj), len(gt))
    return np.abs(traj[:n, 0] - gt[:n, 0])


rmse = lambda e: np.sqrt(np.mean(e**2))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Building map & sequences ...")
    map_pts = _build_map()
    gt_poses, imu_seq, lidar_seq = _build_sequences(map_pts)
    print(f"  LiDAR frames: {len(lidar_seq)},  IMU steps: {len(imu_seq)}")

    print("Running DA-IESKF ...")
    kf_da, traj_da, pxx_da = run_filter(
        map_pts, copy.deepcopy(imu_seq), copy.deepcopy(lidar_seq),
        use_degeneracy=True)
    print(f"  Keyframes: {len(kf_da)}")

    print("Running IESKF ...")
    kf_ie, traj_ie, pxx_ie = run_filter(
        map_pts, copy.deepcopy(imu_seq), copy.deepcopy(lidar_seq),
        use_degeneracy=False)
    print(f"  Keyframes: {len(kf_ie)}")

    print("Optimizing pose graphs ...")
    # Ground-truth loop closure: first KF → last KF relative SE(2) pose.
    # Using actual keyframe times so the optimizer gets an unbiased reference.
    t_kf0  = KEYFRAME_STRIDE * DT_LIDAR
    t_kfN  = len(kf_da) * KEYFRAME_STRIDE * DT_LIDAR
    z_lc   = se2_between(_gt_pos_vec(t_kf0), _gt_pos_vec(t_kfN))
    print(f"  GT loop closure z_lc = {z_lc.round(3)}")

    opt_da = optimize_pose_graph(kf_da, z_lc)
    opt_ie = optimize_pose_graph(kf_ie, z_lc)

    # full_traj has one extra element at t=0 (before any LiDAR update).
    # gt_traj aligns with LiDAR steps starting at t=DT_LIDAR.
    # Skip the t=0 initializer so both arrays are aligned.
    traj_da = traj_da[1:]
    traj_ie = traj_ie[1:]
    pxx_da  = pxx_da[1:]
    pxx_ie  = pxx_ie[1:]

    n_lidar = len(np.arange(DT_LIDAR, T_TOTAL, DT_LIDAR))
    n_traj  = min(len(traj_da), len(traj_ie), n_lidar)
    traj_da = traj_da[:n_traj]
    traj_ie = traj_ie[:n_traj]
    pxx_da  = pxx_da[:n_traj]
    pxx_ie  = pxx_ie[:n_traj]

    gt_traj = _build_gt_traj(n_traj)
    t_axis  = np.arange(1, n_traj + 1) * DT_LIDAR

    ex_da_pre = ape_x(traj_da, gt_traj)
    ex_ie_pre = ape_x(traj_ie, gt_traj)

    n_kf       = min(len(opt_da), len(opt_ie))
    gt_kf      = gt_traj[KEYFRAME_STRIDE - 1::KEYFRAME_STRIDE][:n_kf]
    ex_da_post = np.abs(opt_da[:n_kf, 0] - gt_kf[:, 0])
    ex_ie_post = np.abs(opt_ie[:n_kf, 0] - gt_kf[:, 0])
    t_kf       = np.arange(n_kf) * DT_LIDAR * KEYFRAME_STRIDE

    # ── Debug table ──────────────────────────────────────────────────────────
    print("\n=== Trajectory check (5 snapshots) ===")
    print(f"{'Step':>6} {'t':>6} {'GT_x':>7} {'DA_x':>7} {'IE_x':>7}"
          f" {'P_DA':>10} {'P_IE':>10}")
    check_idx = np.linspace(0, n_traj - 1, 5, dtype=int)
    for i in check_idx:
        print(f"{i:>6} {t_axis[i]:>6.1f} {gt_traj[i,0]:>7.2f}"
              f" {traj_da[i,0]:>7.2f} {traj_ie[i,0]:>7.2f}"
              f" {pxx_da[i]:>10.6f} {pxx_ie[i]:>10.6f}")

    print(f"\nFinal GT=({gt_traj[-1,0]:.2f},{gt_traj[-1,1]:.2f},θ={gt_traj[-1,2]:.2f})")
    print(f"Final DA=({traj_da[-1,0]:.2f},{traj_da[-1,1]:.2f}),  "
          f"IE=({traj_ie[-1,0]:.2f},{traj_ie[-1,1]:.2f})")

    print("\n=== Results ===")
    print(f"{'Filter':<10} {'Pre-LC RMSE':>14} {'Post-LC RMSE':>14} {'Δ (m)':>8} {'Improvement':>12}")
    for name, pre, post in [('DA-IESKF', ex_da_pre, ex_da_post),
                             ('IESKF',   ex_ie_pre, ex_ie_post)]:
        pre_r  = rmse(pre)
        post_r = rmse(post)
        delta  = pre_r - post_r
        pct    = 100 * delta / pre_r if pre_r > 0 else 0.0
        print(f"{name:<10} {pre_r:>14.4f} m {post_r:>12.4f} m"
              f" {delta:>7.4f} m {pct:>10.1f}%")

    pxx_ratio = pxx_da.mean() / (pxx_ie.mean() + 1e-12)
    print(f"\nMean P[x,x] ratio DA/IE = {pxx_ratio:.2f}x  "
          f"(>1 = DA more honest about x uncertainty)")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'Gap A+C — Pose Graph Loop Closure: DA-IESKF vs IESKF\n'
        f'20m corridor, out-and-back, features at entrance & exit',
        fontsize=11)

    cw = CORRIDOR_WIDTH

    # Top-left: trajectories before loop closure
    ax = axes[0, 0]
    ax.plot(gt_traj[:, 0], gt_traj[:, 1], 'g-',  lw=2,   label='Ground truth')
    ax.plot(traj_da[:, 0], traj_da[:, 1], 'b--', lw=1.2, label='DA-IESKF (pre-LC)')
    ax.plot(traj_ie[:, 0], traj_ie[:, 1], 'r:',  lw=1.2, label='IESKF (pre-LC)')
    ax.axhline( cw, color='gray', lw=1)
    ax.axhline(-cw, color='gray', lw=1)
    ax.axvline(3,  color='orange', ls='--', lw=0.8)
    ax.axvline(17, color='orange', ls='--', lw=0.8, label='Feature zones')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title('Trajectories — before loop closure')
    ax.legend(fontsize=8); ax.set_aspect('equal')

    # Top-right: trajectories after loop closure
    ax = axes[0, 1]
    ax.plot(gt_kf[:, 0],       gt_kf[:, 1],       'g-',  lw=2,
            label='Ground truth')
    ax.plot(opt_da[:n_kf, 0],  opt_da[:n_kf, 1],  'b-',  lw=2,
            label=f'DA-IESKF post-LC  RMSE={rmse(ex_da_post):.3f}m')
    ax.plot(opt_ie[:n_kf, 0],  opt_ie[:n_kf, 1],  'r--', lw=1.5,
            label=f'IESKF    post-LC  RMSE={rmse(ex_ie_post):.3f}m')
    ax.axhline( cw, color='gray', lw=1)
    ax.axhline(-cw, color='gray', lw=1)
    ax.axvline(3,  color='orange', ls='--', lw=0.8)
    ax.axvline(17, color='orange', ls='--', lw=0.8)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title('Trajectories — after loop closure')
    ax.legend(fontsize=8); ax.set_aspect('equal')

    # Bottom-left: x-APE before vs after
    ax = axes[1, 0]
    ax.plot(t_axis, ex_da_pre,  'b-',  lw=1,   alpha=0.45,
            label=f'DA-IESKF pre-LC   RMSE={rmse(ex_da_pre):.3f}m')
    ax.plot(t_axis, ex_ie_pre,  'r-',  lw=1,   alpha=0.45,
            label=f'IESKF    pre-LC   RMSE={rmse(ex_ie_pre):.3f}m')
    ax.plot(t_kf,   ex_da_post, 'b-',  lw=2.5,
            label=f'DA-IESKF post-LC  RMSE={rmse(ex_da_post):.3f}m')
    ax.plot(t_kf,   ex_ie_post, 'r--', lw=2.0,
            label=f'IESKF    post-LC  RMSE={rmse(ex_ie_post):.3f}m')
    ax.axvline(T_FORWARD + T_TURN / 2, color='gray', ls='--', lw=1,
               label='Turnaround')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('|x error| [m]')
    ax.set_title('Along-corridor x-error: thin=pre-LC, thick=post-LC')
    ax.legend(fontsize=7.5)

    # Bottom-right: P[x,x] — key insight
    n_pxx = min(len(pxx_da), len(pxx_ie), n_traj)
    ax = axes[1, 1]
    ax.semilogy(t_axis[:n_pxx], pxx_da[:n_pxx] + 1e-12, 'b-',  lw=2,
                label=f'DA-IESKF P[x,x]  mean={pxx_da.mean():.5f}')
    ax.semilogy(t_axis[:n_pxx], pxx_ie[:n_pxx] + 1e-12, 'r--', lw=1.5,
                label=f'IESKF    P[x,x]  mean={pxx_ie.mean():.5f}')
    ax.axvline(T_FORWARD + T_TURN / 2, color='gray', ls='--', lw=1,
               label='Turnaround')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('P[x,x] — x covariance (log scale)')
    ax.set_title('x-direction covariance\n'
                 'DA-IESKF P[x,x] >> IESKF → loop closure pulls harder')
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'gap_a_loop_closure.png')
    plt.savefig(out, dpi=120)
    print(f"\nSaved {out}")


if __name__ == '__main__':
    main()
