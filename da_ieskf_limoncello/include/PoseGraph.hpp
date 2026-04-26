#pragma once

// Pure Eigen pose graph optimizer — no GTSAM, no Boost, no ABI conflicts.
// Replaces the GTSAM version which caused heap corruption inside ROS2 callbacks
// due to boost::ptr_map interacting with ROS2's allocator/thread model.
//
// Math: Gauss-Newton on SE(3) pose graph, sparse LDLT (O(N) for chain).
//   Odometry factor: e = Log(T_meas⁻¹ · T_from⁻¹ · T_to),  Ω = Σ⁻¹
//   Loop closure:    same form, fixed tight Ω
//   DA-IESKF → large Σ along degenerate dirs → small Ω → LC dominates ✓
//   Std  IESKF → small Σ everywhere → large Ω → LC rejected            ✗

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <cmath>

namespace da_ieskf {

// ── SE(3) / SO(3) helpers ────────────────────────────────────────────────────

namespace se3 {

inline Eigen::Matrix3d so3_exp(const Eigen::Vector3d& phi)
{
    double theta = phi.norm();
    if (theta < 1e-10)
        return Eigen::Matrix3d::Identity() +
               (Eigen::Matrix3d() <<    0, -phi(2),  phi(1),
                                    phi(2),      0, -phi(0),
                                   -phi(1),  phi(0),      0).finished();
    Eigen::Vector3d a = phi / theta;
    Eigen::Matrix3d A;
    A <<     0, -a(2),  a(1),
          a(2),     0, -a(0),
         -a(1),  a(0),     0;
    return Eigen::Matrix3d::Identity()
           + std::sin(theta) * A
           + (1.0 - std::cos(theta)) * A * A;
}

inline Eigen::Vector3d so3_log(const Eigen::Matrix3d& R)
{
    double cos_a = std::max(-1.0, std::min(1.0, (R.trace() - 1.0) / 2.0));
    double theta = std::acos(cos_a);
    Eigen::Vector3d v(R(2,1) - R(1,2), R(0,2) - R(2,0), R(1,0) - R(0,1));
    if (theta < 1e-10) return v * 0.5;
    return v * (theta / (2.0 * std::sin(theta)));
}

// SE(3) Exp: ξ = [ρ(3), φ(3)] → 4×4 matrix
inline Eigen::Matrix4d exp(const Eigen::Matrix<double,6,1>& xi)
{
    Eigen::Vector3d rho = xi.head<3>();
    Eigen::Vector3d phi = xi.tail<3>();
    Eigen::Matrix3d R = so3_exp(phi);

    double theta = phi.norm();
    Eigen::Matrix3d V;
    if (theta < 1e-10) {
        V = Eigen::Matrix3d::Identity();
    } else {
        Eigen::Vector3d a = phi / theta;
        Eigen::Matrix3d A;
        A <<     0, -a(2),  a(1),
              a(2),     0, -a(0),
             -a(1),  a(0),     0;
        V = Eigen::Matrix3d::Identity()
            + ((1.0 - std::cos(theta)) / theta) * A
            + ((theta - std::sin(theta)) / theta) * A * A;
    }

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,3>(0,0) = R;
    T.block<3,1>(0,3) = V * rho;
    return T;
}

// SE(3) Log: 4×4 matrix → ξ = [ρ(3), φ(3)]
inline Eigen::Matrix<double,6,1> log(const Eigen::Matrix4d& T)
{
    Eigen::Matrix3d R = T.block<3,3>(0,0);
    Eigen::Vector3d t = T.block<3,1>(0,3);
    Eigen::Vector3d phi = so3_log(R);

    double theta = phi.norm();
    Eigen::Matrix3d V_inv;
    if (theta < 1e-10) {
        V_inv = Eigen::Matrix3d::Identity();
    } else {
        Eigen::Vector3d a = phi / theta;
        Eigen::Matrix3d A;
        A <<     0, -a(2),  a(1),
              a(2),     0, -a(0),
             -a(1),  a(0),     0;
        V_inv = Eigen::Matrix3d::Identity()
                - 0.5 * A
                + (1.0 / (theta * theta)
                   - (1.0 + std::cos(theta)) / (2.0 * theta * std::sin(theta)))
                  * A * A;
    }

    Eigen::Matrix<double,6,1> xi;
    xi.head<3>() = V_inv * t;
    xi.tail<3>() = phi;
    return xi;
}

// Ta⁻¹ · Tb
inline Eigen::Matrix4d between(const Eigen::Matrix4d& Ta, const Eigen::Matrix4d& Tb)
{
    Eigen::Matrix3d Rt = Ta.block<3,3>(0,0).transpose();
    Eigen::Matrix4d inv = Eigen::Matrix4d::Identity();
    inv.block<3,3>(0,0) = Rt;
    inv.block<3,1>(0,3) = -Rt * Ta.block<3,1>(0,3);
    return inv * Tb;
}

}  // namespace se3

// ── Factor types ─────────────────────────────────────────────────────────────

struct OdomFactor {
    int from_id, to_id;
    Eigen::Matrix4d              T_meas;
    Eigen::Matrix<double,6,6>   info;   // Ω = Σ⁻¹
};

struct LCFactor {
    int from_id, to_id;
    Eigen::Matrix4d              T_meas;
    Eigen::Matrix<double,6,6>   info;
};

// ── Config ───────────────────────────────────────────────────────────────────

struct PGConfig {
    double lc_fitness_threshold = 0.3;   // m² — separates genuine (0.04m²) from marginal (0.6m²+)
    double lc_noise_translation = 0.2;   // m  — 1.5m voxel ICP accuracy ~0.2m RMSE
    double lc_noise_rotation    = 0.02;  // rad
    int    max_iterations       = 10;
    double convergence_tol      = 1e-6;
    bool   verbose              = false;
};

// ── PoseGraph ────────────────────────────────────────────────────────────────

class PoseGraph {
public:
    PoseGraph() = default;
    explicit PoseGraph(const PGConfig& cfg) : cfg_(cfg) {}

    // Add sequential odometry factor (covariance from IESKF error-state P).
    // DA-IESKF: large cov along degenerate dirs → small info → LC dominates ✓
    // Std IESKF: small cov everywhere → large info → LC blocked             ✗
    void addOdometryFactor(int from_id, int to_id,
                           const Eigen::Matrix4d& T_rel,
                           const Eigen::Matrix<double,6,6>& covariance)
    {
        grow(std::max(from_id, to_id));

        OdomFactor f;
        f.from_id = from_id;
        f.to_id   = to_id;
        f.T_meas  = T_rel;
        // Regularize then invert
        Eigen::Matrix<double,6,6> cov = covariance;
        cov.diagonal().array() += 1e-8;
        f.info = cov.inverse();
        odom_.push_back(f);

        if (!init_[to_id])
            poses_[to_id] = poses_[from_id] * T_rel, init_[to_id] = true;
    }

    // Add loop closure factor (fitness_score from GICP).
    // Rejected if score > lc_fitness_threshold.
    void addLoopClosureFactor(int from_id, int to_id,
                              const Eigen::Matrix4d& T_rel,
                              double fitness_score)
    {
        if (fitness_score > cfg_.lc_fitness_threshold) return;

        Eigen::Matrix<double,6,6> info = Eigen::Matrix<double,6,6>::Zero();
        double ir = 1.0 / (cfg_.lc_noise_rotation    * cfg_.lc_noise_rotation);
        double it = 1.0 / (cfg_.lc_noise_translation * cfg_.lc_noise_translation);
        info(0,0) = it; info(1,1) = it; info(2,2) = it;
        info(3,3) = ir; info(4,4) = ir; info(5,5) = ir;

        LCFactor f{from_id, to_id, T_rel, info};
        lc_.push_back(f);
        has_new_lc_ = true;
    }

    // Gauss-Newton optimization. Returns true if loop closures were present.
    // Gauss-Newton with sparse LDLT — O(N) for a chain graph.
    // Dense LDLT was O(N³) and took ~12 s at N=255, blocking the ROS2 executor.
    bool optimize()
    {
        int N = static_cast<int>(poses_.size());
        if (N <= 1) return false;

        int opt = N - 1;   // node 0 is anchor
        int dim = opt * 6;

        using SpMat = Eigen::SparseMatrix<double>;
        using Trip   = Eigen::Triplet<double>;

        Eigen::SimplicialLDLT<SpMat> solver;
        bool solver_analysed = false;

        for (int iter = 0; iter < cfg_.max_iterations; ++iter) {
            std::vector<Trip> trips;
            trips.reserve((odom_.size() + lc_.size()) * 4 * 36);
            Eigen::VectorXd b = Eigen::VectorXd::Zero(dim);

            for (const auto& f : odom_) accumulate_sparse(f.from_id, f.to_id, f.T_meas, f.info, trips, b);
            for (const auto& f : lc_)   accumulate_sparse(f.from_id, f.to_id, f.T_meas, f.info, trips, b);

            // Tikhonov regularisation
            for (int k = 0; k < dim; ++k)
                trips.emplace_back(k, k, 1e-6);

            SpMat H(dim, dim);
            H.setFromTriplets(trips.begin(), trips.end());

            if (!solver_analysed) {
                solver.analyzePattern(H);
                solver_analysed = true;
            }
            solver.factorize(H);
            if (solver.info() != Eigen::Success) break;

            Eigen::VectorXd dx = solver.solve(-b);
            if (solver.info() != Eigen::Success) break;

            double max_d = 0.0;
            for (int i = 1; i < N; ++i) {
                auto dxi = dx.segment<6>((i-1) * 6);
                poses_[i] = poses_[i] * se3::exp(dxi);
                max_d = std::max(max_d, dxi.norm());
            }
            if (max_d < cfg_.convergence_tol) break;
        }

        bool had = has_new_lc_;
        has_new_lc_ = false;
        return had;
    }

    Eigen::Matrix4d getOptimizedPose(int id) const
    {
        if (id >= 0 && id < static_cast<int>(poses_.size()) && init_[id])
            return poses_[id];
        return Eigen::Matrix4d::Identity();
    }

    // Compatibility alias used by main.cpp
    Eigen::Matrix4d getOptimizedPoseEigen(int id) const { return getOptimizedPose(id); }

    bool hasNewLoopClosures() const { return has_new_lc_; }
    int  loopClosureCount()   const { return static_cast<int>(lc_.size()); }
    int  odometryCount()      const { return static_cast<int>(odom_.size()); }

private:
    void grow(int id)
    {
        if (id >= static_cast<int>(poses_.size())) {
            poses_.resize(id + 1, Eigen::Matrix4d::Identity());
            init_.resize(id + 1, false);
        }
        if (id == 0 && !init_[0]) init_[0] = true;
    }

    using Trip = Eigen::Triplet<double>;

    void accumulate_sparse(int fi, int ti,
                           const Eigen::Matrix4d& T_meas,
                           const Eigen::Matrix<double,6,6>& info,
                           std::vector<Trip>& trips,
                           Eigen::VectorXd& b) const
    {
        Eigen::Matrix<double,6,1> e =
            se3::log(se3::between(T_meas, se3::between(poses_[fi], poses_[ti])));

        Eigen::Matrix<double,6,1> Oe = info * e;

        int ai = fi - 1;
        int bi = ti - 1;
        int sz = static_cast<int>(b.size());

        auto add_block = [&](int r, int c, double sign) {
            for (int rr = 0; rr < 6; ++rr)
                for (int cc = 0; cc < 6; ++cc)
                    trips.emplace_back(r + rr, c + cc, sign * info(rr, cc));
        };

        if (ai >= 0 && ai * 6 + 5 < sz) {
            add_block(ai*6, ai*6, +1.0);
            b.segment<6>(ai*6) -= Oe;
        }
        if (bi >= 0 && bi * 6 + 5 < sz) {
            add_block(bi*6, bi*6, +1.0);
            b.segment<6>(bi*6) += Oe;
        }
        if (ai >= 0 && bi >= 0 && ai*6+5 < sz && bi*6+5 < sz) {
            add_block(ai*6, bi*6, -1.0);
            add_block(bi*6, ai*6, -1.0);
        }
    }

    PGConfig                     cfg_;
    std::vector<Eigen::Matrix4d> poses_;
    std::vector<bool>            init_;
    std::vector<OdomFactor>      odom_;
    std::vector<LCFactor>        lc_;
    bool                         has_new_lc_ = false;
};

}  // namespace da_ieskf
