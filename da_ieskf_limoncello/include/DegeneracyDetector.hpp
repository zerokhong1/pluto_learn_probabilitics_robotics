#pragma once

#include <Eigen/Dense>
#include <vector>
#include <numeric>
#include <fstream>
#include <string>

/**
 * DegeneracyDetector — Hessian eigenvalue analysis for IESKF degenerate direction
 * covariance inflation.
 *
 * Algorithm:
 *   1. Accumulate H^T V^{-1} H (information matrix) from scan-match residuals.
 *   2. Eigen-decompose the 3×3 position-orientation block.
 *   3. Detect degenerate eigenvectors (eigenvalue ratio < threshold).
 *   4. Inflate P along those directions: P ← P + α V_deg V_deg^T.
 *
 * Reference:
 *   Zhang & Singh, "Laser-visual-inertial odometry and mapping with
 *   high robustness and low drift," JFR 2018, Section III-B.
 *   (Adapted from SE(3) to SGal(3) state space of LIMOncello.)
 */

namespace da_ieskf {

struct DegeneracyResult {
    bool is_degenerate;
    double min_eigenvalue;
    double eigenvalue_ratio;          // lambda_min / lambda_max
    Eigen::VectorXd degenerate_dirs;  // columns are degenerate eigenvectors (dim = state_dim)
    int n_degenerate_dims;
};

class DegeneracyDetector {
public:
    /**
     * @param eigen_thresh   eigenvalue ratio below which a direction is degenerate (default 0.15)
     * @param inflation_alpha  covariance inflation scale along degenerate dirs (default 100.0)
     * @param state_dim      full error-state dimension (LIMOncello SGal(3): 10)
     * @param pose_dim       first pose DOF within state (SGal(3) has 6 pose DOF: ρ,ν,θ;
     *                       for analysis we use the 3 position+orientation dims)
     */
    explicit DegeneracyDetector(double eigen_thresh    = 0.15,
                                 double inflation_alpha = 100.0,
                                 int    state_dim       = 10,
                                 int    pose_dim        = 3)
        : thresh_(eigen_thresh),
          alpha_(inflation_alpha),
          state_dim_(state_dim),
          pose_dim_(pose_dim) {}

    /**
     * Analyse the accumulated information matrix for degeneracy.
     *
     * @param H  n_pts × state_dim Jacobian matrix (stacked residuals)
     * @param W  n_pts × n_pts diagonal weight matrix (V^{-1} per residual)
     * @return   DegeneracyResult
     */
    DegeneracyResult analyse(const Eigen::MatrixXd& H,
                              const Eigen::MatrixXd& W) const {
        // Information matrix restricted to pose block
        Eigen::MatrixXd HtWH = H.transpose() * W * H;  // state_dim × state_dim
        Eigen::MatrixXd A = HtWH.block(0, 0, pose_dim_, pose_dim_);

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(A);
        Eigen::VectorXd eigenvalues  = eig.eigenvalues();   // ascending
        Eigen::MatrixXd eigenvectors = eig.eigenvectors();  // columns

        double lambda_max = eigenvalues.maxCoeff();
        double lambda_min = eigenvalues.minCoeff();

        if (lambda_max < 1e-6) {
            return {false, 0.0, 1.0, Eigen::VectorXd{}, 0};
        }

        double ratio = lambda_min / lambda_max;

        DegeneracyResult res;
        res.is_degenerate    = ratio < thresh_;
        res.min_eigenvalue   = lambda_min;
        res.eigenvalue_ratio = ratio;
        res.n_degenerate_dims = 0;

        // Collect degenerate eigenvectors (embed into full state space)
        std::vector<Eigen::VectorXd> deg_dirs;
        for (int i = 0; i < pose_dim_; ++i) {
            double ev_ratio = (lambda_max > 1e-12) ? eigenvalues(i) / lambda_max : 0.0;
            if (ev_ratio < thresh_) {
                Eigen::VectorXd full_dir = Eigen::VectorXd::Zero(state_dim_);
                full_dir.head(pose_dim_) = eigenvectors.col(i);
                deg_dirs.push_back(full_dir);
                res.n_degenerate_dims++;
            }
        }

        if (!deg_dirs.empty()) {
            res.degenerate_dirs.resize(state_dim_ * static_cast<int>(deg_dirs.size()));
            for (int i = 0; i < static_cast<int>(deg_dirs.size()); ++i)
                res.degenerate_dirs.segment(i * state_dim_, state_dim_) = deg_dirs[i];
        }

        return res;
    }

    /**
     * Analyse a pre-computed information matrix block (HTH / R) directly.
     * More convenient when H^T H is already computed by the IESKF loop.
     *
     * @param info_block  pose_dim × pose_dim information matrix (e.g. H[:,:3]^T H[:,:3] / R)
     * @return            DegeneracyResult
     */
    DegeneracyResult analyse_info(const Eigen::MatrixXd& info_block) const {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(info_block);
        Eigen::VectorXd eigenvalues  = eig.eigenvalues();   // ascending
        Eigen::MatrixXd eigenvectors = eig.eigenvectors();

        double lambda_max = eigenvalues.maxCoeff();
        double lambda_min = eigenvalues.minCoeff();

        // No plane observations (empty/sparse map): H^T H ≈ 0.
        // IESKF gain is already ~0 in this case — inflation is a no-op and
        // would incorrectly flag initialization frames as degenerate.
        if (lambda_max < 1e-6) {
            return {false, 0.0, 1.0, Eigen::VectorXd{}, 0};
        }

        double ratio = lambda_min / lambda_max;

        int actual_pose_dim = static_cast<int>(info_block.rows());

        DegeneracyResult res;
        res.is_degenerate    = ratio < thresh_;
        res.min_eigenvalue   = lambda_min;
        res.eigenvalue_ratio = ratio;
        res.n_degenerate_dims = 0;

        std::vector<Eigen::VectorXd> deg_dirs;
        for (int i = 0; i < actual_pose_dim; ++i) {
            double ev_ratio = (lambda_max > 1e-12) ? eigenvalues(i) / lambda_max : 0.0;
            if (ev_ratio < thresh_) {
                Eigen::VectorXd full_dir = Eigen::VectorXd::Zero(state_dim_);
                full_dir.head(actual_pose_dim) = eigenvectors.col(i);
                deg_dirs.push_back(full_dir);
                res.n_degenerate_dims++;
            }
        }
        if (!deg_dirs.empty()) {
            res.degenerate_dirs.resize(state_dim_ * static_cast<int>(deg_dirs.size()));
            for (int i = 0; i < static_cast<int>(deg_dirs.size()); ++i)
                res.degenerate_dirs.segment(i * state_dim_, state_dim_) = deg_dirs[i];
        }
        return res;
    }

    /**
     * Inflate the filter covariance P along degenerate directions.
     *
     * P_out = P + alpha * Σ_i v_i v_i^T   (sum over degenerate eigenvectors)
     *
     * Accepts both fixed-size (e.g. Matrix<double,24,24>) and dynamic MatrixXd.
     *
     * @param P      state covariance, modified in place
     * @param result DegeneracyResult from analyse() or analyse_info()
     */
    template<typename Derived>
    void inflate(Eigen::MatrixBase<Derived>& P, const DegeneracyResult& result) const {
        if (!result.is_degenerate || result.n_degenerate_dims == 0) return;

        for (int i = 0; i < result.n_degenerate_dims; ++i) {
            Eigen::VectorXd v = result.degenerate_dirs.segment(i * state_dim_, state_dim_);
            P += alpha_ * v * v.transpose();
        }
    }

    /**
     * Soft filter: attenuate information along degenerate eigenvectors.
     * Modifies H in place (nullspace projection of degenerate dirs).
     *
     * This is the alternative to covariance inflation — apply to H before
     * the IESKF gain computation.  The paper compares both; inflation is preferred
     * because it preserves the posterior mean while correctly reporting uncertainty.
     *
     * @param H      Jacobian matrix (n_pts × state_dim), modified in place
     * @param result DegeneracyResult
     */
    void soft_filter_H(Eigen::MatrixXd& H, const DegeneracyResult& result) const {
        if (!result.is_degenerate || result.n_degenerate_dims == 0) return;

        double lambda_max = 0.0;
        // Recompute max eigenvalue from result (not stored directly — use ratio)
        // Simpler: reproject H using eigenvalue-scaled attenuation
        for (int i = 0; i < result.n_degenerate_dims; ++i) {
            Eigen::VectorXd v = result.degenerate_dirs.segment(i * state_dim_, state_dim_);
            // Scale = clip(ratio / thresh, 0, 1)  (matches Python soft filter)
            double scale = std::min(1.0, result.eigenvalue_ratio / thresh_);
            // Project H along v, scale it
            Eigen::VectorXd Hv = H * v;
            H += (scale - 1.0) * Hv * v.transpose();
        }
    }

    double threshold()       const { return thresh_; }
    double inflation_alpha() const { return alpha_;  }

private:
    double thresh_;
    double alpha_;
    int    state_dim_;
    int    pose_dim_;
};

/**
 * EigenvalueLogger — writes per-frame eigenvalue CSV for paper figures.
 *
 * CSV columns: timestamp, lambda1, lambda2, lambda3, ratio, is_degenerate
 */
class EigenvalueLogger {
public:
    explicit EigenvalueLogger(const std::string& path) : path_(path) {
        file_.open(path);
        file_ << "timestamp,lambda1,lambda2,lambda3,ratio,is_degenerate\n";
    }
    ~EigenvalueLogger() { if (file_.is_open()) file_.close(); }

    void log(double t, const DegeneracyResult& res) {
        // eigenvalues stored in ascending order (lambda1 smallest)
        file_ << t << ","
              << res.min_eigenvalue << ","
              << "0.0" << ","       // placeholder for middle eigenvalue
              << "0.0" << ","       // placeholder for max eigenvalue
              << res.eigenvalue_ratio << ","
              << (res.is_degenerate ? 1 : 0) << "\n";
    }

private:
    std::string path_;
    std::ofstream file_;
};

}  // namespace da_ieskf
