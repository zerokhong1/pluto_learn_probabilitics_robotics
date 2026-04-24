#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/inference/Symbol.h>

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <fstream>

/**
 * PoseGraph — GTSAM-based pose graph wrapper for DA-IESKF loop closure.
 *
 * Workflow:
 *   1. addOdometry() — called at each keyframe with filter pose + P
 *   2. addLoopClosure() — called when Scan Context detects a revisit
 *   3. optimize() — runs LM, returns corrected trajectory
 *   4. applyCorrections() — hands back per-keyframe T_correction for i-Octree update
 *
 * Between-factor information matrix: Ω = P^{-1} from the filter posterior.
 * For DA-IESKF, inflated P along degenerate dirs → smaller Ω → LC dominates.
 * For IESKF,    overconfident small P → large Ω → LC cannot overcome odometry.
 *
 * This asymmetry is the central mechanism demonstrated in the paper.
 */

namespace da_ieskf {

using gtsam::symbol_shorthand::X;  // keyframe poses

struct KeyframePose {
    double timestamp;
    gtsam::Pose3 pose;
    Eigen::Matrix<double, 6, 6> covariance;  // filter P[pose block], 6×6 for SE(3)
    int id;
};

struct LoopClosureEdge {
    int from_id;
    int to_id;
    gtsam::Pose3 relative_pose;
    Eigen::Matrix<double, 6, 6> noise_cov;  // from GICP fitness / ICP covariance
};

class PoseGraph {
public:
    explicit PoseGraph(bool verbose = false) : verbose_(verbose), next_id_(0) {}

    /**
     * Add an odometry keyframe.
     *
     * @param t         timestamp
     * @param pose      SE(3) pose from filter
     * @param P_pose    6×6 filter covariance (pose block only)
     */
    void addKeyframe(double t, const gtsam::Pose3& pose,
                     const Eigen::Matrix<double, 6, 6>& P_pose) {
        int id = next_id_++;

        KeyframePose kf;
        kf.timestamp  = t;
        kf.pose       = pose;
        kf.covariance = P_pose;
        kf.id         = id;
        keyframes_.push_back(kf);

        // Add to initial values
        initial_values_.insert(X(id), pose);

        if (id == 0) {
            // Prior on first keyframe (very tight)
            auto prior_noise = gtsam::noiseModel::Gaussian::Covariance(
                1e-6 * Eigen::Matrix<double, 6, 6>::Identity());
            graph_.add(gtsam::PriorFactor<gtsam::Pose3>(X(0), pose, prior_noise));
        } else {
            // Between-factor from odometry
            gtsam::Pose3 T_prev = keyframes_[id - 1].pose;
            gtsam::Pose3 T_rel  = T_prev.inverse() * pose;

            // Ω = P^{-1} — key: DA-IESKF has larger P in degenerate directions
            Eigen::Matrix<double, 6, 6> Omega = P_pose.inverse();
            auto odom_noise = gtsam::noiseModel::Gaussian::Information(Omega);
            graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
                X(id - 1), X(id), T_rel, odom_noise));
        }

        if (verbose_)
            std::cout << "[PoseGraph] Added keyframe " << id
                      << " at t=" << t << "\n";
    }

    /**
     * Add a loop closure edge (from GICP-verified Scan Context match).
     *
     * @param from_id     earlier keyframe index
     * @param to_id       current keyframe index
     * @param T_rel       relative pose from GICP alignment
     * @param gicp_cov    6×6 ICP covariance (from fitness score heuristic)
     */
    void addLoopClosure(int from_id, int to_id,
                        const gtsam::Pose3& T_rel,
                        const Eigen::Matrix<double, 6, 6>& gicp_cov) {
        LoopClosureEdge edge{from_id, to_id, T_rel, gicp_cov};
        loop_edges_.push_back(edge);

        auto lc_noise = gtsam::noiseModel::Gaussian::Covariance(gicp_cov);
        graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
            X(from_id), X(to_id), T_rel, lc_noise));

        if (verbose_)
            std::cout << "[PoseGraph] Loop closure: " << from_id
                      << " → " << to_id << "\n";
    }

    /**
     * Optimize with Levenberg-Marquardt.
     *
     * @param max_iter   LM max iterations
     * @param rel_tol    relative error tolerance
     * @return           true if converged
     */
    bool optimize(int max_iter = 100, double rel_tol = 1e-5) {
        gtsam::LevenbergMarquardtParams params;
        params.maxIterations        = max_iter;
        params.relativeErrorTol     = rel_tol;
        params.verbosityLM          = verbose_
            ? gtsam::LevenbergMarquardtParams::SUMMARY
            : gtsam::LevenbergMarquardtParams::SILENT;

        try {
            gtsam::LevenbergMarquardtOptimizer optimizer(
                graph_, initial_values_, params);
            optimized_values_ = optimizer.optimize();
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[PoseGraph] Optimization failed: " << e.what() << "\n";
            return false;
        }
    }

    /**
     * Return per-keyframe correction transforms T_correction[i] = T_opt[i] * T_raw[i]^{-1}.
     *
     * Used by the caller to update the i-Octree map.
     */
    std::vector<gtsam::Pose3> getCorrectionTransforms() const {
        std::vector<gtsam::Pose3> corrections;
        corrections.reserve(keyframes_.size());
        for (const auto& kf : keyframes_) {
            gtsam::Pose3 T_opt = optimized_values_.at<gtsam::Pose3>(X(kf.id));
            corrections.push_back(T_opt * kf.pose.inverse());
        }
        return corrections;
    }

    /**
     * Return optimized trajectory as vector of SE(3) poses.
     */
    std::vector<gtsam::Pose3> getOptimizedTrajectory() const {
        std::vector<gtsam::Pose3> traj;
        traj.reserve(keyframes_.size());
        for (const auto& kf : keyframes_)
            traj.push_back(optimized_values_.at<gtsam::Pose3>(X(kf.id)));
        return traj;
    }

    /**
     * Dump trajectory to TUM format for evo evaluation.
     * Format: timestamp tx ty tz qx qy qz qw
     */
    void saveTUM(const std::string& path) const {
        std::ofstream f(path);
        for (const auto& kf : keyframes_) {
            gtsam::Pose3 T = optimized_values_.at<gtsam::Pose3>(X(kf.id));
            gtsam::Point3 t = T.translation();
            gtsam::Quaternion q = T.rotation().toQuaternion();
            f << std::fixed << std::setprecision(6)
              << kf.timestamp << " "
              << t.x() << " " << t.y() << " " << t.z() << " "
              << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
        }
    }

    int numKeyframes()    const { return static_cast<int>(keyframes_.size()); }
    int numLoopEdges()    const { return static_cast<int>(loop_edges_.size()); }

private:
    bool verbose_;
    int  next_id_;

    std::vector<KeyframePose>    keyframes_;
    std::vector<LoopClosureEdge> loop_edges_;

    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values               initial_values_;
    gtsam::Values               optimized_values_;
};

}  // namespace da_ieskf
