#pragma once

#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>

#include <vector>
#include <deque>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <memory>
#include <fstream>
#include <stdexcept>

/**
 * ScanContextManager — loop closure detection using Scan Context descriptors
 * followed by GICP alignment for pose estimation.
 *
 * Reference: Kim & Kim, "Scan Context: Egocentric Spatial Descriptor for
 * Place Recognition within 3D Point Cloud Map," IROS 2018.
 *
 * Pipeline:
 *   1. addScan()    — store descriptor + raw cloud at each keyframe
 *   2. detectLoop() — KNN search in descriptor space, return candidate pairs
 *   3. align()      — GICP on candidate pair, return T_rel + fitness
 *   4. accept()     — fitness threshold gate
 */

namespace da_ieskf {

using PointCloud = pcl::PointCloud<pcl::PointXYZ>;

// ── Scan Context descriptor ─────────────────────────────────────────────────

struct ScanContext {
    // SC matrix: N_ring × N_sector, each cell = max z-height of points
    static constexpr int N_RING   = 20;
    static constexpr int N_SECTOR = 60;
    static constexpr double MAX_RADIUS = 80.0;  // meters

    Eigen::MatrixXf sc;     // N_RING × N_SECTOR
    Eigen::VectorXf ring_key;  // column-wise L2 norm (for fast KNN)

    ScanContext() : sc(N_RING, N_SECTOR), ring_key(N_RING) {
        sc.setZero();
        ring_key.setZero();
    }
};

/**
 * Build a Scan Context descriptor from a 3D point cloud.
 * Points are binned by (range, azimuth); each bin stores max z.
 */
inline ScanContext buildScanContext(const PointCloud& cloud) {
    ScanContext desc;

    for (const auto& pt : cloud) {
        double r     = std::sqrt(pt.x * pt.x + pt.y * pt.y);
        double theta = std::atan2(pt.y, pt.x);  // [-π, π]

        if (r < 0.5 || r > ScanContext::MAX_RADIUS) continue;

        int ring   = std::min(static_cast<int>(r / ScanContext::MAX_RADIUS *
                                                ScanContext::N_RING),
                              ScanContext::N_RING - 1);
        int sector = static_cast<int>(
            (theta + M_PI) / (2.0 * M_PI) * ScanContext::N_SECTOR);
        sector = std::clamp(sector, 0, ScanContext::N_SECTOR - 1);

        desc.sc(ring, sector) = std::max(desc.sc(ring, sector),
                                          static_cast<float>(pt.z));
    }

    // Compute ring key (column-wise L2 norm)
    for (int r = 0; r < ScanContext::N_RING; ++r)
        desc.ring_key(r) = desc.sc.row(r).norm();

    return desc;
}

/**
 * Scan Context distance between two descriptors (minimum over column shifts).
 * Column shift corresponds to rotation; we find the best alignment angle.
 *
 * @return  (distance, best_shift_cols)
 */
inline std::pair<double, int> scanContextDistance(const ScanContext& a,
                                                    const ScanContext& b) {
    double min_dist = std::numeric_limits<double>::max();
    int    best_shift = 0;

    for (int shift = 0; shift < ScanContext::N_SECTOR; ++shift) {
        double dist = 0.0;
        for (int ring = 0; ring < ScanContext::N_RING; ++ring) {
            // Cosine distance per ring
            Eigen::VectorXf va = a.sc.row(ring).transpose();
            Eigen::VectorXf vb_shifted = Eigen::VectorXf(ScanContext::N_SECTOR);
            for (int s = 0; s < ScanContext::N_SECTOR; ++s)
                vb_shifted(s) = b.sc(ring, (s + shift) % ScanContext::N_SECTOR);

            double na = va.norm(), nb = vb_shifted.norm();
            if (na > 1e-6 && nb > 1e-6)
                dist += 1.0 - va.dot(vb_shifted) / (na * nb);
            else
                dist += 1.0;
        }
        dist /= ScanContext::N_RING;
        if (dist < min_dist) {
            min_dist  = dist;
            best_shift = shift;
        }
    }
    return {min_dist, best_shift};
}

// ── ScanContextManager ───────────────────────────────────────────────────────

struct LoopCandidate {
    int   query_id;
    int   match_id;
    double sc_distance;
    double yaw_init;    // initial yaw from SC column shift (radians)
};

struct GICPResult {
    bool   converged;
    double fitness_score;
    Eigen::Matrix4f T_rel;   // 4×4 SE(3) relative transform
    Eigen::Matrix<double, 6, 6> covariance;  // ICP covariance estimate
};

class ScanContextManager {
public:
    /**
     * @param sc_dist_thresh   SC distance below which a candidate is proposed (default 0.15)
     * @param gicp_fitness_max maximum GICP fitness score to accept a loop (default 0.3)
     * @param min_time_gap_kf  minimum keyframe gap before considering loop (default 50)
     * @param n_candidates     top-K candidates to verify with GICP (default 3)
     */
    explicit ScanContextManager(double sc_dist_thresh   = 0.15,
                                 double gicp_fitness_max = 0.3,
                                 int    min_time_gap_kf  = 50,
                                 int    n_candidates     = 3)
        : sc_thresh_(sc_dist_thresh),
          gicp_max_fitness_(gicp_fitness_max),
          min_gap_(min_time_gap_kf),
          n_candidates_(n_candidates) {}

    /**
     * Store descriptor + cloud for new keyframe.
     */
    void addScan(int keyframe_id,
                 double timestamp,
                 const PointCloud::ConstPtr& cloud) {
        ids_.push_back(keyframe_id);
        timestamps_.push_back(timestamp);
        clouds_.push_back(cloud);
        descs_.push_back(buildScanContext(*cloud));
    }

    /**
     * Search for loop closure candidates for the most recent keyframe.
     *
     * @return list of candidates (may be empty)
     */
    std::vector<LoopCandidate> detectLoops() const {
        int n = static_cast<int>(descs_.size());
        if (n < min_gap_ + 1) return {};

        const ScanContext& query = descs_.back();
        int query_id = ids_.back();

        // Score all keyframes except recent ones (< min_gap)
        std::vector<std::pair<double, int>> scored;
        for (int i = 0; i < n - min_gap_; ++i) {
            // Quick ring-key pre-filter (L2 on ring keys)
            double ring_dist = (query.ring_key - descs_[i].ring_key).norm();
            if (ring_dist > 3.0) continue;  // fast reject

            auto [sc_dist, shift] = scanContextDistance(query, descs_[i]);
            if (sc_dist < sc_thresh_)
                scored.emplace_back(sc_dist, i);
        }

        // Sort by SC distance, take top-K
        std::sort(scored.begin(), scored.end());
        scored.resize(std::min(static_cast<int>(scored.size()), n_candidates_));

        std::vector<LoopCandidate> candidates;
        for (auto& [dist, idx] : scored) {
            auto [_, shift] = scanContextDistance(query, descs_[idx]);
            double yaw = -shift * 2.0 * M_PI / ScanContext::N_SECTOR;
            candidates.push_back({query_id, ids_[idx], dist, yaw});
        }
        return candidates;
    }

    /**
     * Verify a loop candidate using GICP alignment.
     *
     * @param candidate   loop candidate from detectLoops()
     * @param yaw_init    initial yaw estimate from SC (radians)
     * @return GICPResult
     */
    GICPResult align(const LoopCandidate& candidate) const {
        // Find cloud indices
        auto it_q = std::find(ids_.begin(), ids_.end(), candidate.query_id);
        auto it_m = std::find(ids_.begin(), ids_.end(), candidate.match_id);
        if (it_q == ids_.end() || it_m == ids_.end())
            return {false, 1e9, Eigen::Matrix4f::Identity(), {}};

        int qi = std::distance(ids_.begin(), it_q);
        int mi = std::distance(ids_.begin(), it_m);

        PointCloud::Ptr cloud_q(new PointCloud(*clouds_[qi]));
        PointCloud::Ptr cloud_m(new PointCloud(*clouds_[mi]));

        // Initialize GICP with yaw from SC
        Eigen::Matrix4f T_init = Eigen::Matrix4f::Identity();
        float c = std::cos(static_cast<float>(candidate.yaw_init));
        float s = std::sin(static_cast<float>(candidate.yaw_init));
        T_init(0, 0) = c; T_init(0, 1) = -s;
        T_init(1, 0) = s; T_init(1, 1) =  c;

        pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
        gicp.setMaxCorrespondenceDistance(1.0);
        gicp.setMaximumIterations(50);
        gicp.setTransformationEpsilon(1e-6);
        gicp.setEuclideanFitnessEpsilon(1e-6);
        gicp.setInputSource(cloud_q);
        gicp.setInputTarget(cloud_m);

        PointCloud aligned;
        gicp.align(aligned, T_init);

        GICPResult res;
        res.converged     = gicp.hasConverged();
        res.fitness_score = gicp.getFitnessScore();
        res.T_rel         = gicp.getFinalTransformation();

        // ICP covariance heuristic: σ² ∝ fitness_score
        double sigma2 = std::max(res.fitness_score, 0.01);
        res.covariance = sigma2 * Eigen::Matrix<double, 6, 6>::Identity();

        return res;
    }

    bool accept(const GICPResult& r) const {
        return r.converged && r.fitness_score < gicp_max_fitness_;
    }

    int numScans() const { return static_cast<int>(ids_.size()); }

private:
    double sc_thresh_;
    double gicp_max_fitness_;
    int    min_gap_;
    int    n_candidates_;

    std::vector<int>                     ids_;
    std::vector<double>                  timestamps_;
    std::vector<PointCloud::ConstPtr>    clouds_;
    std::vector<ScanContext>             descs_;
};

}  // namespace da_ieskf
