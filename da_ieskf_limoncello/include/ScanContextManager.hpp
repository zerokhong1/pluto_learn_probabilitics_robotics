#pragma once

#include <Eigen/Dense>
#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>
#include <memory>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace da_ieskf {

struct SCConfig {
    int    num_rings           = 20;    // radial bins
    int    num_sectors         = 60;    // angular bins
    double max_range           = 80.0;  // metres
    double sc_dist_threshold   = 0.2;   // cosine dist threshold for loop candidate
    int    exclude_recent      = 50;    // skip N most recent frames (no self-match)
    int    ring_key_candidates = 10;    // top-K ring-key candidates for full SC check
};

struct LoopCandidate {
    bool   valid    = false;
    int    frame_id = -1;
    double distance = 1e9;   // scan context cosine distance (lower = more similar)
};

struct SCFrame {
    int              frame_id;
    Eigen::MatrixXd  descriptor;  // (num_rings × num_sectors) — max-height per bin
    Eigen::VectorXd  ring_key;    // per-row L1 norm for fast pre-filter
    Eigen::Matrix4d  pose;        // SE(3) pose at insertion time
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;  // downsampled local cloud for ICP
};

class ScanContextManager {
public:
    explicit ScanContextManager(const SCConfig& cfg = SCConfig()) : cfg_(cfg) {}

    // ── Descriptor ───────────────────────────────────────────────────────────

    Eigen::MatrixXd computeDescriptor(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) const
    {
        Eigen::MatrixXd desc =
            Eigen::MatrixXd::Zero(cfg_.num_rings, cfg_.num_sectors);

        for (const auto& pt : cloud->points) {
            // Guard against NaN/Inf from deskewing — UB cast causes heap corruption
            if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
                continue;

            double r = std::sqrt(pt.x * pt.x + pt.y * pt.y);
            if (r < 1e-3 || r > cfg_.max_range) continue;

            double angle = std::atan2(pt.y, pt.x) + M_PI;  // [0, 2π)

            int ring = static_cast<int>(r / cfg_.max_range * cfg_.num_rings);
            int sector = static_cast<int>(angle / (2.0 * M_PI) * cfg_.num_sectors);

            // Clamp (paranoid bounds check)
            if (ring < 0 || ring >= cfg_.num_rings) continue;
            if (sector < 0 || sector >= cfg_.num_sectors) continue;

            if (pt.z > desc(ring, sector)) desc(ring, sector) = pt.z;
        }
        return desc;
    }

    // Row-wise L1 norm — rotation-invariant ring summary for fast pre-filter
    Eigen::VectorXd computeRingKey(const Eigen::MatrixXd& desc) const
    {
        return desc.rowwise().lpNorm<1>();
    }

    // ── Database ─────────────────────────────────────────────────────────────

    void addFrame(int frame_id,
                  const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                  const Eigen::Matrix4d& pose)
    {
        auto f        = std::make_shared<SCFrame>();
        f->frame_id   = frame_id;
        f->descriptor = computeDescriptor(cloud);
        f->ring_key   = computeRingKey(f->descriptor);
        f->pose       = pose;
        f->cloud      = cloud;
        frames_.push_back(f);
    }

    // ── Loop detection ────────────────────────────────────────────────────────

    // Detect loop candidate for the most recently added frame.
    // Two-stage: ring-key pre-filter → full SC distance on top-K.
    LoopCandidate detectLoop() const
    {
        LoopCandidate result;
        int n = static_cast<int>(frames_.size());
        if (n <= cfg_.exclude_recent + 1) return result;

        const auto& query  = *frames_.back();
        int search_end     = n - cfg_.exclude_recent - 1;

        // Stage 1: ring-key ranking (O(n), cheap)
        std::vector<std::pair<double, int>> scores;
        scores.reserve(search_end);
        for (int i = 0; i < search_end; ++i)
            scores.emplace_back(
                (query.ring_key - frames_[i]->ring_key).norm(), i);

        int top_k = std::min(cfg_.ring_key_candidates, search_end);
        std::partial_sort(scores.begin(), scores.begin() + top_k, scores.end());

        // Stage 2: full SC comparison with column shift on top-K
        for (int k = 0; k < top_k; ++k) {
            int i      = scores[k].second;
            double dist = computeSCDistance(query.descriptor,
                                            frames_[i]->descriptor);
            if (dist < result.distance) {
                result.distance = dist;
                result.frame_id = frames_[i]->frame_id;
            }
        }

        if (result.distance < cfg_.sc_dist_threshold)
            result.valid = true;
        return result;
    }

    const SCFrame* getFrame(int frame_id) const
    {
        for (const auto& f : frames_)
            if (f->frame_id == frame_id) return f.get();
        return nullptr;
    }

    int size() const { return static_cast<int>(frames_.size()); }

private:
    // Rotation-invariant cosine distance: min over all column shifts
    double computeSCDistance(const Eigen::MatrixXd& a,
                             const Eigen::MatrixXd& b) const
    {
        int cols = cfg_.num_sectors;
        double best = 1e9;

        for (int shift = 0; shift < cols; ++shift) {
            double total = 0.0;
            int valid    = 0;
            for (int c = 0; c < cols; ++c) {
                int bc   = (c + shift) % cols;
                double na = a.col(c).norm();
                double nb = b.col(bc).norm();
                if (na < 1e-6 || nb < 1e-6) continue;
                total += 1.0 - a.col(c).dot(b.col(bc)) / (na * nb);
                valid++;
            }
            if (valid > 0) best = std::min(best, total / valid);
        }
        return best;
    }

    SCConfig cfg_;
    std::deque<std::shared_ptr<SCFrame>> frames_;
};

}  // namespace da_ieskf
