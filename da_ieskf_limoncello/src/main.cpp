#include <mutex>
#include <condition_variable>

#include <Eigen/Dense>

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/bool.hpp>

#include "Core/Octree.hpp"
#include "Core/State.hpp"
#include "Core/Cloud.hpp"
#include "Core/Imu.hpp"

#include "Utils/Config.hpp"
#include "Utils/PCL.hpp"

#include "ROSutils.hpp"

// ── DA-IESKF Phase 5: loop closure (Eigen-only pose graph, no GTSAM) ─────────
#include "ScanContextManager.hpp"
#include "PoseGraph.hpp"
#include <pcl/filters/voxel_grid.h>
#include <memory>


class Manager : public rclcpp::Node {

  State state_;
  States state_buffer_;
  
  Imu prev_imu_;
  double first_imu_stamp_;

  bool imu_calibrated_;

  std::mutex mtx_state_;
  std::mutex mtx_buffer_;

  std::condition_variable cv_prop_stamp_;

  charlie::Octree ioctree_;
  bool stop_ioctree_update_;

  // ── Loop closure (DA-IESKF Phase 5) ────────────────────────────────────────
  da_ieskf::ScanContextManager sc_manager_;
  da_ieskf::PoseGraph          pose_graph_;   // Eigen-only: safe to construct anywhere
  int   lc_frame_counter_ = 0;
  int   last_lc_frame_    = -100;
  static constexpr int kSCInterval = 5;
  std::mutex mtx_lc_;   // serialises LC block (outside mtx_state_)
  // Raw IESKF pose from the PREVIOUS scan — used to compute T_rel for odometry factors.
  // Must be raw (not optimized) so LC corrections don't contaminate future odometry factors.
  Eigen::Matrix4d prev_raw_ieskf_pose_ = Eigen::Matrix4d::Identity();

  // Subscribers
  rclcpp::SubscriptionBase::SharedPtr                    lidar_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr   stop_sub_;

  // Publishers
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_state_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_frame_;

  // TF Broadcaster
  tf2_ros::TransformBroadcaster tf_broadcaster_;

  // Debug
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_raw_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_deskewed_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_downsampled_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_filtered_;


public:
  Manager() : Node("limoncello", 
                   rclcpp::NodeOptions()
                      .allow_undeclared_parameters(true)
                      .automatically_declare_parameters_from_overrides(true)),
              first_imu_stamp_(-1.0), 
              state_buffer_(1000), 
              ioctree_(),
              stop_ioctree_update_(false),
              tf_broadcaster_(*this)  {

    Config& cfg = Config::getInstance();
    fill_config(cfg, this);

    state_.init();

    imu_calibrated_ = not (cfg.sensors.calibration.gravity_align
                           or cfg.sensors.calibration.accel
                           or cfg.sensors.calibration.gyro)
                      or cfg.sensors.calibration.time <= 0.; 

    ioctree_.setBucketSize(cfg.ioctree.bucket_size);
    ioctree_.setDownsample(cfg.ioctree.downsample);
    ioctree_.setMinExtent(cfg.ioctree.min_extent);

    // Set callbacks and publishers
    rclcpp::SubscriptionOptions lidar_opt, imu_opt, stop_opt;
    lidar_opt.callback_group = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    imu_opt.callback_group   = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    stop_opt.callback_group  = create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        cfg.topics.input.imu, 3000, 
        std::bind(&Manager::imu_callback, this, std::placeholders::_1), imu_opt);

    stop_sub_ = this->create_subscription<std_msgs::msg::Bool>(
        cfg.topics.input.stop_ioctree_update, 10,
        std::bind(&Manager::stop_update_callback, this, std::placeholders::_1), stop_opt);

    switch (cfg.sensors.lidar.type) {
      case 0:
      case 1:
      case 2:
      case 3:
        lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            cfg.topics.input.lidar, 5,
            std::bind(&Manager::PointCloud2_callback, this, std::placeholders::_1), lidar_opt);
        break;

      case 4:
        lidar_sub_ = this->create_subscription<livox_interfaces::msg::CustomMsg>(
            cfg.topics.input.lidar, 5,
            std::bind(&Manager::livox_interfaces_callback, this, std::placeholders::_1), lidar_opt);
        break;

      case 5:
        lidar_sub_ = this->create_subscription<livox_ros_driver::msg::CustomMsg>(
            cfg.topics.input.lidar, 5,
            std::bind(&Manager::livox_ros_driver_callback, this, std::placeholders::_1), lidar_opt);
        break;

      case 6:
        lidar_sub_ = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(
            cfg.topics.input.lidar, 5,
            std::bind(&Manager::livox_ros_driver2_callback, this, std::placeholders::_1), lidar_opt);
        break;

      default:
        RCLCPP_ERROR(this->get_logger(),
          "Unknown lidar type %d in config. Cannot create subscriber.", cfg.sensors.lidar.type);
        throw std::runtime_error("Invalid lidar type");
    }

    pub_state_       = this->create_publisher<nav_msgs::msg::Odometry>(cfg.topics.output.state, 10);
    pub_frame_       = this->create_publisher<sensor_msgs::msg::PointCloud2>(cfg.topics.output.frame, 10);

    pub_raw_         = this->create_publisher<sensor_msgs::msg::PointCloud2>("debug/raw",         10);
    pub_deskewed_    = this->create_publisher<sensor_msgs::msg::PointCloud2>("debug/deskewed",    10);
    pub_downsampled_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("debug/downsampled", 10);
    pub_filtered_    = this->create_publisher<sensor_msgs::msg::PointCloud2>("debug/filtered",    10);
  }
  

  void imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr& msg) {

    Config& cfg = Config::getInstance();

    Imu imu = fromROS(msg);

    if (first_imu_stamp_ < 0.)
      first_imu_stamp_ = imu.stamp;
    
    if (not imu_calibrated_) {
      static int N(0);
      static Eigen::Vector3d gyro_avg(0., 0., 0.);
      static Eigen::Vector3d accel_avg(0., 0., 0.);

      if ((imu.stamp - first_imu_stamp_) < cfg.sensors.calibration.time) {
        gyro_avg  += imu.ang_vel;
        accel_avg += imu.lin_accel; 
        N++;

      } else {
        gyro_avg /= N;
        accel_avg /= N;

        if (cfg.sensors.calibration.gravity_align) {
          Eigen::Vector3d g_m = (accel_avg - state_.b_a()).normalized(); 
                          g_m *= cfg.sensors.extrinsics.gravity;
          
          Eigen::Vector3d g_b = state_.quat().conjugate() * state_.g();
          Eigen::Quaterniond dq = Eigen::Quaterniond::FromTwoVectors(g_b, g_m);

          state_.quat((state_.quat() * dq).normalized());
        }
        
        if (cfg.sensors.calibration.gyro)
          state_.b_w(gyro_avg);

        if (cfg.sensors.calibration.accel)
          state_.b_a(accel_avg - state_.R().transpose()*state_.g());

        imu_calibrated_ = true;
      }

    } else {
      double dt = imu.stamp - prev_imu_.stamp;

      if (dt < 0)
        RCLCPP_ERROR(get_logger(), "IMU timestamps not correct");

      dt = (dt < 0 or dt >= imu.stamp) ? 1./cfg.sensors.imu.hz : dt;

      // Correct acceleration
      imu.lin_accel = cfg.sensors.intrinsics.sm * imu.lin_accel;
      prev_imu_ = imu;

      mtx_state_.lock();
        state_.predict(imu, dt);
      mtx_state_.unlock();

      mtx_buffer_.lock();
        state_buffer_.push_front(state_);
      mtx_buffer_.unlock();

      cv_prop_stamp_.notify_one();

      pub_state_->publish(toROS(state_, imu.stamp));
      publishTFs(state_, tf_broadcaster_, imu.stamp);
    }
  }

  void PointCloud2_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) {
    process_cloud([&]() {
      PointCloudT::Ptr raw(new PointCloudT);
      pcl::fromROSMsg(*msg, *raw);
      return raw;
    }, rclcpp::Time(msg->header.stamp).seconds());
  }

  void livox_ros_driver_callback(const livox_ros_driver::msg::CustomMsg::ConstSharedPtr& msg) {
    process_cloud([&]() {
      PointCloudT::Ptr raw(new PointCloudT);
      fromROS(*msg, *raw);
      return raw;
    }, rclcpp::Time(msg->header.stamp).seconds());
  }

  void livox_ros_driver2_callback(const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr& msg) {
    process_cloud([&]() {
      PointCloudT::Ptr raw(new PointCloudT);
      fromROS(*msg, *raw);
      return raw;
    }, rclcpp::Time(msg->header.stamp).seconds());
  }

  void livox_interfaces_callback(const livox_interfaces::msg::CustomMsg::ConstSharedPtr& msg) {
    process_cloud([&]() {
      PointCloudT::Ptr raw(new PointCloudT);
      fromROS(*msg, *raw);
      return raw;
    }, rclcpp::Time(msg->header.stamp).seconds());
  }

  template<typename F>
  void process_cloud(F&& producer, const double& sweep_time) {
    Config& cfg = Config::getInstance();

    if (not imu_calibrated_)
      return;
    
    if (state_buffer_.empty()) {
      RCLCPP_ERROR(get_logger(), "[LIMONCELLO] No IMUs received");
      return;
    }
    
    PointCloudT::Ptr raw = producer();

    if (raw->points.empty()) {
      RCLCPP_ERROR(get_logger(), "[LIMONCELLO] Raw PointCloud is empty!");
      return;
    }

    min_at_front_max_at_back(raw); // oldest point to front and newest to back
    PointTime point_time = point_time_func();
    
    double offset = 0.0;
    if (cfg.sensors.time_offset) { // automatic sync (not precise!)
      offset = state_.stamp - point_time(raw->points.back(), sweep_time) - 1.e-4; 
      if (offset > 0.0) offset = 0.0; // don't jump into future
    }

    // Wait for state buffer
    double start_stamp = point_time(raw->points.front(), sweep_time) + offset;
    double end_stamp   = point_time(raw->points.back(),  sweep_time) + offset;

    if (state_buffer_.front().stamp < end_stamp) {
      std::unique_lock<decltype(mtx_buffer_)> lock(mtx_buffer_);

      RCLCPP_INFO(
        get_logger(),
        "PROPAGATE WAITING...\n"
        "     - buffer time: %.20f\n"
        "     - end scan time: %.20f",
        state_buffer_.front().stamp, end_stamp);

      cv_prop_stamp_.wait(lock, [this, &end_stamp] { 
        return state_buffer_.front().stamp >= end_stamp;
      });
    } 

  mtx_buffer_.lock();
    States interpolated = filter_states(state_buffer_, start_stamp, end_stamp);
  mtx_buffer_.unlock();

    if (start_stamp < interpolated.front().stamp or interpolated.size() == 0) {
      // every point needs to have a state associated not in the past
      RCLCPP_WARN(get_logger(), "Not enough interpolated states for deskewing pointcloud \n");
      return;
    }

  mtx_state_.lock();

    PointCloudT::Ptr deskewed    = deskew(raw, state_, interpolated, offset, sweep_time);
    PointCloudT::Ptr downsampled = voxel_grid(deskewed);
    PointCloudT::Ptr filtered    = filter(downsampled, 
                                          cfg.sensors.extrinsics.imu2baselink * state_.L2I_isometry());

    if (filtered->points.empty()) {
      RCLCPP_ERROR(get_logger(), "Filtered cloud is empty!");
      mtx_state_.unlock();
      return;
    }
    
    state_.update(filtered, ioctree_);
    Eigen::Isometry3f T = (state_.isometry() * state_.L2I_isometry()).cast<float>();

    // Extract pose/covariance for LC block (runs outside mtx_state_)
    Eigen::Matrix4d cur_pose = state_.isometry().matrix().cast<double>();
    Eigen::Matrix<double,6,6> pose_cov = Eigen::Matrix<double,6,6>::Zero();
    {
      auto& P = state_.P;
      // SGal3d tangent order: velocity[0:3], rotation[3:6], translation[6:9]
      pose_cov.block<3,3>(0,0) = P.block<3,3>(3,3).template cast<double>();
      pose_cov.block<3,3>(3,3) = P.block<3,3>(6,6).template cast<double>();
      pose_cov.block<3,3>(0,3) = P.block<3,3>(3,6).template cast<double>();
      pose_cov.block<3,3>(3,0) = P.block<3,3>(6,3).template cast<double>();
      pose_cov += Eigen::Matrix<double,6,6>::Identity() * 1e-6;
    }

  // Release mtx_state_ BEFORE the LC block so the IMU callback is not blocked
  // during GICP / Gauss-Newton (both are O(N²–N³) and must not hold the state lock).
  mtx_buffer_.lock();
    state_buffer_[0] = state_;
  mtx_buffer_.unlock();

  mtx_state_.unlock();

    // ── DA-IESKF Phase 5: pose graph + loop closure (Eigen-only) ────────────
    // mtx_lc_ serialises concurrent scan callbacks; mtx_state_ is NOT held here.
    {
      std::lock_guard<std::mutex> lc_guard(mtx_lc_);

      if (lc_frame_counter_ > 0) {
        // Use raw IESKF relative motion — NOT optimized-prev^-1 * cur_pose.
        // After LC optimization, getOptimizedPose() jumps; cur_pose doesn't.
        // Using optimized prev as anchor would inject the LC correction back as
        // a spurious odometry factor, undoing the correction on the next optimize().
        Eigen::Matrix4d raw_T_rel = prev_raw_ieskf_pose_.inverse() * cur_pose;
        pose_graph_.addOdometryFactor(
            lc_frame_counter_ - 1, lc_frame_counter_,
            raw_T_rel, pose_cov);
      }

      // Scan Context every kSCInterval frames
      if (lc_frame_counter_ % kSCInterval == 0) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr sc_cloud(
            new pcl::PointCloud<pcl::PointXYZI>);
        for (const auto& pt : filtered->points) {
          if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
            continue;
          pcl::PointXYZI p;
          p.x = pt.x; p.y = pt.y; p.z = pt.z; p.intensity = pt.intensity;
          sc_cloud->push_back(p);
        }
        sc_manager_.addFrame(lc_frame_counter_, sc_cloud, cur_pose);

        auto lc = sc_manager_.detectLoop();
        if (lc.valid && (lc_frame_counter_ - last_lc_frame_) > 30) {
          const auto* cand = sc_manager_.getFrame(lc.frame_id);
          if (cand && cand->cloud) {
            // Near-range filter (15m sphere) then 1.5m VoxelGrid.
            // VoxelGrid on a 15m-radius point set gives ~200-500 spatially
            // uniform centroids — no stride cap needed, avoiding the hash-
            // ordering aliasing that caused run10 ICP fitness 2.5-3.9m².
            const float kNearRange2 = 15.0f * 15.0f;
            pcl::PointCloud<pcl::PointXYZI>::Ptr near_src(new pcl::PointCloud<pcl::PointXYZI>);
            for (const auto& p : sc_cloud->points)
              if (p.x*p.x + p.y*p.y + p.z*p.z < kNearRange2)
                near_src->push_back(p);
            pcl::PointCloud<pcl::PointXYZI>::Ptr near_tgt(new pcl::PointCloud<pcl::PointXYZI>);
            for (const auto& p : cand->cloud->points)
              if (p.x*p.x + p.y*p.y + p.z*p.z < kNearRange2)
                near_tgt->push_back(p);

            pcl::VoxelGrid<pcl::PointXYZI> vg;
            vg.setLeafSize(1.5f, 1.5f, 1.5f);

            pcl::PointCloud<pcl::PointXYZI>::Ptr src_down(new pcl::PointCloud<pcl::PointXYZI>);
            vg.setInputCloud(near_src);
            vg.filter(*src_down);

            pcl::PointCloud<pcl::PointXYZI>::Ptr tgt_down(new pcl::PointCloud<pcl::PointXYZI>);
            vg.setInputCloud(near_tgt);
            vg.filter(*tgt_down);

            // Eigen-only SVD point-to-point ICP — zero PCL registration.
            // T accumulates correctly: T_new = dT * T_old each iteration.
            const int   kIcpIter  = 15;
            const float kMaxDist  = 3.0f;   // > 1.5m√3≈2.6m voxel diagonal
            const float kMaxDist2 = kMaxDist * kMaxDist;

            int N = static_cast<int>(src_down->size());
            int M = static_cast<int>(tgt_down->size());
            Eigen::Matrix4f T = (cand->pose.inverse() * cur_pose).cast<float>();

            bool converged = (N >= 3 && M >= 3);
            for (int iter = 0; iter < kIcpIter && converged; ++iter) {
              // Apply current accumulated T to original source
              std::vector<Eigen::Vector3f> src_t(N);
              for (int i = 0; i < N; ++i) {
                auto& p = src_down->points[i];
                Eigen::Vector4f h(p.x, p.y, p.z, 1.f);
                src_t[i] = (T * h).head<3>();
              }
              // Brute-force 1-NN
              std::vector<int>   nn(N, -1);
              std::vector<float> nn_d2(N, kMaxDist2);
              for (int i = 0; i < N; ++i)
                for (int j = 0; j < M; ++j) {
                  auto& q = tgt_down->points[j];
                  float dx = src_t[i].x() - q.x;
                  float dy = src_t[i].y() - q.y;
                  float dz = src_t[i].z() - q.z;
                  float d2 = dx*dx + dy*dy + dz*dz;
                  if (d2 < nn_d2[i]) { nn_d2[i] = d2; nn[i] = j; }
                }
              // Centroids over inlier correspondences
              Eigen::Vector3f sc = Eigen::Vector3f::Zero();
              Eigen::Vector3f tc = Eigen::Vector3f::Zero();
              int cnt = 0;
              for (int i = 0; i < N; ++i) {
                if (nn[i] < 0) continue;
                sc += src_t[i];
                auto& q = tgt_down->points[nn[i]];
                tc += Eigen::Vector3f(q.x, q.y, q.z);
                cnt++;
              }
              if (cnt < 3) { converged = false; break; }
              sc /= cnt; tc /= cnt;
              // Cross-covariance → SVD → incremental rotation R, translation t
              Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
              for (int i = 0; i < N; ++i) {
                if (nn[i] < 0) continue;
                auto& q = tgt_down->points[nn[i]];
                H += (src_t[i] - sc) * (Eigen::Vector3f(q.x,q.y,q.z) - tc).transpose();
              }
              Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
              Eigen::Matrix3f R = svd.matrixV() * svd.matrixU().transpose();
              if (R.determinant() < 0) {
                Eigen::Matrix3f V = svd.matrixV(); V.col(2) *= -1.f;
                R = V * svd.matrixU().transpose();
              }
              // dT maps src_t-frame points to target frame
              Eigen::Matrix4f dT = Eigen::Matrix4f::Identity();
              dT.block<3,3>(0,0) = R;
              dT.block<3,1>(0,3) = tc - R * sc;
              // Accumulate: total transform = dT * previous_T
              T = dT * T;
            }

            if (converged) {
              // Fitness: mean squared NN distance under final accumulated T
              float fitness = 0.f; int cnt = 0;
              for (int i = 0; i < N; ++i) {
                auto& p = src_down->points[i];
                Eigen::Vector4f h(p.x, p.y, p.z, 1.f);
                Eigen::Vector3f tp = (T * h).head<3>();
                float best = kMaxDist2;
                for (int j = 0; j < M; ++j) {
                  auto& q = tgt_down->points[j];
                  float dx = tp.x()-q.x, dy = tp.y()-q.y, dz = tp.z()-q.z;
                  best = std::min(best, dx*dx+dy*dy+dz*dz);
                }
                if (best < kMaxDist2) { fitness += best; cnt++; }
              }
              double score = cnt > 0 ? static_cast<double>(fitness / cnt) : kMaxDist2;

              pose_graph_.addLoopClosureFactor(
                  lc.frame_id, lc_frame_counter_,
                  T.cast<double>(), score);
              last_lc_frame_ = lc_frame_counter_;

              RCLCPP_INFO(get_logger(),
                "LC: frame %d → %d, SC=%.3f, ICP=%.4f %s",
                lc.frame_id, lc_frame_counter_, lc.distance, score,
                score < 0.3 ? "[ACCEPTED]" : "[REJECTED]");
            }
          }
        }
      }

      if (pose_graph_.hasNewLoopClosures()) {
        pose_graph_.optimize();
        Eigen::Matrix4d opt = pose_graph_.getOptimizedPoseEigen(lc_frame_counter_);
        Eigen::Vector3d raw_p = cur_pose.block<3,1>(0,3);
        Eigen::Vector3d opt_p = opt.block<3,1>(0,3);
        RCLCPP_INFO(get_logger(),
          "PG: frame %d raw=(%.3f,%.3f,%.3f) opt=(%.3f,%.3f,%.3f) delta=(%.4f,%.4f,%.4f)",
          lc_frame_counter_,
          raw_p.x(), raw_p.y(), raw_p.z(),
          opt_p.x(), opt_p.y(), opt_p.z(),
          opt_p.x()-raw_p.x(), opt_p.y()-raw_p.y(), opt_p.z()-raw_p.z());
        // Correct state_ directly — future IMU-rate publishes naturally reflect the fix.
        // This avoids the 400Hz data-race copy that caused the run15 regression.
        Eigen::Quaterniond opt_q(opt.block<3,3>(0,0));
        opt_q.normalize();
        mtx_state_.lock();
        state_.X.element<0>() = manif::SGal3d(opt_p, opt_q, state_.v(), state_.t());
        mtx_state_.unlock();
        prev_raw_ieskf_pose_ = opt;  // corrected pose is new raw base for next T_rel
      } else {
        prev_raw_ieskf_pose_ = cur_pose;
      }
      lc_frame_counter_++;
    }
    // ── end loop closure ──────────────────────────────────────────────────────

    PointCloudT::Ptr global(new PointCloudT);
    deskewed->height = 1;                     
    deskewed->width  = static_cast<uint32_t>(deskewed->points.size());
    pcl::transformPointCloud(*deskewed, *global, T);
    
    PointCloudT::Ptr to_save(new PointCloudT);
    filtered->height = 1;                     
    filtered->width  = static_cast<uint32_t>(filtered->points.size());
    pcl::transformPointCloud(*filtered, *to_save, T);

    pub_state_->publish(toROS(state_, sweep_time));
    pub_frame_->publish(toROS(global, sweep_time));

    if (cfg.debug) {
      pub_raw_->publish(toROS(raw, sweep_time));
      pub_deskewed_->publish(toROS(deskewed, sweep_time));
      pub_downsampled_->publish(toROS(downsampled, sweep_time));
      pub_filtered_->publish(toROS(to_save, sweep_time));
    }

    // Update map
    if (not stop_ioctree_update_)
      ioctree_.update(to_save->points);

    if (cfg.verbose)
      PROFC_PRINT()
  }


  void stop_update_callback(const std_msgs::msg::Bool::ConstSharedPtr msg) {
    if (not stop_ioctree_update_ and msg->data) {
      stop_ioctree_update_ = msg->data;
      RCLCPP_INFO(this->get_logger(), "Stopping ioctree updates from now onwards");
    }
  }

};


int main(int argc, char** argv) {

  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
  
  rclcpp::init(argc, argv);

  rclcpp::Node::SharedPtr manager = std::make_shared<Manager>();

  rclcpp::executors::MultiThreadedExecutor executor; // by default using all available cores
  executor.add_node(manager);
  executor.spin();

  rclcpp::shutdown();

  return 0;
}

