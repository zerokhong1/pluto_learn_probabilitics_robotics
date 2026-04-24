#!/usr/bin/env python3
"""
Generate minimal synthetic rosbag2 for LIMOncello smoke test.

Publishes:
  /imu/data   (sensor_msgs/Imu)      at 200 Hz
  /points     (sensor_msgs/PointCloud2) at 10 Hz

PointCloud2 layout matches LIMOncello's PointT struct (type=1 VELODYNE):
  x, y, z : float32
  intensity: float32
  time     : float32 (seconds since start of sweep, 0.0 for stationary)

Two scenarios:
  box_room   — 6 walls: full-rank observation (non-degenerate)
  corridor   — 2 walls: degenerate along x-axis

Usage:
  python3 generate_synthetic_bag.py [box|corridor|both]
  Default: both
"""

import sys
import struct
import shutil
import numpy as np
from pathlib import Path
from rosbags.rosbag2 import Writer
from rosbags.typesys import get_typestore, Stores
from rosbags.typesys.stores.ros2_humble import (
    sensor_msgs__msg__Imu         as Imu,
    sensor_msgs__msg__PointCloud2 as PointCloud2,
    sensor_msgs__msg__PointField  as PointField,
    std_msgs__msg__Header         as Header,
    builtin_interfaces__msg__Time as Time,
    geometry_msgs__msg__Quaternion as Quaternion,
    geometry_msgs__msg__Vector3    as Vector3,
)

RNG = np.random.default_rng(42)

# Bag output locations
BAG_DIR = Path.home() / "LIMOncello_ws" / "test"
BOX_BAG     = BAG_DIR / "smoke_box_bag"
CORRIDOR_BAG = BAG_DIR / "smoke_corridor_bag"

DURATION_S  = 15
IMU_HZ      = 200
LIDAR_HZ    = 10
ROOM_HALF   = 5.0   # box room ±5m
ROOM_H      = 3.0
CORRIDOR_W  = 1.5   # corridor half-width (±1.5m in y)
CORRIDOR_L  = 25.0  # corridor half-length in x (no wall)


# ── Point cloud generators ────────────────────────────────────────────────────

def box_room_points(n_per_wall: int = 250) -> np.ndarray:
    """6 walls of a box room. Robot at (0,0,1). All 3 axes observable."""
    h = ROOM_HALF
    pts = []
    for axis, sign in [(0, 1), (0, -1), (1, 1), (1, -1), (2, 0), (2, 1)]:
        # fixed coordinate
        if axis == 0:
            xs = np.full(n_per_wall, sign * h)
            ys = RNG.uniform(-h, h, n_per_wall)
            zs = RNG.uniform(0, ROOM_H, n_per_wall)
        elif axis == 1:
            xs = RNG.uniform(-h, h, n_per_wall)
            ys = np.full(n_per_wall, sign * h)
            zs = RNG.uniform(0, ROOM_H, n_per_wall)
        else:  # floor / ceiling
            xs = RNG.uniform(-h, h, n_per_wall)
            ys = RNG.uniform(-h, h, n_per_wall)
            zs = np.full(n_per_wall, 0.0 if sign == 0 else ROOM_H)
        pts.append(np.stack([xs, ys, zs], axis=1))

    cloud = np.concatenate(pts, axis=0).astype(np.float32)
    cloud += RNG.normal(0, 0.01, cloud.shape).astype(np.float32)
    return cloud


def corridor_points(n_per_wall: int = 400) -> np.ndarray:
    """Two parallel walls at y=±CORRIDOR_W. No walls in x-direction.
    Robot at (0,0,1). x-axis is unobservable → degenerate."""
    xs = RNG.uniform(-CORRIDOR_L, CORRIDOR_L, n_per_wall)
    zs = RNG.uniform(0, ROOM_H, n_per_wall)

    wall_pos = np.stack([xs, np.full(n_per_wall,  CORRIDOR_W), zs], axis=1)
    wall_neg = np.stack([xs, np.full(n_per_wall, -CORRIDOR_W), zs], axis=1)

    # Add floor for z-observability
    xs_f = RNG.uniform(-CORRIDOR_L, CORRIDOR_L, n_per_wall // 2)
    ys_f = RNG.uniform(-CORRIDOR_W, CORRIDOR_W, n_per_wall // 2)
    floor = np.stack([xs_f, ys_f, np.zeros(n_per_wall // 2)], axis=1)

    cloud = np.concatenate([wall_pos, wall_neg, floor], axis=0).astype(np.float32)
    cloud += RNG.normal(0, 0.01, cloud.shape).astype(np.float32)
    return cloud


# ── ROS message builders ──────────────────────────────────────────────────────

def make_time_msg(t_ns: int) -> Time:
    return Time(sec=int(t_ns // 1_000_000_000),
                nanosec=int(t_ns % 1_000_000_000))


def make_header(t_ns: int, frame_id: str) -> Header:
    return Header(stamp=make_time_msg(t_ns), frame_id=frame_id)


def make_imu(t_ns: int) -> Imu:
    """Stationary robot: gravity on z, small noise."""
    return Imu(
        header=make_header(t_ns, 'imu_link'),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        orientation_covariance=np.zeros(9, dtype=np.float64),
        angular_velocity=Vector3(
            x=float(RNG.normal(0, 1e-3)),
            y=float(RNG.normal(0, 1e-3)),
            z=float(RNG.normal(0, 1e-3)),
        ),
        angular_velocity_covariance=np.zeros(9, dtype=np.float64),
        linear_acceleration=Vector3(
            x=float(RNG.normal(0, 1e-2)),
            y=float(RNG.normal(0, 1e-2)),
            z=float(RNG.normal(9.81, 1e-2)),
        ),
        linear_acceleration_covariance=np.zeros(9, dtype=np.float64),
    )


def make_pointcloud2(pts: np.ndarray, t_ns: int) -> PointCloud2:
    """
    Build PointCloud2 matching LIMOncello's PointT (VELODYNE type=1):
      x(f32) y(f32) z(f32) _ intensity(f32) time(f32) _ _
    Offsets come from PCL's PointT registration:
      x=0, y=4, z=8, intensity=16, time=20
    point_step = 32 (PCL EIGEN_ALIGN16 pads to 32 bytes)
    """
    n = len(pts)
    fields = [
        PointField(name='x',         offset=0,  datatype=7, count=1),  # FLOAT32
        PointField(name='y',         offset=4,  datatype=7, count=1),
        PointField(name='z',         offset=8,  datatype=7, count=1),
        PointField(name='intensity', offset=16, datatype=7, count=1),
        PointField(name='time',      offset=20, datatype=7, count=1),  # VELODYNE per-point time
    ]
    point_step = 32  # PCL EIGEN_ALIGN16: 16 (xyz+pad) + 4 (intensity) + 8 (union) + 4 (pad) = 32

    data = np.zeros(n * point_step, dtype=np.uint8)
    for i, (x, y, z) in enumerate(pts):
        base = i * point_step
        struct.pack_into('fff', data, base,      x, y, z)   # offset 0,4,8
        struct.pack_into('f',   data, base + 16, 1.0)       # intensity
        struct.pack_into('f',   data, base + 20, 0.0)       # time=0 (stationary)

    return PointCloud2(
        header=make_header(t_ns, 'lidar_link'),
        height=1,
        width=n,
        fields=fields,
        is_bigendian=False,
        point_step=point_step,
        row_step=point_step * n,
        data=data,
        is_dense=True,
    )


# ── Bag writer ────────────────────────────────────────────────────────────────

def write_bag(bag_path: Path, cloud_fn, label: str):
    if bag_path.exists():
        shutil.rmtree(bag_path)

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    imu_type   = 'sensor_msgs/msg/Imu'
    lidar_type = 'sensor_msgs/msg/PointCloud2'

    imu_dt_ns   = 1_000_000_000 // IMU_HZ
    lidar_dt_ns = 1_000_000_000 // LIDAR_HZ
    total_ns    = DURATION_S * 1_000_000_000

    n_imu   = DURATION_S * IMU_HZ
    n_lidar = DURATION_S * LIDAR_HZ

    print(f"\nGenerating {label}  →  {bag_path}")
    print(f"  IMU:   {n_imu} msgs @ {IMU_HZ} Hz")
    print(f"  LiDAR: {n_lidar} msgs @ {LIDAR_HZ} Hz  ({DURATION_S}s)")

    with Writer(bag_path, version=8) as writer:
        imu_conn   = writer.add_connection('/imu/data', imu_type,   typestore=typestore)
        lidar_conn = writer.add_connection('/points',   lidar_type, typestore=typestore)

        next_lidar_ns = 0
        lidar_count   = 0

        for i in range(n_imu):
            t_ns = i * imu_dt_ns

            # IMU message
            imu_msg = make_imu(t_ns)
            writer.write(imu_conn, t_ns,
                         typestore.serialize_cdr(imu_msg, imu_type))

            # LiDAR at lower rate
            if t_ns >= next_lidar_ns and lidar_count < n_lidar:
                pts = cloud_fn()
                pc2 = make_pointcloud2(pts, t_ns)
                writer.write(lidar_conn, t_ns,
                             typestore.serialize_cdr(pc2, lidar_type))
                lidar_count   += 1
                next_lidar_ns += lidar_dt_ns

    print(f"  Done. {lidar_count} LiDAR scans, {n_imu} IMU messages.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    if mode in ("box", "both"):
        write_bag(BOX_BAG,      box_room_points,  "Box Room (non-degenerate)")
    if mode in ("corridor", "both"):
        write_bag(CORRIDOR_BAG, corridor_points,  "Corridor (degenerate in x)")

    print("\nVerify with:")
    if mode in ("box", "both"):
        print(f"  ros2 bag info {BOX_BAG}")
    if mode in ("corridor", "both"):
        print(f"  ros2 bag info {CORRIDOR_BAG}")


if __name__ == "__main__":
    main()
