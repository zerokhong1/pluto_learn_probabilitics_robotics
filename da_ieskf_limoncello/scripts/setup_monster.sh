#!/usr/bin/env bash
# setup_monster.sh — install DA-IESKF dependencies on the monster server
# Run once before downloading datasets or building LIMOncello.
set -euo pipefail

echo "=== DA-IESKF setup on monster ==="
echo "Platform: $(uname -srm)"
echo "Date:     $(date -u)"
echo ""

# ── ROS 2 Jazzy ──────────────────────────────────────────────────────────────
if ! command -v ros2 &>/dev/null; then
    echo "[1/6] Installing ROS 2 Jazzy..."
    sudo apt-get update -q
    sudo apt-get install -y software-properties-common curl
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
        http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
        | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
    sudo apt-get update -q
    sudo apt-get install -y ros-jazzy-desktop python3-colcon-common-extensions
    echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
else
    echo "[1/6] ROS 2 Jazzy already installed — skipping."
fi
source /opt/ros/jazzy/setup.bash

# ── GTSAM 4.2 ────────────────────────────────────────────────────────────────
if ! dpkg -l | grep -q libgtsam; then
    echo "[2/6] Installing GTSAM 4.2..."
    sudo apt-get install -y cmake libboost-all-dev libtbb-dev
    cd /tmp
    git clone --depth 1 --branch 4.2 https://github.com/borglab/gtsam.git gtsam_build
    cd gtsam_build
    mkdir -p build && cd build
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGTSAM_BUILD_TESTS=OFF \
        -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
        -DGTSAM_USE_SYSTEM_EIGEN=ON \
        -DGTSAM_BUILD_WITH_MARCH_NATIVE=ON
    make -j$(nproc)
    sudo make install
    sudo ldconfig
    cd /tmp && rm -rf gtsam_build
    echo "[2/6] GTSAM installed."
else
    echo "[2/6] GTSAM already installed — skipping."
fi

# ── PCL + yaml-cpp ───────────────────────────────────────────────────────────
echo "[3/6] Installing PCL, yaml-cpp, Eigen..."
sudo apt-get install -y \
    libpcl-dev \
    libyaml-cpp-dev \
    libeigen3-dev \
    libflann-dev \
    python3-pip \
    python3-venv

# ── evo trajectory evaluation tool ──────────────────────────────────────────
echo "[4/6] Installing evo..."
pip3 install --user --upgrade evo 2>/dev/null || pip3 install evo
echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"
evo_ape --version 2>/dev/null && echo "evo OK" || echo "WARNING: evo not found in PATH"

# ── Python packages for plot_results.py ──────────────────────────────────────
echo "[5/6] Installing Python plotting deps..."
pip3 install --user numpy matplotlib pandas seaborn scipy

# ── LIMOncello workspace ──────────────────────────────────────────────────────
echo "[6/6] Setting up LIMOncello workspace..."
mkdir -p ~/ros2_ws_limo/src
if [ ! -d ~/ros2_ws_limo/src/LIMOncello ]; then
    cd ~/ros2_ws_limo/src
    git clone https://github.com/CPerezRuiz335/LIMOncello.git
    cd LIMOncello && git submodule update --init --recursive
    echo "LIMOncello cloned."
else
    echo "LIMOncello already present — skipping clone."
fi

# Copy DA-IESKF headers
LIMONCELLO_INC=~/ros2_ws_limo/src/LIMOncello/include
DADIR=$(dirname "$(realpath "$0")")/../include
echo "Copying DA-IESKF headers to $LIMONCELLO_INC ..."
cp "$DADIR/DegeneracyDetector.hpp"  "$LIMONCELLO_INC/"
cp "$DADIR/PoseGraph.hpp"           "$LIMONCELLO_INC/"
cp "$DADIR/ScanContextManager.hpp"  "$LIMONCELLO_INC/"

# Copy config
cp "$(dirname "$(realpath "$0")")/../config/da_ieskf_params.yaml" \
    ~/ros2_ws_limo/src/LIMOncello/config/

echo ""
echo "=== Setup complete ==="
echo "Next: edit include/State.hpp per patches/State_hpp_changes.md"
echo "Then: cd ~/ros2_ws_limo && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release"
