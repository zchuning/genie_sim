#!/bin/bash

set -e

# env
export ROS_DISTRO=jazzy

# bashrc
echo "alias omni_python='/isaac-sim/python.sh'" >>~/.bashrc
echo "alias run_server='omni_python source/geniesim/app/raise_standalone_sim.py'" >>~/.bashrc
echo "alias run_client='omni_python source/geniesim/benchmark/task_benchmark.py --task_name'" >>~/.bashrc
echo "alias run_teleop='omni_python source/geniesim/teleop/teleop.py --task_name'" >>~/.bashrc
echo "alias run_replay='omni_python source/geniesim/teleop/replay_state.py'" >>~/.bashrc

echo "source /isaac-sim/${ROS_DISTRO}_ws/install/setup.bash" >>~/.bashrc

echo "export SIM_ASSETS=/root/assets" >>~/.bashrc
echo "export SIM_REPO_ROOT=/root/workspace/main" >>~/.bashrc

echo "export ROS_DISTRO=jazzy" >>~/.bashrc
echo "export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST" >>~/.bashrc

# you can add more customized cmds here

# setup ros2 environment
source "/isaac-sim/${ROS_DISTRO}_ws/install/setup.bash" --
# exec "$@"

#
/bin/bash
