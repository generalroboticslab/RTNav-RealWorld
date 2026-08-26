# RTNav Real-World

Real-world deployment for
**[RTNav: Towards Real-Time Zero-Shot Object Navigation →](https://github.com/generalroboticslab/RTNav)**

Run RTNav on a real Stretch 3 with a Jetson AGX Thor as the compute host. The
robot publishes ROS 1 data, a bridge container exposes it as ROS 2 `/obs/*`
topics, and an agent container performs perception, mapping, planning, and
control.

## Quick start: RTNav

The normal workflow uses three terminals:

1. an SSH session on the Stretch;
2. a Thor shell running the observation bridge;
3. a Thor shell running RTNav with a target passed through `--target`.

### 1. Prerequisites

- Complete the [Stretch-side installation](docs/robot_install.md).
- Install Docker with NVIDIA Container Runtime support on the Thor.
- Make sure your GitHub account can access this repository and its `rt_ovn`
  submodule.
- Use Rerun 0.22.1 when opening the live 3D visualization.

### 2. Clone the release

```bash
mkdir -p ~/Desktop/Nav/final
git clone --branch main --recurse-submodules \
  git@github.com:generalroboticslab/RTNav-RealWorld.git \
  ~/Desktop/Nav/final/RTNav-RealWorld
cd ~/Desktop/Nav/final/RTNav-RealWorld
git submodule update --init --recursive
```

The parent repository pins a tested `RTNav` commit. Its configured update
branch is `thor`.

### 3. Build the Thor images

Run these commands from the `RTNav-RealWorld` root:

```bash
# ROS 1 -> ROS 2 observation bridge
docker build -f docker/Dockerfile.obs_bridge \
  -t stretch-real-obs-bridge:latest .

# RTNav CUDA 13 base for Jetson AGX Thor
# This build compiles and installs the sg_accel and perception_accel C++ modules.
docker build -f rt_ovn/agents/rtnav/docker/Dockerfile \
  -t rt-ovn-agent:latest rt_ovn

# Fail now if either compiled module is unavailable.
docker run --rm rt-ovn-agent:latest \
  python3 -c 'import sg_accel, perception_accel; print("C++ accelerators: OK")'

# Stretch-specific adapters layered onto RTNav
docker build -f docker/Dockerfile.agent \
  -t stretch-real-agent:latest .
```

`stretch-real-agent` extends `rt-ovn-agent`, so build them in that order.
After a change only outside `rt_ovn/`, Docker can reuse the RTNav base image.

### 4. Download model weights

```bash
python3 -m pip install -U huggingface_hub
HF_TOKEN=your_hf_token_here \
  python3 rt_ovn/agents/rtnav/rtnav/download_models.py

mkdir -p data
wget -nc -P data \
  https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
wget -nc -P data \
  https://raw.githubusercontent.com/rai-opensource/vlfm/main/data/pointnav_weights.pth
wget -nc -P data \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -nc -P data \
  https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt

# Verify every Hugging Face model and standalone checkpoint before continuing.
PYTHONPATH=rt_ovn/agents/rtnav python3 -c \
  'from rtnav.download_models import MODELS, _PKG, weights_present; missing = [name for name, (_, path) in MODELS.items() if not weights_present(_PKG / path)]; assert not missing, f"Missing Hugging Face models: {missing}"; print("Hugging Face models: OK")'
test -s data/mobile_sam.pt \
  && test -s data/pointnav_weights.pth \
  && test -s data/groundingdino_swint_ogc.pth \
  && test -s data/yolov7-e6e.pt \
  && echo "Standalone checkpoints: OK"
```

### 5. Start the robot

In the robot SSH session, replace the example address if necessary:

```bash
ssh hello-robot@192.168.8.3

stretch_system_check.py
stretch_robot_home.py
stretch_robot_stow.py

source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash

export ROBOT_IP=192.168.8.3
export ROS_MASTER_URI=http://$ROBOT_IP:11311
export ROS_IP=$ROBOT_IP

roslaunch home_robot_hw startup_stretch_hector_slam.launch teleop_keyboard:=true
```

### 6. Start the observation bridge

In Thor shell A, run:

```bash
cd ~/Desktop/Nav/final/RTNav-RealWorld
test -f src/env/launch.py || { echo "Wrong checkout directory"; exit 1; }
xhost +local:

export ROBOT_IP=192.168.8.3
export THOR_IP=$(hostname -I | awk '{print $1}')

docker run -it --rm --network=host --name stretch-obs-bridge \
  -e ROS_MASTER_URI=http://$ROBOT_IP:11311 \
  -e ROS_IP=$THOR_IP \
  -e ROS_DOMAIN_ID=0 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$PWD:/workspace" \
  stretch-real-obs-bridge:latest
```

After the prompt changes to `/workspace`, run:

```bash
python3 /workspace/src/env/launch.py
```

Leave this container running.

### 7. Start a run with a target

In Thor shell B, run:

```bash
cd ~/Desktop/Nav/final/RTNav-RealWorld
test -f src/agent/rtnav_runner.py || { echo "Wrong checkout directory"; exit 1; }
xhost +local:

docker run --runtime=nvidia -it --rm --network=host \
  --name stretch-real-agent \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e ROS_DOMAIN_ID=0 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e USER=rt_ovn -e LOGNAME=rt_ovn -e HOME=/tmp \
  -e DISPLAY=$DISPLAY \
  -e MOBILE_SAM_CHECKPOINT=/opt/rt_ovn/data/mobile_sam.pt \
  -e POINTNAV_CKPT=/opt/rt_ovn/data/pointnav_weights.pth \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$PWD/rt_ovn:/opt/rt_ovn" \
  -v "$PWD/data:/opt/rt_ovn/data" \
  -v "$PWD:/workspace" \
  stretch-real-agent:latest
```

After the prompt changes to `/workspace`, verify that the complete runner,
including both compiled acceleration modules, imports before enabling motion:

```bash
python3 /workspace/src/agent/rtnav_runner.py --help >/dev/null \
  && echo "RTNav imports: OK"
```

Then start RTNav:

```bash
TARGET="chair"
python3 /workspace/src/agent/rtnav_runner.py \
  --target "$TARGET" \
  --enable-navigation \
  --controller track \
  --map-viz-web \
  --det-viz-web
```

Change `TARGET` for each run. Multiword targets must remain quoted, for
example `TARGET="potted plant"`.

> **Motion warning:** `--enable-navigation` lets the agent command the base
> immediately. Keep the terminal interactive so ENTER remains available as
> the emergency stop. Omit the flag for a perception-and-mapping dry run.

Frontier detection depends on the environment and the size of its openings.
Tune `vlfm_min_frontier_width_m`, `vlfm_area_thresh_m2`, and
`vlfm_frontier_merge_radius_m` in `src/agent/config.py` for the deployment
environment.

Useful run options:

| option | behavior |
| --- | --- |
| `--target "chair"` | object to find; defaults to `chair` |
| `--controller track` | continuously follows the planned path; default |
| `--controller step` | uses blocking 25 cm / 30° primitives for debugging |
| `--no-record` | disables recording under `experiments/` |
| `--no-reasoning` | skips vLLM and goal selection; ignores `--target` |
| `--max-seconds N` | stops after `N` seconds |

## Run output

Recording is enabled by default. Each run creates
`experiments/<timestamp>_<target>/` with synchronized RGB, obstacle-map and
analysis videos, trajectory and planner logs, VLM decisions, a final map,
scene graph, and `result.json`. Generated recordings are ignored by Git.

`result.json` marks a finalized run. To summarize all runs or repair video
timing:

```bash
python3 src/utils/summarize.py
python3 src/utils/correct_analysis_timing.py experiments --overwrite
```

## No-hardware checks

Run these from `/workspace` inside `stretch-real-agent`. They do not connect
to or move the robot:

```bash
python3 src/utils/correct_analysis_timing.py --self-test
python3 tests/08_test_navigation_contracts.py
python3 tests/09_test_goto_primitives.py
PYTHONPATH=rt_ovn/agents/rtnav python3 -m unittest \
  rt_ovn/agents/rtnav/tests/test_canonical_behavior.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=rt_ovn/agents/rtnav \
  python3 -m pytest -q rt_ovn/agents/rtnav/tests/test_model_paths.py
```

Scripts `00`–`04` inspect a live observation stream. Scripts `05`–`07`
are hardware calibration tools: `05` sends its requested primitives
immediately, while `06` and `07` move the base unless passed `--dry-run`.

## Architecture and repository layout

```text
robot (ROS 1)                     Jetson AGX Thor
──────────────────────            ────────────────────────────────────────────
Stretch drivers      ─┐           bridge container           agent container
Hector SLAM           ├── ROS 1 ─▶ src/env/launch.py ─ /obs/* ▶ RTNav
RealSense D435i      ─┘             obs_node + bridge          map → plan → act
```

| path | role |
| --- | --- |
| `src/env/` | builds and bridges the `/obs/*` stream |
| `src/agent/` | real-world RTNav runner and Stretch-specific adapters |
| `src/agent/patches/` | Stretch-only integration points for RT-OVN |
| `rt_ovn/` | pinned upstream RT-OVN submodule |
| `docker/` | bridge and Stretch agent image layers |
| `tests/` | stream, calibration, and no-hardware contract checks |

Both Thor containers must use the same `ROS_DOMAIN_ID` and
`RMW_IMPLEMENTATION`. Host networking lets the bridge reach ROS 1 on the
robot, while CycloneDDS stays on Thor loopback so Wi-Fi route changes do not
interrupt communication between the local containers.
