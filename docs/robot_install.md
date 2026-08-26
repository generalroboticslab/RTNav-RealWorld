# Robot Install

Run these steps once on the Stretch robot after a fresh OS setup or when
setting up `home-robot` for the first time.

> `rt_ovn` is not installed on the robot. It runs in the agent container on the
> Thor and is included in `stretch-real` as a Git submodule. Initialize it
> on the Thor with `git submodule update --init --recursive` before
> building the containers.

## Prerequisites

The robot must already be functional:

```
stretch_system_check.py   # must be all green before proceeding
```

If anything is red, follow the
[Hello Robot hardware documentation](https://docs.hello-robot.com/0.2/) first.

---

## 1. System packages

```bash
sudo apt update
sudo apt install python-is-python3 pybind11-dev ros-noetic-hector-slam
```

---

## 2. Clone home-robot on the robot

```bash
cd ~
git clone https://github.com/facebookresearch/home-robot.git
export HOME_ROBOT_ROOT=~/home-robot
```

Add to `~/.bashrc`:

```bash
export HOME_ROBOT_ROOT=~/home-robot
```

---

## 3. Install the core Python package (robot side only needs home_robot_hw)

```bash
cd $HOME_ROBOT_ROOT/src/home_robot
pip install -r requirements.txt
pip install -e .

cd $HOME_ROBOT_ROOT/src/home_robot_hw
pip install -e .
```

---

## 4. Set up catkin workspace

```bash
# Create workspace if it does not exist
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
catkin_init_workspace 2>/dev/null || true

# Clone stretch_ros
cd ~/catkin_ws/src
git clone https://github.com/hello-robot/stretch_ros.git --branch noetic

# Symlink home_robot_hw as a ROS package
ln -s $HOME_ROBOT_ROOT/src/home_robot_hw ~/catkin_ws/src/home_robot_hw

# Build
cd ~/catkin_ws
catkin_make
```

Add to `~/.bashrc`:

```bash
source ~/catkin_ws/devel/setup.bash
```

Then reload:

```bash
source ~/.bashrc
```

---

## 5. Verify ROS can find home_robot_hw

```bash
rospack find home_robot_hw
# should print a path, not an error
```

---

## 6. Home the robot

```bash
stretch_robot_home.py
```

This must be done after every power cycle before running any ROS nodes.
