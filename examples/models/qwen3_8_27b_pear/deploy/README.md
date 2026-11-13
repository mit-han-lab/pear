# Deploying Qwen3.8-27B-PEAR

`run_server.sh` serves [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), or a model PEAR trained from it, with
vLLM on one GPU, and `run_station.sh` lets it drive a YAM or an SO-101 robot from the operator's page.

## Hardware

| GPU | `GPU` | Model |
| --- | --- | --- |
| GB300 | `gb300` | NVFP4 or bf16 |
| RTX 5090 | `rtx5090` | NVFP4 |

The model is served with a DFlash draft of it.

## Serve

On the machine of the GPU:

```bash
export WORK_DIR=<work dir>
bash scripts/install.sh server

export GPU=gb300 MODEL=<model dir> DRAFT=<draft dir> RUN=<run dir>   # on an RTX 5090: GPU=rtx5090
bash examples/models/qwen3_8_27b_pear/deploy/run_server.sh
```

The server answers at `http://<host>:8000/v1` under the name `qwen3_8_27b`.

## Drive a Robot

On the Linux machine the robot is plugged into:

```bash
export WORK_DIR=<work dir>
bash scripts/install.sh station
export PATH=$WORK_DIR/envs/station/bin:$PATH   # the pear-station command
```

The station's own file, `<station file>` below, names the robot and fills the values the robot's defaults leave `???`
(`pear/deploy/station/control/robots/<robot>/config.yaml`). Start it with the `robot_class` line alone; every command
takes it with `--config`, and the robot does not move while a value is still `???` or a calibration is missing.

### YAM

Each arm has its own USB-to-CAN adapter; the three RealSense cameras sit on USB 3 and need libusb
(`sudo apt-get install libusb-1.0-0`). Fit an emergency stop that cuts the arms' power.

1. Bring each CAN interface up, after every boot:

   ```bash
   sudo ip link set can0 up type can bitrate 1000000
   sudo ip link set can1 up type can bitrate 1000000
   ```

2. Find the devices and fill in the file: `pear-station yam check --config <station file>` lists the CAN interfaces
   with their adapters' serials, `pear-station yam cameras` the cameras' serials.

   ```yaml
   robot_class: pear.deploy.station.control.robots.yam.robot.YAMRobot
   run_dir: <dir>   # the commands' records and snapshots
   robot:
     arms:   # the sides as the operator sees them, facing the worktable
       left:
         channel: can0
         adapter_serial: "<serial>"
       right:
         channel: can1
         adapter_serial: "<serial>"
     cameras:   # serials in quotes: YAML would read some as numbers
       - {serial: "<serial>", role: left}
       - {serial: "<serial>", role: top}
       - {serial: "<serial>", role: right}
   ```

3. Calibrate each gripper with nothing between its fingers (only that gripper moves) and add the
   `gripper_limits: [<closed>, <open>]` it prints under that arm:

   ```bash
   pear-station yam calibrate-gripper left --config <station file> --yes
   pear-station yam calibrate-gripper right --config <station file> --yes
   ```

4. Check without moving: no motor is enabled.

   ```bash
   pear-station yam preflight --config <station file>
   pear-station yam snapshot --config <station file>
   ```

5. First motion, supervised, with the arms supported and the table clear: joint 1 out and back, then a gripper.

   ```bash
   pear-station yam smoke --side left --config <station file> --yes    # then --side right
   pear-station yam gripper-check left close --config <station file> --yes
   ```

6. Identify each arm's gravity and friction, about 9 minutes an arm: only that arm moves, the other holds. Stop the
   page's service, clear the table and stay at the emergency stop. Without `--yes` it prints its sweeps and moves
   nothing. It ends with three lines: put them under that arm in the file, in place of any earlier ones.

   ```bash
   pear-station yam friction-sweep --side left --config <station file>
   pear-station yam friction-sweep --side left --config <station file> --yes    # then --side right
   ```

7. Check each arm's stop accuracy: it visits a few poses without the feedforward, then with it, and passes when the
   tool stops within 3 mm of its target at the median pose and within 6 mm at every pose. On a failure it says what
   to check.

   ```bash
   pear-station yam accuracy-check --side left --config <station file> --yes    # then --side right
   ```

   Do steps 6 and 7 again after changing a gripper, its load or `joint_position_gains`.

### SO-101

Set up the servo IDs as in lerobot's SO-101 guide and add your user to the group of the serial devices (often
`dialout`). Fix the base to the table and keep the power switch within reach.

1. Find the devices and fill in the file: `pear-station so101 check --config <station file>` lists the serial ports and
   the cameras by their stable paths.

   ```yaml
   robot_class: pear.deploy.station.control.robots.so101.robot.SO101Robot
   run_dir: <dir>   # the commands' records and snapshots
   robot:
     hardware: {port: /dev/serial/by-id/<servo board>}
     camera: {path: /dev/v4l/by-id/<camera>}
   ```

2. Calibrate with lerobot's procedure, torque off: the arm in the middle of its range, Enter, each joint through its
   full range, Enter. The calibration goes to `calibration/` beside the station's file.

   ```bash
   pear-station so101 calibrate --config <station file>
   ```

3. Check without moving: torque is left as found.

   ```bash
   pear-station so101 preflight --config <station file>
   pear-station so101 pose --watch --config <station file>   # the measured joints and the pose, until Ctrl-C
   pear-station so101 snapshot --config <station file>
   ```

4. First motion, supervised, with the workspace clear: one joint out and back.

   ```bash
   pear-station so101 smoke --config <station file> --yes
   ```

5. Identify the arm's gravity and friction, a few minutes, with the workspace clear. Without `--yes` it plans its
   steps and moves nothing. It ends with a `feedforward` section: put it under `robot:` in the file, in place of any
   earlier one.

   ```bash
   pear-station so101 friction-sweep --config <station file> --yes
   ```

6. Check the arm's stop accuracy: it visits a few poses without the feedforward, then with it, and passes when the
   fingertip stops within 3 mm of its target at the median pose and within 6 mm at every pose.

   ```bash
   pear-station so101 accuracy-check --config <station file> --yes
   ```

   Do steps 5 and 6 again after changing `position_p_coefficient`: every feedforward value scales with 1 / P.

### Run

With the server up:

```bash
export STATION_CONFIG=<station file> SERVER=<server host>:8000 NAME=qwen3_8_27b RUN=<run dir>
bash examples/models/qwen3_8_27b_pear/deploy/run_station.sh
```

Open the login URL it prints in Chrome on that machine, enable the motors, press Home, then start a task. A run's log,
the tasks' records and their videos go under `RUN`.

## Variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `GPU` | Required | `gb300` or `rtx5090` |
| `MODEL` | Required | The model directory, NVFP4 or bf16 |
| `DRAFT` | Required | The DFlash draft of the model |
| `RUN` | Required | The run directory |
| `PORT` | 8000 | The server's port |
| `NAME` | `qwen3_8_27b` | The model's name in the API |
| `STATION_CONFIG` | Required | The station's file |
| `SERVER` | Required | The server, `host:port` |

Extra arguments are Hydra overrides.
