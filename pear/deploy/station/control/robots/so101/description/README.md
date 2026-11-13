# SO-101 robot description

`so101_new_calib.urdf` is from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) at commit
`a758567c3978dfeefe282ede0500085a48fe8f78`, path `Simulation/SO101/`, licensed under Apache-2.0 (see `LICENSE`).
One change: the `<visual>` and `<collision>` elements of every link, and the comments that name their parts, are
removed. The station uses the kinematic chain and the inertias, and placo would load the mesh files those elements
name.

- sha256 of the file as SO-ARM100 has it: `3a65d2d35e68a8d2f0c2cc176d19b884506543c93ba72980145b80abe276022c`
- Zero convention: "new calibration". Each joint's zero is the middle of its range, which
  matches lerobot's `use_degrees=True` normalization.
- End-effector frame: `gripper_frame_link`. The arm joints are `shoulder_pan`,
  `shoulder_lift`, `elbow_flex`, `wrist_flex` and `wrist_roll`.
