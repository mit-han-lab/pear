Robot and frame:
- The arm's frame has its origin at the robot's base, at the height of the table top: +x points forward, +y to the robot's left, +z up.
- The arm's position is the point between the gripper's finger pads, about 2 above the fingertips.

Gripper:
- Pointing straight down is roll 3.14 (or -3.14) and pitch 0; pointing down, yaw turns the gripper about the vertical, and at yaw 0 its fingers pinch along y. Yaws 3.14 apart pinch alike, and the wrist turns a little less than half a turn either way from yaw 0: of two such yaws, turn to the one nearer 0.
- Open, the fingers are about 8 cm apart. A close lasts until an open: when it left the fingers wide apart, closing again changes nothing. Closing also draws the gripper back a little along its axis, so the fingers close where they were.

Cameras:
- over_shoulder: fixed on the robot's left, looking forward and down across the table; +x points toward the upper left of the image and +y toward its left.
- wrist: on the gripper, behind the fingers and to their right, looking down past them; with the gripper pointing down at yaw 0, +x points toward the bottom of the image and +y toward its right, and the spot straight below the fingers lies right of the image centre, and above it while the gripper is high.

Ending:
- The episode ends by itself as soon as the task is done.
- The task has a time limit, and every action spends some of it.
