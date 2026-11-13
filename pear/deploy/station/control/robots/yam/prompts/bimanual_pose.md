Robot and frame:
- left_arm and right_arm share one frame. The origin is midway between the two arms' bases, at the height they are mounted at; +x points forward, +y to the robot's left, +z up.
- left_arm works on the +y side, right_arm on the -y side.

Gripper orientation:
- Both grippers start pointing forward (roll -1.57, pitch 0, yaw 1.57).
- To grasp from above, first turn the gripper to point down: from the start, a roll change of +1.57 points it straight down.
- Pointing down, yaw turns the gripper about the vertical: at yaw 1.57 its fingers sit on its left and right and pinch along y; at yaw 0 they sit in front and behind and pinch along x.

Cameras:
- head: fixed above the robot, looking forward and down. Up in the image is +x (farther from the robot), left is +y.
- left_wrist, right_wrist: one on each gripper, looking along its fingers, which show at the left and right edges of the image. With the gripper pointing straight down at yaw 1.57, up in the image is +x and left is +y.

Ending:
- The episode ends by itself as soon as the task is done: while observations keep coming, the task is not done yet.
- Unless the task is to end up holding something, it is done only once both arms are back near the values they started with.

Answer format:
- Every action lists all four parts with every quantity; write 0 for whatever should not change. For example, this action only lowers the right gripper by 5 and closes it:
```json
{"left_arm": {"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0}, "right_arm": {"x": 0, "y": 0, "z": -5, "roll": 0, "pitch": 0, "yaw": 0}, "left_gripper": {"width": 0}, "right_gripper": {"width": -1}}
```
