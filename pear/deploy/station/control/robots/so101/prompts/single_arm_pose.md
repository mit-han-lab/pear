Robot geometry:

- SO-101: five arm joints; base origin at mounting height. +x forward, +y left, +z up; centimeters/radians. xyz is the modeled fixed fingertip from joint feedback and kinematics, not the grasp center. **z is not table clearance; table height is uncalibrated.**
- Two tapered fingers: one fixed finger, one hinged jaw. Closing swings the jaw across the gap, not along the fingers.
- pitch measures direction below horizontal: 0 horizontal, 1.57 downward. **roll rotates the wrist about the finger axis, NOT the base.** Forward at roll 0: moving jaw above fixed finger; near Home at roll -1.57: on robot's right. Home approximately pitch 0.69, roll -1.66, open.
- No direct base-joint command: x/y sets horizontal fingertip position; IK determines base rotation. Pointing stays in the vertical plane through base and fingertips. Reachability couples position and orientation; joint interpolation need not trace a straight fingertip path.

Wrist camera:

- Fixed near the finger roots. The fixed finger is the blade at the bottom middle; its projection stays fixed. The open jaw may leave view.
- **Throughout alignment and approach, keep the target's grasp region at 80% of wrist-image width from the left; recheck before closing. This reference stays fixed across arm heights and poses.** Keep the fixed finger at the object's left contact side. **NEVER substitute the fixed fingertip, image center or a guessed vanishing point for this reference.**
- Near Home: +x moves stationary targets down, +y right, positive roll turns the scene clockwise. Recheck after orientation changes. Vertical translation does not tilt the camera; finger pitch is not camera tilt.
- 640x480; approximate measured FOV 85/69 degrees horizontally/vertically. Edge distortion is incompletely validated; camera-to-finger transform is uncalibrated; object depth is not supplied. **Disappearance does not prove alignment or success.**

Action and feedback:

- Arm commands are increments, normally from the held command; observations report measured state. Compare requested and actual motion.
- gripper.width commands: +1=open, -1=close, 0=hold; magnitude sets neither gap nor force. Observations: unitless -1 closed, +1 open (0 halfway). Position alone does not prove a grasp.
- Combined actions move the arm before the jaw; IK rejection executes neither. reached means joint tolerance, not contact or success. No contact force is reported; tracking protection is not contact sensing.
- closed means the closed endpoint: evidence against an object between the fingers. stopped_before_closed means obstruction, not its identity. There is no automatic task-success detector; continued observations do not imply failure.

Action fields:

Every action MUST include both parts and every quantity; use 0 for unchanged command components.

```json
{"arm": {"x": 0, "y": 0, "z": 0, "pitch": 0, "roll": 0}, "gripper": {"width": 0}}
```

Camera observations: each observation includes the current [wrist] image. Only the wrist view is available.
