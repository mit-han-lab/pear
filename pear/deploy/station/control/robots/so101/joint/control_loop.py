from pear.deploy.station.control.joint.control_loop import Braking, JointControlLoop
from pear.deploy.station.control.joint.recovery import MeasuredHoldRecovery
from pear.deploy.station.control.robots.so101.joint.grippers import SO101GripperOperation


class SO101ControlLoop(JointControlLoop):
    """The SO-101's control loop: servos with feedforward goal offsets, a jaw judged where it stops, and the measured
    pose held after a fault."""

    GRIPPER_OPERATION_CLASS = SO101GripperOperation  # the jaw, which no force sensing tells on an object
    RECOVERY_POLICY_CLASS = MeasuredHoldRecovery  # the stiff position servos keep pushing toward an old target
    BRAKING = Braking.STATION_LIMITS  # a STOP brakes with the arm's STOP caps
    CHECKS_VELOCITY_RMS = False  # a recovery's window bounds the mean velocity alone
