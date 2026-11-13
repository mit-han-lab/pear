from pear.deploy.station.control.joint.control_loop import Braking, JointControlLoop
from pear.deploy.station.control.joint.recovery import ReleaseRecovery
from pear.deploy.station.control.robots.yam.joint.grippers import YAMGripperOperation


class YAMControlLoop(JointControlLoop):
    """The control loop of both YAM arms: grippers that confirm a contact, a slow release when it recovers, and a STOP
    that brakes with the caps of the motion it stops."""

    GRIPPER_OPERATION_CLASS = YAMGripperOperation
    RECOVERY_POLICY_CLASS = ReleaseRecovery
    BRAKING = Braking.MOVE_CAPS
    CHECKS_VELOCITY_RMS = True
