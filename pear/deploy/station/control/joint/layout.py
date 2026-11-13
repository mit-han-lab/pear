from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from pear.deploy.station.control.joint.texts import ControlTexts


@dataclass(frozen=True)
class JointLayout:
    """A robot's streamed joint vector: its arms in order, each with the same coordinates, the gripper last."""

    ARM_LABEL = "side"  # the key of the arm in the labels of a coordinate, as metrics and records spell it
    COORDINATE_LABEL = "joint"  # the key of the coordinate's name in those labels
    JOINT_UNIT = "rad"  # the unit of an arm joint
    GRIPPER_UNIT = "opening"  # the unit of a gripper: 0 closed, 1 open

    arms: tuple[str, ...]
    coordinates: tuple[str, ...]  # one arm's coordinate names in metrics and messages
    texts: ControlTexts

    @property
    def coordinate_count(self) -> int:
        """How many coordinates one arm has, its gripper among them."""
        return len(self.coordinates)

    @property
    def joint_count(self) -> int:
        """How many joints one arm has before its gripper."""
        return self.coordinate_count - 1

    @property
    def vector_length(self) -> int:
        """The length of the joint vector."""
        return len(self.arms) * self.coordinate_count

    @property
    def gripper_indices(self) -> tuple[int, ...]:
        """Where each arm's gripper sits in the joint vector."""
        return tuple((arm_index + 1) * self.coordinate_count - 1 for arm_index in range(len(self.arms)))

    def split(self, vector: Sequence[float]) -> dict[str, tuple[float, ...]]:
        """Returns the joint vector as one tuple for each arm."""
        coordinate_count = self.coordinate_count

        return {
            arm: tuple(vector[arm_index * coordinate_count : (arm_index + 1) * coordinate_count])
            for arm_index, arm in enumerate(self.arms)
        }

    def join(self, arm_values: Mapping[str, Sequence[float]]) -> tuple[float, ...]:
        """Returns one sequence of values for each arm as the joint vector."""
        return tuple(value for arm in self.arms for value in arm_values[arm])

    def arm_axis_labels(self, arm: str, coordinate_index: int) -> dict[str, str]:
        """Returns how metrics name one coordinate of ``arm``: the arm when there are several, and the coordinate."""
        joint_label = {self.COORDINATE_LABEL: self.coordinates[coordinate_index]}

        return {self.ARM_LABEL: arm, **joint_label} if len(self.arms) > 1 else joint_label

    def axis_unit(self, index: int) -> str:
        """Returns the unit of the coordinate at ``index`` of the joint vector or of one arm's coordinates."""
        return self.GRIPPER_UNIT if index % self.coordinate_count == self.joint_count else self.JOINT_UNIT

    def vector(self, values: Sequence[float]) -> np.ndarray:
        """Returns ``values`` as a finite joint vector, and raises if they are not one."""
        return self._finite_array(values, self.vector_length)

    def arm_vector(self, values: Sequence[float]) -> np.ndarray:
        """Returns ``values`` as one arm's finite coordinates, and raises if they are not."""
        return self._finite_array(values, self.coordinate_count)

    def _finite_array(self, values: Sequence[float], length: int) -> np.ndarray:
        """Returns ``values`` as an array of ``length`` finite numbers, and raises if they are not one."""
        array = np.asarray(values, dtype=float)
        if array.shape != (length,) or not np.isfinite(array).all():
            raise ValueError(f"Expected {self.joint_count} finite joint radians and 1 finite gripper opening")

        return array

    @classmethod
    def axis_name(cls, labels: Mapping[str, Any]) -> str:
        """Returns the name a message gives the coordinate that ``labels`` label: its arm, if labelled, and itself."""
        return " ".join(labels[key] for key in (cls.ARM_LABEL, cls.COORDINATE_LABEL) if key in labels)

    @staticmethod
    def joint_values(arm_values: Sequence[float]) -> Sequence[float]:
        """Returns the joints' values of one arm's coordinates, without its gripper's."""
        return arm_values[:-1]

    @staticmethod
    def gripper_value(arm_values: Sequence[float]) -> float:
        """Returns the gripper's value of one arm's coordinates."""
        return arm_values[-1]

    @classmethod
    def with_gripper(cls, arm_values: Sequence[float], gripper_value: float) -> tuple[float, ...]:
        """Returns one arm's coordinates with ``gripper_value`` in the gripper's place."""
        return (*cls.joint_values(arm_values), gripper_value)
