import operator
from functools import reduce
from typing import Annotated, Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, create_model, model_validator

from pear.deploy.station.control.joint.layout import JointLayout
from pear.deploy.station.control.joint.operations import SequenceOperation


class StrictModel(BaseModel):
    """The base of every model of an action and of the state a model reads: no key but its fields, no value of
    another type, no number that is not finite."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False, revalidate_instances="always")


class GripperQuantity(StrictModel):
    """The one quantity of a gripper, as a model reads it of the state and sets it in an action."""

    width: float


class JointPositions(StrictModel):
    """A finite chunk of absolute targets of a robot's joint vector, streamed at its own rate and settled once after
    its last row."""

    LAYOUT: ClassVar[JointLayout]  # the joint vector the rows are of: for_layout gives it to the model it makes

    type: Literal["joint_positions"] = "joint_positions"
    positions: list[list[float]]
    rate_hz: float = Field(gt=0)

    @classmethod
    def for_layout(cls, layout: JointLayout) -> "type[Self]":  # quoted: in this class body, type is the field
        """Returns the model of a chunk of targets of the joint vector of ``layout``: every row is one joint
        vector, and a chunk holds at most as many rows as a sequence takes."""
        row = Annotated[list[float], Field(min_length=layout.vector_length, max_length=layout.vector_length)]
        rows = Annotated[list[row], Field(min_length=1, max_length=SequenceOperation.MAX_ROWS)]

        model = create_model(cls.__name__, __base__=cls, __doc__=cls.__doc__, positions=(rows, ...))
        model.LAYOUT = layout

        return model

    @model_validator(mode="after")
    def check_gripper_range(self) -> Self:
        """Raises unless every gripper's target of every row is an opening within [0, 1]."""
        gripper_indices = self.LAYOUT.gripper_indices
        if any(not 0 <= row[gripper_index] <= 1 for row in self.positions for gripper_index in gripper_indices):
            raise ValueError("Gripper targets must be within [0, 1]")

        return self


class ActionAdapter:
    """Reads an action as the model of its type among the action models of a robot."""

    def __init__(self, *action_models: type[StrictModel]) -> None:
        action_union = reduce(operator.or_, action_models)
        self.adapter = TypeAdapter(Annotated[action_union, Field(discriminator="type")])

    def validate(self, action: Any) -> StrictModel:
        """Returns ``action``, a model or the dict of one, as the model its type names, and raises if it is no
        action of the robot."""
        return self.adapter.validate_python(action)
