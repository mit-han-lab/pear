from dataclasses import dataclass


@dataclass(frozen=True)
class ControlTexts:
    """A robot's wording of the control errors a model may read, as ``str.format`` templates."""

    outside_bounds: str  # {arm}, {details}: a target or a reading outside the position bounds
    # {axis}, {value}, {unit}, {lower}, {upper}, {allowance}: one coordinate of outside_bounds
    outside_bounds_coordinate: str
    command_outside_bounds: str  # {arm}, {details}: a streamed command leaving its bounds
    # the fields of outside_bounds_coordinate: one coordinate of command_outside_bounds
    command_outside_bounds_coordinate: str
    bound_axes: tuple[str, ...]  # one arm's coordinate names in the two coordinate templates
    stale_feedback: str  # {arm}, {age}: feedback too old or not finite
    tracking_error_text: str  # {details}: measured joints too far from their targets
    unsettled_start: str  # enabling found the arms still moving
    failed_hold: str  # recovery refused after the fault hold could not be sent; the reason follows
