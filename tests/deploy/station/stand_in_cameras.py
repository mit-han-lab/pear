import time

import numpy as np
from PIL import Image, ImageDraw

from pear.deploy.station.control.cameras import CameraSource
from pear.deploy.station.control.schemas import Camera, CameraFrame


class StandInCameras(CameraSource):
    """A robot's cameras that draw, at every read, the scene each camera would see: a table with a ball and a block,
    and an arm whose elbow sways."""

    # the cameras of a robot with two arms, in the order they show: one at each wrist and one overhead
    ROW = (
        Camera("left", "left_wrist", "Left Wrist"),
        Camera("top", "head", "Overhead"),
        Camera("right", "right_wrist", "Right Wrist"),
    )
    SINGLE = (Camera("wrist", "wrist", "Wrist Camera"),)  # the only camera of a robot with one arm, at its wrist
    FRAME_SIZE = (640, 480)  # width and height of every frame
    FRAME_RATE = 30  # frames per second the frame numbers count, as a camera's would
    BACKGROUND_COLOR = "#e8e6df"  # of the scene
    GRID_COLOR = "#d8d6ce"  # of the grid lines behind the table
    GRID_SPACING = 40  # between two grid lines
    TABLE_BOX = (100, 150, 540, 400)  # left, top, right and bottom of the table
    TABLE_RADIUS = 18  # of the table's corners
    TABLE_COLOR = "#c7c4b9"  # of the table
    BALL_BOX = (270, 225, 355, 310)  # left, top, right and bottom of the ball
    BALL_COLOR = "#f59b52"  # of the ball
    BLOCK_BOX = (395, 200, 475, 280)  # left, top, right and bottom of the block
    BLOCK_RADIUS = 10  # of the block's corners
    BLOCK_COLOR = "#5e8691"  # of the block
    ARM_LEFT_X = 70  # where the arm enters, seen from the left
    ARM_RIGHT_X = 570  # where the arm enters, seen from the right
    ARM_BASE_Y = 450  # where the arm enters, from the bottom
    ARM_ELBOW_Y = 320  # of the elbow
    ARM_TIP = (300, 180)  # where the arm ends, over the table
    ARM_WIDTH = 24  # of the arm's line
    ARM_COLOR = "#60777b"  # of the arm
    ELBOW_X = 200  # centre of the elbow's sway
    ELBOW_SWAY = 25  # amplitude of the elbow's sway
    ELBOW_SWAY_RATE = 0.5  # rad/s of the elbow's sway
    GRIPPER_RADIUS = 20  # of the gripper drawn around the elbow
    GRIPPER_COLOR = "#374f54"  # of the gripper
    BANNER_HEIGHT = 38  # of the label bar at the top
    BANNER_COLOR = "#20363e"  # of the label bar, and of the footer's text
    BANNER_TEXT_COLOR = "white"  # of the label, which says that the picture is drawn
    TEXT_X = 16  # left edge of the label and of the footer
    TITLE_Y = 12  # of the label
    FOOTER_Y = 456  # of the footer, which numbers the frame

    def __init__(self, cameras: tuple[Camera, ...]) -> None:
        self.roles = tuple(camera.role for camera in cameras)  # in the order the cameras show
        self.start_time = time.monotonic()  # from when the frames are numbered, on the monotonic clock
        self.is_closed = False

    def read_preview(self) -> tuple[dict[str, CameraFrame], dict[str, str]]:
        """Returns a fresh frame of every camera, and no camera without one."""
        if self.is_closed:
            raise RuntimeError("Stand-in cameras stopped")

        now = time.monotonic()
        frame_number = int((now - self.start_time) * self.FRAME_RATE)
        frames = {}

        for index, role in enumerate(self.roles):
            picture = Image.new("RGB", self.FRAME_SIZE, self.BACKGROUND_COLOR)
            canvas = ImageDraw.Draw(picture)
            self._draw_scene(canvas, index, now)
            canvas.rectangle((0, 0, self.FRAME_SIZE[0], self.BANNER_HEIGHT), fill=self.BANNER_COLOR)
            title = f"SIMULATION / GENERATED / {role.upper()}"
            canvas.text((self.TEXT_X, self.TITLE_Y), title, fill=self.BANNER_TEXT_COLOR)
            footer = f"Preview frame {frame_number}  |  Not a physical simulation"
            canvas.text((self.TEXT_X, self.FOOTER_Y), footer, fill=self.BANNER_COLOR)
            frames[role] = {
                "rgb": np.asarray(picture).copy(),
                "received_monotonic_s": now,
                "frame_number": frame_number,
            }

        return frames, {}

    def close(self) -> None:
        """Stops delivering frames."""
        self.is_closed = True

    def _draw_scene(self, canvas: ImageDraw.ImageDraw, index: int, now: float) -> None:
        """Draws the scene as the camera at ``index`` of the order sees it at ``now``."""
        width, height = self.FRAME_SIZE
        for line_x in range(0, width, self.GRID_SPACING):
            canvas.line((line_x, 0, line_x, height), fill=self.GRID_COLOR)
        for line_y in range(0, height, self.GRID_SPACING):
            canvas.line((0, line_y, width, line_y), fill=self.GRID_COLOR)

        canvas.rounded_rectangle(self.TABLE_BOX, radius=self.TABLE_RADIUS, fill=self.TABLE_COLOR)
        canvas.ellipse(self.BALL_BOX, fill=self.BALL_COLOR)
        canvas.rounded_rectangle(self.BLOCK_BOX, radius=self.BLOCK_RADIUS, fill=self.BLOCK_COLOR)

        # each camera sees the sway at a phase of its own; one in the right half of the order sees the arm enter
        # from the right
        elbow_x = self.ELBOW_X + int(self.ELBOW_SWAY * np.sin((now - self.start_time) * self.ELBOW_SWAY_RATE + index))
        entry_x = self.ARM_RIGHT_X if index >= len(self.roles) / 2 else self.ARM_LEFT_X
        arm_points = (entry_x, self.ARM_BASE_Y, elbow_x, self.ARM_ELBOW_Y, *self.ARM_TIP)
        canvas.line(arm_points, fill=self.ARM_COLOR, width=self.ARM_WIDTH)
        radius = self.GRIPPER_RADIUS
        gripper_box = (elbow_x - radius, self.ARM_ELBOW_Y - radius, elbow_x + radius, self.ARM_ELBOW_Y + radius)
        canvas.ellipse(gripper_box, fill=self.GRIPPER_COLOR)
