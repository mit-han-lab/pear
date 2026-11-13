import json
import math
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from pear.deploy.station.control.schemas import Camera
from pear.deploy.station.records.recording.captions import CaptionPhase, Captions
from pear.deploy.station.records.recording.ffmpeg import FRAME_RATE, SOURCE_VIDEO_SIZE, VIDEO_SIZE

UNKNOWN_MODEL_LABEL = "Unknown"  # what a recording names the model of a task whose record names none


@dataclass(frozen=True)
class CaptionedLayout:
    """Where a segment's full video puts the cameras and the text."""

    MARGIN = 32  # left and right of the text, in pixels
    TILE_TOP = 152  # y of the camera tiles' top edge
    ROW_TILE_SIZE = SOURCE_VIDEO_SIZE  # of each tile when the cameras stand side by side: a camera's video as it is
    SINGLE_TILE_SIZE = (960, 720)  # of the only camera's tile
    SINGLE_COLUMN_LEFT = 992  # left edge of the text column beside the only camera's tile
    # where the texts stand beside the only camera's tile: turn, phase and task time; badge; usage lines; the
    # action's heading and lines
    SINGLE_CAMERA_POSITIONS = {
        "status": (SINGLE_COLUMN_LEFT, 170),
        "badge": (SINGLE_COLUMN_LEFT, 214),
        "inference": (SINGLE_COLUMN_LEFT, 262),
        "context": (SINGLE_COLUMN_LEFT, 292),
        "context_anchor": "la",
        "heading": (SINGLE_COLUMN_LEFT, 340),
        "code": (SINGLE_COLUMN_LEFT, 374),
        "lines_per_page": 20,
        "is_one_quantity_per_line": True,
        "unavailable_label_offset": (360, 350),
    }
    # where the same texts stand below a row of tiles; the context line ends at the right margin
    ROW_POSITIONS = {
        "status": (MARGIN, 657),
        "badge": (1380, 665),
        "inference": (MARGIN, 710),
        "context": (VIDEO_SIZE[0] - MARGIN, 710),
        "context_anchor": "ra",
        "heading": (MARGIN, 748),
        "code": (MARGIN, 782),
        "lines_per_page": 9,
        "is_one_quantity_per_line": False,
        "unavailable_label_offset": (180, 208),
    }

    tiles: dict[str, list[int]]  # camera role -> [x, y, width, height]
    status: tuple[int, int]  # turn, phase and task time
    badge: tuple[int, int]
    inference: tuple[int, int]
    context: tuple[int, int]
    context_anchor: str
    heading: tuple[int, int]  # MODEL ACTION (JSON)
    code: tuple[int, int]  # the first action line
    code_width: int
    lines_per_page: int  # action lines on one page
    is_one_quantity_per_line: bool  # one quantity on each line instead of one part
    unavailable_label_offset: tuple[int, int]  # of CAMERA UNAVAILABLE from its tile's corner

    @classmethod
    def for_cameras(cls, cameras: tuple[Camera, ...]) -> "CaptionedLayout":
        """Returns the layout of the only camera beside the text, or of a row of cameras above it."""
        if len(cameras) == 1:
            tiles = {cameras[0].role: [0, cls.TILE_TOP, *cls.SINGLE_TILE_SIZE]}
            code_width = VIDEO_SIZE[0] - cls.SINGLE_COLUMN_LEFT - cls.MARGIN

            return cls(tiles=tiles, code_width=code_width, **cls.SINGLE_CAMERA_POSITIONS)

        width, height = cls.ROW_TILE_SIZE
        tiles = {camera.role: [column * width, cls.TILE_TOP, width, height] for column, camera in enumerate(cameras)}

        return cls(tiles=tiles, code_width=VIDEO_SIZE[0] - 2 * cls.MARGIN, **cls.ROW_POSITIONS)


class CaptionedFrames:
    """Draws the frames of a segment's full video, each from one sampled packet."""

    BACKGROUND_COLOUR = "#101923"  # of the whole frame
    ACCENT_COLOUR = "#82b5c5"  # of headings and usage lines
    TILE_HEADER_COLOUR = "#253645"  # of the bar above each camera tile
    EMPTY_TILE_COLOUR = "#26313d"  # of a tile whose camera has no picture
    CODE_COLOUR = "#e2edf5"  # of the action text
    BADGE_COLOUR = "#ecb26d"  # of the badge
    TEXT_COLOUR = "white"  # of the instruction, the status line and the labels
    FONT_FILE = "DejaVuSans.ttf"  # the font of every text but the action
    MONOSPACED_FONT_FILE = "DejaVuSansMono.ttf"  # the font of the action text
    LABEL_SIZE = 19  # type size of headings and labels
    BODY_SIZE = 28  # type size of the instruction
    CODE_SIZE = 25  # type size of the action text
    TITLE_SIZE = 26  # type size of the status line
    TEXT_WIDTH = VIDEO_SIZE[0] - 2 * CaptionedLayout.MARGIN  # the width the instruction wraps to
    PAGE_DURATION_S = 4  # each page of instruction or action text stays this long
    INSTRUCTION_LINES_PER_PAGE = 2  # instruction lines on one page
    INSTRUCTION_Y = 42  # y of the first instruction line
    INSTRUCTION_PITCH = 34  # between two instruction lines
    HEADING_Y = 14  # y of the TASK INSTRUCTION heading
    TILE_HEADER_Y = 120  # y of the bar above the camera tiles
    TILE_HEADER_HEIGHT = 32  # height of that bar
    TILE_LABEL_OFFSET = (18, 124)  # of a camera's label: x from its tile's left edge, y in the frame
    CODE_PITCH = 30  # between two action lines
    JSON_INDENT = 2  # spaces of one level of an action shown one quantity on each line
    ELLIPSIS = "..."  # ends a model caption cut short
    BADGE = "LIVE ROBOT"  # beside the status line

    def __init__(self, task: dict[str, Any], cameras: tuple[Camera, ...]) -> None:
        self.layout = CaptionedLayout.for_cameras(cameras)
        self.camera_labels = {camera.role: camera.name.replace("_", " ").upper() for camera in cameras}
        self.label_font = self._font(self.LABEL_SIZE)
        self.body_font = self._font(self.BODY_SIZE)
        self.code_font = self._font(self.CODE_SIZE, is_monospaced=True)
        self.title_font = self._font(self.TITLE_SIZE)

        self.instruction_lines = self._wrap(task["instruction"], self.body_font, self.TEXT_WIDTH)
        model_label = task.get("model_label") or task.get("model") or UNKNOWN_MODEL_LABEL
        self.model_caption = "MODEL: " + " ".join(str(model_label).split())
        if task.get("action_format") and task["action_format"] not in self.model_caption:
            self.model_caption += f"  |  {task['action_format']}"

        # the panel drawn last, by what it shows: only the cameras and the status line change from frame to frame
        self.cached_key: tuple[Any, ...] | None = None
        self.cached_panel: Image.Image | None = None

    def render(self, packet: dict[str, Any], frame_index: int) -> Image.Image:
        """Returns the frame ``frame_index`` of the video, which shows the cameras and the state of ``packet``."""
        # the state was fixed when the packet was sampled: nothing of the running service is read here
        state = packet["state"]
        elapsed_s = frame_index / FRAME_RATE

        page = int(elapsed_s / self.PAGE_DURATION_S) % max(1, self._instruction_page_count())
        action_text = self._action_text(state.get("action"))
        action_lines = self._wrap(action_text, self.code_font, self.layout.code_width)
        action_page_count = max(1, math.ceil(len(action_lines) / self.layout.lines_per_page))
        action_page = int(elapsed_s / self.PAGE_DURATION_S) % action_page_count
        context_caption = Captions.context(state)
        inference_caption = Captions.inference(state)

        key = (page, action_page, action_text, context_caption, inference_caption)
        if key != self.cached_key:
            panel = self._panel(page, action_lines, action_page, context_caption, inference_caption)
            self.cached_key, self.cached_panel = key, panel

        frame_image = self.cached_panel.copy()
        self._paste_cameras(frame_image, packet["frames"])
        self._draw_status(ImageDraw.Draw(frame_image), state, elapsed_s)

        return frame_image

    def _instruction_page_count(self) -> int:
        """Returns the pages the instruction takes."""
        return math.ceil(len(self.instruction_lines) / self.INSTRUCTION_LINES_PER_PAGE)

    def _action_text(self, action: dict[str, Any] | None) -> str:
        """Returns the action as JSON text: one quantity on each line, or one part where the layout has few lines."""
        if action is None:
            return "Waiting for model output"
        if self.layout.is_one_quantity_per_line:
            return json.dumps(action, indent=self.JSON_INDENT, ensure_ascii=True)

        # each part on one line where it fits, with every field and value
        fields = ",\n".join(f'  "{key}": {json.dumps(value, ensure_ascii=True)}' for key, value in action.items())

        return "{\n" + fields + "\n}"

    def _panel(
        self, page: int, action_lines: list[str], action_page: int, context_caption: str, inference_caption: str
    ) -> Image.Image:
        """Returns the frame without the cameras and the status line: instruction, tile labels, usage and action."""
        layout = self.layout
        panel = Image.new("RGB", VIDEO_SIZE, self.BACKGROUND_COLOUR)
        image_draw = ImageDraw.Draw(panel)
        self._draw_instruction(image_draw, page)

        for role, (tile_left, _, width, _) in layout.tiles.items():
            image_draw.rectangle(
                (
                    tile_left,
                    self.TILE_HEADER_Y,
                    tile_left + width - 1,
                    self.TILE_HEADER_Y + self.TILE_HEADER_HEIGHT - 1,
                ),
                fill=self.TILE_HEADER_COLOUR,
            )
            image_draw.text(
                (tile_left + self.TILE_LABEL_OFFSET[0], self.TILE_LABEL_OFFSET[1]),
                self.camera_labels[role],
                font=self.label_font,
                fill=self.TEXT_COLOUR,
            )

        page_count = math.ceil(len(action_lines) / layout.lines_per_page)
        suffix = f"  |  Page {action_page + 1}/{page_count}" if page_count > 1 else ""
        image_draw.text(layout.inference, inference_caption, font=self.label_font, fill=self.ACCENT_COLOUR)
        image_draw.text(
            layout.context, context_caption, font=self.label_font, fill=self.ACCENT_COLOUR, anchor=layout.context_anchor
        )
        image_draw.text(layout.heading, "MODEL ACTION (JSON)" + suffix, font=self.label_font, fill=self.ACCENT_COLOUR)

        first_line = action_page * layout.lines_per_page
        code_left, code_top = layout.code
        for line_index, line in enumerate(action_lines[first_line : first_line + layout.lines_per_page]):
            image_draw.text(
                (code_left, code_top + line_index * self.CODE_PITCH), line, font=self.code_font, fill=self.CODE_COLOUR
            )

        return panel

    def _draw_instruction(self, image_draw: ImageDraw.ImageDraw, page: int) -> None:
        """Draws the instruction's heading, the model's caption and one page of the instruction."""
        page_count = self._instruction_page_count()
        suffix = f"  |  Page {page + 1}/{page_count}" if page_count > 1 else ""
        heading = "TASK INSTRUCTION" + suffix
        image_draw.text(
            (CaptionedLayout.MARGIN, self.HEADING_Y), heading, font=self.label_font, fill=self.ACCENT_COLOUR
        )

        # the model stays on every page, right of the heading
        width = self.TEXT_WIDTH - self.label_font.getlength(heading) - CaptionedLayout.MARGIN
        caption = self._wrap(self.model_caption, self.label_font, width)[0]
        if caption != self.model_caption:
            while self.label_font.getlength(caption + self.ELLIPSIS) > width:
                caption = caption[:-1]
            caption += self.ELLIPSIS
        image_draw.text(
            (VIDEO_SIZE[0] - CaptionedLayout.MARGIN, self.HEADING_Y),
            caption,
            font=self.label_font,
            fill=self.ACCENT_COLOUR,
            anchor="ra",
        )

        first_line = page * self.INSTRUCTION_LINES_PER_PAGE
        lines = self.instruction_lines[first_line : first_line + self.INSTRUCTION_LINES_PER_PAGE]
        for line_index, line in enumerate(lines):
            image_draw.text(
                (CaptionedLayout.MARGIN, self.INSTRUCTION_Y + line_index * self.INSTRUCTION_PITCH),
                line,
                font=self.body_font,
                fill=self.TEXT_COLOUR,
            )

    def _paste_cameras(self, frame_image: Image.Image, frames: dict[str, Any]) -> None:
        """Pastes each camera's picture into its tile, and marks the tile of a camera without one."""
        image_draw = ImageDraw.Draw(frame_image)
        for role, (tile_left, tile_top, width, height) in self.layout.tiles.items():
            frame = frames.get(role)
            if frame is not None:
                frame_image.paste(Image.fromarray(frame["rgb"]).resize((width, height)), (tile_left, tile_top))
                continue

            image_draw.rectangle(
                (tile_left, tile_top, tile_left + width - 1, tile_top + height - 1), fill=self.EMPTY_TILE_COLOUR
            )
            offset_x, offset_y = self.layout.unavailable_label_offset
            image_draw.text(
                (tile_left + offset_x, tile_top + offset_y),
                "CAMERA UNAVAILABLE",
                font=self.label_font,
                fill=self.TEXT_COLOUR,
            )

    def _draw_status(self, image_draw: ImageDraw.ImageDraw, state: dict[str, Any], elapsed_s: float) -> None:
        """Draws the turn, the phase and the task time, and the badge."""
        minutes, seconds = divmod(elapsed_s, 60)
        task_time_text = f"{int(minutes):02d}:{seconds:06.3f}"
        phase = state.get("phase", CaptionPhase.STARTING.value)
        turn = state.get("turn_number") or "-"
        image_draw.text(
            self.layout.status,
            f"TURN {turn}   /   {phase.upper()}   /   TASK TIME {task_time_text}",
            font=self.title_font,
            fill=self.TEXT_COLOUR,
        )
        image_draw.text(self.layout.badge, self.BADGE, font=self.label_font, fill=self.BADGE_COLOUR)

    @classmethod
    def _font(cls, size: int, *, is_monospaced: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """Returns the host's font at ``size``, or Pillow's own where the host has none."""
        try:
            return ImageFont.truetype(cls.MONOSPACED_FONT_FILE if is_monospaced else cls.FONT_FILE, size)
        except OSError:
            return ImageFont.load_default(size=size)

    @staticmethod
    def _wrap(text: object, face: ImageFont.FreeTypeFont | ImageFont.ImageFont, width: float) -> list[str]:
        """Returns the lines of ``text`` wrapped to ``width`` pixels, a word broken where it is too long."""
        lines = []
        for paragraph in str(text).splitlines() or [""]:
            line = ""
            for character in paragraph:
                if line and face.getlength(line + character) > width:
                    lines.append(line)
                    line = ""
                line += character
            lines.append(line)

        return lines
