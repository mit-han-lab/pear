from bisect import bisect_right
from dataclasses import dataclass

import numpy as np
from PIL import Image

from pear.deploy.station.control.schemas import Camera
from pear.deploy.station.records.recording.ffmpeg import FRAME_RATE, SOURCE_VIDEO_SIZE, VIDEO_SIZE
from pear.deploy.station.records.video.action_code import ActionCode
from pear.deploy.station.records.video.ink import Ink, Palette
from pear.deploy.station.records.video.task_timeline import TurnSpan


@dataclass(frozen=True)
class StyledLayout:
    """Where a styled frame puts each of its parts for a number of cameras; a box is (left, top, right, bottom)."""

    MARGIN = 40  # left and right of the frame's content
    TILE_GAP = 20  # between two camera tiles
    COLUMN_GAP = 34  # between the only camera and the action column beside it
    TILE_TOP = 180  # y of the camera tiles' top edge
    CAMERA_LABEL_BASELINE = 166  # baseline of the cameras' labels
    ROW_TILE_WIDTH = 600  # the widest tile in a row of cameras
    SINGLE_TILE_WIDTH = 960  # width of the only camera's tile
    STATISTICS_DROP = 44  # from the bottom of a row of tiles to the statistics' baseline
    CODE_DROP = 62  # from the bottom of a row of tiles to the action block
    CODE_CLEARANCE = 18  # between the action block and the progress bar
    BAR_TOP = 1030  # y of the progress bar's top edge

    tiles: tuple[tuple[int, int, int, int], ...]  # (x, y, width, height) of each camera, in the order they show
    # (left, right, baseline): "Model action" begins at left, the token count ends at right
    statistics: tuple[int, int, int]
    code_box: tuple[int, int, int, int]  # the box the action block may fill
    is_code_box_fixed: bool  # the block fills its box instead of fitting its lines

    @classmethod
    def for_cameras(cls, camera_count: int) -> "StyledLayout":
        """Returns the layout of the only camera beside the action, or of a row of cameras above it."""
        right = VIDEO_SIZE[0] - cls.MARGIN

        if camera_count == 1:
            width = cls.SINGLE_TILE_WIDTH
            height = width * SOURCE_VIDEO_SIZE[1] // SOURCE_VIDEO_SIZE[0]
            column_left = cls.MARGIN + width + cls.COLUMN_GAP
            return cls(
                tiles=((cls.MARGIN, cls.TILE_TOP, width, height),),
                statistics=(column_left, right, cls.CAMERA_LABEL_BASELINE),
                code_box=(column_left, cls.TILE_TOP, right, cls.TILE_TOP + height),
                is_code_box_fixed=True,
            )

        width = min(cls.ROW_TILE_WIDTH, (right - cls.MARGIN - (camera_count - 1) * cls.TILE_GAP) // camera_count)
        bottom = cls.TILE_TOP + width * SOURCE_VIDEO_SIZE[1] // SOURCE_VIDEO_SIZE[0]
        return cls(
            tiles=tuple(
                (cls.MARGIN + index * (width + cls.TILE_GAP), cls.TILE_TOP, width, bottom - cls.TILE_TOP)
                for index in range(camera_count)
            ),
            statistics=(cls.MARGIN, right, bottom + cls.STATISTICS_DROP),
            code_box=(cls.MARGIN, bottom + cls.CODE_DROP, right, cls.BAR_TOP - cls.CODE_CLEARANCE),
            is_code_box_fixed=False,
        )


class StyledFrames:
    """Draws the frames of a styled video for a robot's cameras."""

    LABEL_SIZE = 24  # type size of the labels
    LABEL_BASELINE = 62  # of "Task instruction" and "Model"
    TITLE_BASELINE = 112  # of the instruction and the model name
    TITLE_SIZES = range(40, 31, -2)  # an instruction on one line takes the largest that fits
    WRAPPED_TITLE_SIZE = 28  # type size of an instruction on two lines
    WRAPPED_TITLE_PITCH = 34  # between the two lines of a wrapped instruction
    WRAPPED_TITLE_RISE = 16  # of the first of two lines above TITLE_BASELINE
    TITLE_GAP = 56  # between the instruction and the model
    TITLE_OUTDENT = 1  # the instruction's large type stands this far left, so its stems line up with the label
    MODEL_NAME_SIZE = 30  # type size of the model name
    MODEL_MAX_WIDTH = 760  # a longer model name is cut short
    BADGE_SIZE = 20  # type size of the badge
    BADGE_PAD = 16  # beside the badge's text
    BADGE_RISE_AND_DROP = (29, 9)  # of the badge's top above TITLE_BASELINE, and of its bottom below it
    BADGE_TEXT_RISE = 3  # of the badge text's baseline above TITLE_BASELINE
    BADGE_GAP = 16  # between the model name and the badge
    BADGE_OPACITIES = (0.08, 0.22)  # of the badge's fill, and of its outline
    BADGE_OUTLINE_WIDTH = 1.2  # of the badge's outline
    DOT_CENTRE = (21, 10)  # of the model's amber dot: left of the name, above TITLE_BASELINE
    DOT_RADIUS = 5  # of the model's amber dot
    HALO_RADIUS = 11  # of the halo around the dot
    HALO_OPACITY = 0.16  # of the halo
    DOT_ROOM = 32  # the dot and its halo take this much, left of the name
    CAMERA_RADIUS = 18  # corner radius of a camera tile
    CAMERA_LABEL_INDENT = 1  # so that the label's stems line up with the tile's edge
    HAIRLINE_WIDTH = 1.0  # of the outline of a camera tile and of the action box
    MISSING_CAMERA_LABEL = "Camera unavailable"  # shown in the tile of a camera without a picture
    MISSING_CAMERA_LABEL_DROP = LABEL_SIZE / 3  # of that label's baseline below the tile's middle: it looks centred
    STATISTICS_SIZE = 32  # type size of the turn number and of the token count
    STATISTICS_LABEL_GAP = 14  # between a label and its number
    STATISTICS_GAP = 52  # between the turn number and the token label
    NUMBER_PAD = 2  # around a number's slot, for anti-aliasing
    MISSING_MARK = "–"  # stands for a turn or a token count that is not known
    CODE_PAD = (36, 20)  # between the action box and its text: at the sides, then above and below
    CODE_RADIUS = 18  # corner radius of the action box
    CODE_SIZE = 30  # the largest type size of the action
    MIN_CODE_SIZE = 24  # an action that would need smaller type has its objects wrapped
    CODE_PITCH = 40  # the largest line pitch of the action
    CODE_TYPE_SHARE = 0.75  # the type size is at most this share of the line pitch
    CODE_BASELINE_SHARE = 0.36  # the baseline lies this share of the type size below a line's middle
    HIGHLIGHT_INSET = (6, 5)  # of the amber wash around a number that is not zero: beyond its sides, inside its line
    HIGHLIGHT_RADIUS = 8  # corner radius of the amber wash
    HIGHLIGHT_OPACITY = 0.16  # of the amber wash
    # the syntax role of a span of the action -> its colour
    COLOURS = {
        "p": Palette.PUNCTUATION,
        "k": Palette.TEXT,
        "key": Palette.SECONDARY_TEXT,
        "s": Palette.SECONDARY_TEXT,
        "num": Palette.TERTIARY_TEXT,
        "lit": Palette.LIGHT_AMBER,
        "note": Palette.TERTIARY_TEXT,
    }
    BAR_HEIGHT = 8  # of the progress bar
    BAR_GAP = 4  # between the bar segments of two turns
    PLAYHEAD_HALF_SIZE = (2.5, 10)  # half the width and half the height of the white handle
    PLAYHEAD_CLEARANCE = (6, 13)  # half the width and half the height of the frame-coloured box around the handle
    TURN_SLIDE_S = 8 / FRAME_RATE  # the old turn number slides up and out, the new one in
    TURN_SLIDE_PX = 10  # how far the turn number slides
    FADE_EXPONENT = 2  # the old turn number fades with this power of what is left of the slide
    TOKEN_COUNT_UP_S = 16 / FRAME_RATE  # the token usage counts up to the new turn's
    FRAME_S = 1 / FRAME_RATE  # a frame shows the progress up to its own end, so the last one is full

    def __init__(
        self,
        cameras: tuple[Camera, ...],
        instruction: str,
        model_label: str,
        badge: str | None,
        turns: list[TurnSpan],
        duration_s: float,
    ) -> None:
        self.layout = StyledLayout.for_cameras(len(cameras))
        # pictures that have their tile's size already are pasted as they are
        self.tiles = {camera.role: tile for camera, tile in zip(cameras, self.layout.tiles)}
        self.turns = turns
        self.turn_starts = [turn.start_s for turn in turns]
        self.duration_s = duration_s

        self.static_layer = np.empty((VIDEO_SIZE[1], VIDEO_SIZE[0], 3), np.uint8)  # what no frame changes
        self.static_layer[:] = Palette.NIGHT
        self._draw_header(instruction, model_label, badge)

        for camera, (tile_left, tile_top, width, height) in zip(cameras, self.layout.tiles):
            label = camera.name.replace("_", " ").capitalize()
            label_x, baseline = tile_left + self.CAMERA_LABEL_INDENT, StyledLayout.CAMERA_LABEL_BASELINE
            Ink.draw_text(
                self.static_layer, label_x, baseline, label, Ink.MEDIUM_FONT, self.LABEL_SIZE, Palette.SECONDARY_TEXT
            )
            Ink.draw_box(
                self.static_layer,
                tile_left,
                tile_top,
                tile_left + width,
                tile_top + height,
                self.CAMERA_RADIUS,
                Palette.WELL,
            )

        self._draw_statistics()
        self._lay_out_code()
        self._draw_code_box()
        self.bar_segments = self._draw_progress()
        self.edges = {role: self._edge(tile) for role, tile in self.tiles.items()}
        # the static layer with what the turn shown last shows, by that turn and whether its action shows
        self.layer_key: tuple[int, bool] | None = None
        self.layer: np.ndarray | None = None

    def frame(self, images: dict[str, np.ndarray], time_s: float) -> np.ndarray:
        """Returns the RGB frame at ``time_s`` as uint8 of (height, width, 3); ``images`` holds the cameras'
        pictures by role."""
        index = bisect_right(self.turn_starts, time_s) - 1  # the current turn; -1 before the first
        key = (index, index >= 0 and time_s < self.turns[index].end_s)
        if key != self.layer_key:
            self.layer_key, self.layer = key, self._turn_layer(*key)

        frame = self.layer.copy()
        self._paste_cameras(frame, images)
        if index > 0 and time_s - self.turns[index].start_s < self.TOKEN_COUNT_UP_S:
            self._animate_statistics(frame, index, time_s - self.turns[index].start_s)
        self._draw_playhead(frame, index, time_s)

        return frame

    def _draw_header(self, instruction: str, model_label: str, badge: str | None) -> None:
        """Draws the model on the right, and the instruction in the room that is left."""
        right_edge = VIDEO_SIZE[0] - StyledLayout.MARGIN
        Ink.draw_text(
            self.static_layer,
            right_edge,
            self.LABEL_BASELINE,
            "Model",
            Ink.MEDIUM_FONT,
            self.LABEL_SIZE,
            Palette.TERTIARY_TEXT,
            alignment="r",
        )

        if badge:
            width = Ink.text_width(badge, Ink.MONOSPACE_FONT, self.BADGE_SIZE)
            badge_rise, badge_drop = self.BADGE_RISE_AND_DROP
            box = (
                right_edge - width - 2 * self.BADGE_PAD,
                self.TITLE_BASELINE - badge_rise,
                right_edge,
                self.TITLE_BASELINE + badge_drop,
            )
            radius = (badge_rise + badge_drop) / 2
            fill_opacity, outline_opacity = self.BADGE_OPACITIES
            Ink.draw_box(self.static_layer, *box, radius, Palette.WHITE, fill_opacity)
            Ink.draw_box(
                self.static_layer,
                *box,
                radius,
                Palette.WHITE,
                outline_opacity,
                outline_width=self.BADGE_OUTLINE_WIDTH,
            )
            baseline = self.TITLE_BASELINE - self.BADGE_TEXT_RISE
            Ink.draw_text(
                self.static_layer,
                right_edge - self.BADGE_PAD,
                baseline,
                badge,
                Ink.MONOSPACE_FONT,
                self.BADGE_SIZE,
                Palette.WHITE,
                alignment="r",
            )
            right_edge -= width + 2 * self.BADGE_PAD + self.BADGE_GAP

        name = Ink.fit(model_label, Ink.MEDIUM_FONT, self.MODEL_NAME_SIZE, self.MODEL_MAX_WIDTH)
        right_edge -= Ink.draw_text(
            self.static_layer,
            right_edge,
            self.TITLE_BASELINE,
            name,
            Ink.MEDIUM_FONT,
            self.MODEL_NAME_SIZE,
            Palette.TEXT,
            alignment="r",
        )
        dot_centre_x, dot_centre_y = right_edge - self.DOT_CENTRE[0], self.TITLE_BASELINE - self.DOT_CENTRE[1]
        for radius, opacity in ((self.HALO_RADIUS, self.HALO_OPACITY), (self.DOT_RADIUS, 1.0)):
            Ink.draw_box(
                self.static_layer,
                dot_centre_x - radius,
                dot_centre_y - radius,
                dot_centre_x + radius,
                dot_centre_y + radius,
                radius,
                Palette.AMBER,
                opacity,
            )
        right_edge -= self.DOT_ROOM

        left = StyledLayout.MARGIN
        Ink.draw_text(
            self.static_layer,
            left,
            self.LABEL_BASELINE,
            "Task instruction",
            Ink.MEDIUM_FONT,
            self.LABEL_SIZE,
            Palette.TERTIARY_TEXT,
        )
        room, title_x = right_edge - self.TITLE_GAP - left, left - self.TITLE_OUTDENT
        size = next(
            (size for size in self.TITLE_SIZES if Ink.text_width(instruction, Ink.SEMIBOLD_FONT, size) <= room), None
        )
        if size:
            Ink.draw_text(
                self.static_layer, title_x, self.TITLE_BASELINE, instruction, Ink.SEMIBOLD_FONT, size, Palette.WHITE
            )
            return

        # a long instruction wraps onto two lines; the second ends in an ellipsis when it is still too long
        size, lines = self.WRAPPED_TITLE_SIZE, [""]
        for word in instruction.split():
            candidate_line = f"{lines[-1]} {word}".strip()
            if not lines[-1] or Ink.text_width(candidate_line, Ink.SEMIBOLD_FONT, size) <= room:
                lines[-1] = candidate_line
            else:
                lines.append(word)
        first_line, *other_lines = lines
        lines = [Ink.fit(text, Ink.SEMIBOLD_FONT, size, room) for text in (first_line, " ".join(other_lines)) if text]
        for line_index, text in enumerate(lines):
            baseline = self.TITLE_BASELINE
            if len(lines) > 1:
                baseline += line_index * self.WRAPPED_TITLE_PITCH - self.WRAPPED_TITLE_RISE
            Ink.draw_text(self.static_layer, title_x, baseline, text, Ink.SEMIBOLD_FONT, size, Palette.WHITE)

    def _draw_statistics(self) -> None:
        """Draws the labels of the statistics, each in a slot of a fixed place, and lays out the numbers' slots."""
        row_left, row_right, baseline = self.layout.statistics
        Ink.draw_text(
            self.static_layer,
            row_left,
            baseline,
            "Model action",
            Ink.MEDIUM_FONT,
            self.LABEL_SIZE,
            Palette.TERTIARY_TEXT,
        )

        # the widest count, with every digit drawn as 0, and the widest turn number give the slots their widths
        digits_to_zero = str.maketrans("123456789", "000000000")
        counts = [self._count_text(turn.tokens).translate(digits_to_zero) for turn in self.turns] or [self.MISSING_MARK]
        token_width = max(Ink.text_width(count, Ink.SEMIBOLD_FONT, self.STATISTICS_SIZE) for count in counts)
        numbers = [str(turn.number) for turn in self.turns] or [self.MISSING_MARK]
        turn_width = max(Ink.text_width(number, Ink.SEMIBOLD_FONT, self.STATISTICS_SIZE) for number in numbers)

        label_right = row_right - token_width - self.STATISTICS_LABEL_GAP
        label_right -= Ink.draw_text(
            self.static_layer,
            label_right,
            baseline,
            "Token usage",
            Ink.MEDIUM_FONT,
            self.LABEL_SIZE,
            Palette.TERTIARY_TEXT,
            alignment="r",
        )
        self.turn_x = label_right - self.STATISTICS_GAP - turn_width  # left edge of the turn number
        label_x = self.turn_x - self.STATISTICS_LABEL_GAP
        Ink.draw_text(
            self.static_layer,
            label_x,
            baseline,
            "Turn",
            Ink.MEDIUM_FONT,
            self.LABEL_SIZE,
            Palette.TERTIARY_TEXT,
            alignment="r",
        )

        pad = self.NUMBER_PAD
        # (left, right) of the turn number's slot and of the token count's
        self.number_slots = [
            (int(self.turn_x) - pad, int(self.turn_x + turn_width) + pad),
            (int(row_right - token_width) - pad, int(row_right) + pad),
        ]
        # (top, bottom) of the rows the two numbers and their slide take
        self.number_rows = (
            baseline - self.STATISTICS_SIZE - self.TURN_SLIDE_PX,
            baseline + self.TURN_SLIDE_PX + pad,
        )

    def _lay_out_code(self) -> None:
        """Lays out each turn's action, with one type size and line pitch that fit the widest and the longest."""
        box_left, box_top, box_right, box_bottom = self.layout.code_box
        pad_x, pad_y = self.CODE_PAD
        width, height = box_right - box_left - 2 * pad_x, box_bottom - box_top - 2 * pad_y
        # the width of a character for each pixel of type size
        advance = Ink.text_width("0", Ink.MONOSPACE_FONT, self.CODE_SIZE) / self.CODE_SIZE

        self.codes = [ActionCode(turn.action) for turn in self.turns]
        if width / (advance * max((code.column_count for code in self.codes), default=1)) < self.MIN_CODE_SIZE:
            max_columns = int(width / (advance * self.MIN_CODE_SIZE))
            self.codes = [ActionCode(turn.action, max_columns) for turn in self.turns]

        self.code_line_count = max((code.line_count for code in self.codes), default=1)
        column_count = max((code.column_count for code in self.codes), default=1)
        self.code_pitch = min(self.CODE_PITCH, height // self.code_line_count)
        self.code_size = int(
            min(self.CODE_SIZE, self.code_pitch * self.CODE_TYPE_SHARE, width / (advance * column_count))
        )
        self.code_advance = Ink.text_width("0", Ink.MONOSPACE_FONT, self.code_size)
        self.code_origin = (box_left + pad_x, box_top + pad_y)

    def _draw_code_box(self) -> None:
        """Draws the box of the action block, with its outline and its tag."""
        box_left, box_top, box_right, box_bottom = self.layout.code_box
        if not self.layout.is_code_box_fixed:
            box_bottom = box_top + 2 * self.CODE_PAD[1] + self.code_pitch * self.code_line_count

        Ink.draw_box(self.static_layer, box_left, box_top, box_right, box_bottom, self.CODE_RADIUS, Palette.WELL)
        Ink.draw_box(
            self.static_layer,
            box_left,
            box_top,
            box_right,
            box_bottom,
            self.CODE_RADIUS,
            Palette.WHITE,
            Palette.LINE_OPACITY,
            outline_width=self.HAIRLINE_WIDTH,
        )
        tag_right, baseline = box_right - self.CODE_PAD[0], self._code_baseline(0)
        Ink.draw_text(
            self.static_layer,
            tag_right,
            baseline,
            "JSON",
            Ink.MEDIUM_FONT,
            self.LABEL_SIZE,
            Palette.TERTIARY_TEXT,
            alignment="r",
        )

    def _draw_progress(self) -> list[tuple[float, float]]:
        """Draws one empty bar segment for each turn, as long as the turn, and returns each segment's (left,
        right)."""
        bar_left, bar_right = StyledLayout.MARGIN, VIDEO_SIZE[0] - StyledLayout.MARGIN
        top, last_index = StyledLayout.BAR_TOP, len(self.turns) - 1

        bar_segments = []
        for turn_index, turn in enumerate(self.turns):
            start = (
                bar_left
                + turn.start_s / self.duration_s * (bar_right - bar_left)
                + (self.BAR_GAP / 2 if turn_index else 0)
            )
            end = (
                bar_left
                + turn.end_s / self.duration_s * (bar_right - bar_left)
                - (self.BAR_GAP / 2 if turn_index < last_index else 0)
            )
            bar_segments.append((start, end))
            radius = self.BAR_HEIGHT / 2
            Ink.draw_box(
                self.static_layer, start, top, end, top + self.BAR_HEIGHT, radius, Palette.WHITE, Palette.LINE_OPACITY
            )

        return bar_segments

    def _edge(
        self, tile: tuple[int, int, int, int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns the pixels of a camera tile that its rounded corners and its outline change: their rows and
        columns, their two weights and the background behind them."""
        tile_left, tile_top, width, height = tile
        alpha = Ink.coverage(0, 0, width, height, self.CAMERA_RADIUS)[0][1:-1, 1:-1]
        outline = Ink.coverage(0, 0, width, height, self.CAMERA_RADIUS, self.HAIRLINE_WIDTH)[0][1:-1, 1:-1]
        rows, columns = np.nonzero((alpha < 1) | (outline > 0))
        background = self.static_layer[tile_top + rows, tile_left + columns].astype(np.float32)

        return (
            rows,
            columns,
            alpha[rows, columns, None],
            outline[rows, columns, None] * Palette.LINE_OPACITY,
            background,
        )

    def _turn_layer(self, index: int, is_showing_action: bool) -> np.ndarray:
        """Returns the static layer with what a turn shows for its whole span: its action, its numbers and the
        bar segments of the turns before it."""
        layer = self.static_layer.copy()
        if index < 0:
            return layer

        turn = self.turns[index]
        if is_showing_action:
            self._draw_code(layer, self.codes[index])

        _, row_right, baseline = self.layout.statistics
        Ink.draw_text(
            layer, self.turn_x, baseline, str(turn.number), Ink.SEMIBOLD_FONT, self.STATISTICS_SIZE, Palette.WHITE
        )
        count_text = self._count_text(turn.tokens)
        Ink.draw_text(
            layer,
            row_right,
            baseline,
            count_text,
            Ink.SEMIBOLD_FONT,
            self.STATISTICS_SIZE,
            Palette.WHITE,
            alignment="r",
        )

        top, radius = StyledLayout.BAR_TOP, self.BAR_HEIGHT / 2
        for start, end in self.bar_segments[:index]:
            Ink.draw_box(layer, start, top, end, top + self.BAR_HEIGHT, radius, Palette.AMBER)

        return layer

    def _draw_code(self, layer: np.ndarray, code: ActionCode) -> None:
        """Draws an action as plain JSON; a number that is not zero is amber on an amber wash."""
        left, top = self.code_origin
        inset_x, inset_y = self.HIGHLIGHT_INSET

        for line_index, column, text, syntax_role in code.spans:
            span_left = left + column * self.code_advance
            if syntax_role == "lit":
                wash_top = top + line_index * self.code_pitch + inset_y
                wash_bottom = top + (line_index + 1) * self.code_pitch - inset_y
                wash_right = span_left + len(text) * self.code_advance + inset_x
                Ink.draw_box(
                    layer,
                    span_left - inset_x,
                    wash_top,
                    wash_right,
                    wash_bottom,
                    self.HIGHLIGHT_RADIUS,
                    Palette.AMBER,
                    self.HIGHLIGHT_OPACITY,
                )
            Ink.draw_text(
                layer,
                span_left,
                self._code_baseline(line_index),
                text,
                Ink.MONOSPACE_FONT,
                self.code_size,
                self.COLOURS[syntax_role],
            )

    def _paste_cameras(self, frame: np.ndarray, images: dict[str, np.ndarray]) -> None:
        """Pastes each camera's picture into its tile, with the tile's rounded corners and outline; the tile of a
        camera without a picture says so."""
        for role, (tile_left, tile_top, width, height) in self.tiles.items():
            tile = frame[tile_top : tile_top + height, tile_left : tile_left + width]
            image = images.get(role)
            if image is None:
                baseline = tile_top + height / 2 + self.MISSING_CAMERA_LABEL_DROP
                Ink.draw_text(
                    frame,
                    tile_left + width / 2,
                    baseline,
                    self.MISSING_CAMERA_LABEL,
                    Ink.MEDIUM_FONT,
                    self.LABEL_SIZE,
                    Palette.TERTIARY_TEXT,
                    alignment="c",
                )
            elif image.shape[:2] != (height, width):
                tile[:] = np.asarray(Image.fromarray(image).resize((width, height), Image.Resampling.BILINEAR))
            else:
                tile[:] = image

            # the corners are rounded onto the background, and the hairline outline is drawn
            rows, columns, alpha, outline, background = self.edges[role]
            pixels = background + (tile[rows, columns].astype(np.float32) - background) * alpha
            pixels += (255 - pixels) * outline
            tile[rows, columns] = (pixels + 0.5).astype(np.uint8)

    def _animate_statistics(self, frame: np.ndarray, index: int, elapsed_s: float) -> None:
        """Draws the turn's first moments: the old turn number slides up and out while the new one rises in, and
        the token usage counts up."""
        top, bottom = self.number_rows
        for slot_left, slot_right in self.number_slots:
            frame[top:bottom, slot_left:slot_right] = self.static_layer[top:bottom, slot_left:slot_right]

        turn, previous_turn = self.turns[index], self.turns[index - 1]
        _, row_right, baseline = self.layout.statistics
        size, shift = self.STATISTICS_SIZE, self.TURN_SLIDE_PX

        slide = self._ease(elapsed_s / self.TURN_SLIDE_S)
        if slide < 1:
            old_number_text = str(previous_turn.number)
            Ink.draw_text(
                frame,
                self.turn_x,
                baseline - slide * shift,
                old_number_text,
                Ink.SEMIBOLD_FONT,
                size,
                Palette.WHITE,
                (1 - slide) ** self.FADE_EXPONENT,
            )
        new_number_text = str(turn.number)
        Ink.draw_text(
            frame,
            self.turn_x,
            baseline + (1 - slide) * shift,
            new_number_text,
            Ink.SEMIBOLD_FONT,
            size,
            Palette.WHITE,
            slide,
        )

        tokens = turn.tokens
        if tokens is not None and previous_turn.tokens is not None:
            progress = self._ease(elapsed_s / self.TOKEN_COUNT_UP_S)
            tokens = round(previous_turn.tokens + (tokens - previous_turn.tokens) * progress)
        Ink.draw_text(
            frame,
            row_right,
            baseline,
            self._count_text(tokens),
            Ink.SEMIBOLD_FONT,
            size,
            Palette.WHITE,
            alignment="r",
        )

    def _draw_playhead(self, frame: np.ndarray, index: int, time_s: float) -> None:
        """Fills the current turn's bar segment up to ``time_s`` and marks that moment with the playhead."""
        bar_left, bar_right = StyledLayout.MARGIN, VIDEO_SIZE[0] - StyledLayout.MARGIN
        top, bottom = StyledLayout.BAR_TOP, StyledLayout.BAR_TOP + self.BAR_HEIGHT
        position = bar_left + min((time_s + self.FRAME_S) / self.duration_s, 1) * (bar_right - bar_left)

        if index >= 0:
            start, end = self.bar_segments[index]
            fill_right = min(end, max(position, start + self.BAR_HEIGHT))
            Ink.draw_box(frame, start, top, fill_right, bottom, self.BAR_HEIGHT / 2, Palette.AMBER)

        bar_middle_y = (top + bottom) / 2
        handle_half_width, handle_half_height = self.PLAYHEAD_HALF_SIZE
        box_half_width, box_half_height = self.PLAYHEAD_CLEARANCE
        handle_centre_x = min(max(position, bar_left + handle_half_width), bar_right - handle_half_width)
        Ink.draw_box(
            frame,
            handle_centre_x - box_half_width,
            bar_middle_y - box_half_height,
            handle_centre_x + box_half_width,
            bar_middle_y + box_half_height,
            box_half_width,
            Palette.NIGHT,
        )
        Ink.draw_box(
            frame,
            handle_centre_x - handle_half_width,
            bar_middle_y - handle_half_height,
            handle_centre_x + handle_half_width,
            bar_middle_y + handle_half_height,
            handle_half_width,
            Palette.WHITE,
        )

    def _code_baseline(self, line_index: int) -> int:
        """Returns the baseline of a line of the action."""
        return (
            self.code_origin[1]
            + line_index * self.code_pitch
            + round(self.code_pitch / 2 + self.code_size * self.CODE_BASELINE_SHARE)
        )

    @classmethod
    def _count_text(cls, tokens: int | None) -> str:
        """Returns a token count as text, the mark of a count that is not known for None."""
        return cls.MISSING_MARK if tokens is None else f"{tokens:,}"

    @staticmethod
    def _ease(progress: float) -> float:
        """Returns the cubic ease-out of a progress, which is clamped to [0, 1]."""
        return 1 - (1 - min(max(progress, 0.0), 1.0)) ** 3
