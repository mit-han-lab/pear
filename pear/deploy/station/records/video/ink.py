from functools import cache, lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class Palette:
    """The colours of a styled frame, as RGB."""

    NIGHT = (15, 25, 35)  # the frame
    WELL = (10, 18, 25)  # behind a camera and the action
    TEXT = (233, 238, 243)  # the model name and the action's top-level keys
    SECONDARY_TEXT = (179, 189, 200)  # camera labels, inner keys and strings
    TERTIARY_TEXT = (140, 152, 166)  # labels and numbers that are zero
    PUNCTUATION = (98, 111, 125)  # JSON punctuation
    WHITE = (255, 255, 255)  # the instruction, the numbers and the playhead; thinned by LINE_OPACITY, the hairlines
    AMBER = (238, 190, 67)  # what the model does, and nothing else
    LIGHT_AMBER = (243, 212, 126)  # the model's numbers that are not zero
    LINE_OPACITY = 0.13  # of white hairlines and of the empty progress bar


class Ink:
    """Anti-aliased drawing of rounded boxes and of text onto an RGB frame."""

    FONTS_DIR = Path(__file__).resolve().parent / "fonts"  # the folder of the fonts that ship with the package
    MEDIUM_FONT = "IBMPlexSans-Medium"  # labels and the model name
    SEMIBOLD_FONT = "IBMPlexSans-SemiBold"  # the instruction and the numbers
    MONOSPACE_FONT = "JetBrainsMono-Regular"  # the action and the badge
    TEXT_PAD = 3  # pixels around a text mask, for anti-aliasing
    MASK_CACHE_SIZE = 4096  # text masks kept for reuse
    ELLIPSIS = "…"  # ends a text cut short

    @classmethod
    def draw_box(
        cls,
        frame: np.ndarray,
        box_left: float,
        box_top: float,
        box_right: float,
        box_bottom: float,
        radius: float,
        colour: tuple[int, int, int],
        opacity: float = 1.0,
        outline_width: float = 0.0,
    ) -> None:
        """Fills a rounded box, or with ``outline_width`` draws only its outline."""
        if box_right <= box_left or box_bottom <= box_top:
            return

        alpha, left, top = cls.coverage(box_left, box_top, box_right, box_bottom, radius, outline_width)
        cls.blend(frame, left, top, alpha, colour, opacity)

    @classmethod
    def draw_text(
        cls,
        frame: np.ndarray,
        anchor_x: float,
        baseline: float,
        text: str,
        font_name: str,
        size: int,
        colour: tuple[int, int, int],
        opacity: float = 1.0,
        alignment: str = "l",
    ) -> float:
        """Draws ``text`` on ``baseline`` with ``anchor_x`` as its left edge, its centre ("c") or its right edge
        ("r"), and returns its width."""
        alpha, width, ascent = cls.mask(text, font_name, size)
        left = anchor_x - width if alignment == "r" else anchor_x - width / 2 if alignment == "c" else anchor_x
        top = int(round(baseline)) - ascent - cls.TEXT_PAD
        cls.blend(frame, int(round(left)) - cls.TEXT_PAD, top, alpha, colour, opacity)

        return width

    @classmethod
    def fit(cls, text: str, font_name: str, size: int, width: float) -> str:
        """Returns ``text``, cut short with an ellipsis when it is wider than ``width``."""
        if cls.text_width(text, font_name, size) <= width:
            return text

        while text and cls.text_width(text + cls.ELLIPSIS, font_name, size) > width:
            text = text[:-1]

        return text.rstrip() + cls.ELLIPSIS

    @classmethod
    def text_width(cls, text: str, font_name: str, size: int) -> float:
        """Returns the width ``text`` takes when it is drawn."""
        return cls.font(font_name, size).getlength(text)

    @staticmethod
    def coverage(
        box_left: float, box_top: float, box_right: float, box_bottom: float, radius: float, outline_width: float = 0.0
    ) -> tuple[np.ndarray, int, int]:
        """Returns how much of each pixel a rounded box covers, with the box's top-left pixel; an ``outline_width``
        keeps only an outline that wide."""
        # the signed distance to the box's edge, sampled at the pixels' centres, one pixel of margin all round
        radius = min(radius, (box_right - box_left) / 2, (box_bottom - box_top) / 2)
        left, top = int(np.floor(box_left)) - 1, int(np.floor(box_top)) - 1
        column_centres = np.arange(left, int(np.ceil(box_right)) + 1) + 0.5
        row_centres = np.arange(top, int(np.ceil(box_bottom)) + 1) + 0.5
        outside_x = np.abs(column_centres - (box_left + box_right) / 2)[None, :] - ((box_right - box_left) / 2 - radius)
        outside_y = np.abs(row_centres - (box_top + box_bottom) / 2)[:, None] - ((box_bottom - box_top) / 2 - radius)
        distance = (
            np.hypot(np.maximum(outside_x, 0), np.maximum(outside_y, 0))
            + np.minimum(np.maximum(outside_x, outside_y), 0)
            - radius
        )

        alpha = np.clip(0.5 - distance, 0, 1)
        if outline_width:
            alpha *= np.clip(0.5 + distance + outline_width, 0, 1)

        return alpha.astype(np.float32), left, top

    @staticmethod
    def blend(
        frame: np.ndarray, left: int, top: int, alpha: np.ndarray, colour: tuple[int, int, int], opacity: float
    ) -> None:
        """Mixes ``colour`` into ``frame`` through the coverage ``alpha``, whose top-left pixel is at (left, top)."""
        height, width = alpha.shape
        clip_left, clip_top = max(left, 0), max(top, 0)
        clip_right, clip_bottom = min(left + width, frame.shape[1]), min(top + height, frame.shape[0])
        if clip_right <= clip_left or clip_bottom <= clip_top:
            return

        weight = alpha[clip_top - top : clip_bottom - top, clip_left - left : clip_right - left, None] * opacity
        region = frame[clip_top:clip_bottom, clip_left:clip_right].astype(np.float32)
        frame[clip_top:clip_bottom, clip_left:clip_right] = (
            region + (np.asarray(colour, np.float32) - region) * weight + 0.5
        ).astype(np.uint8)

    @staticmethod
    @lru_cache(maxsize=MASK_CACHE_SIZE)
    def mask(text: str, font_name: str, size: int) -> tuple[np.ndarray, float, int]:
        """Returns the coverage of ``text``, its advance width and the font's ascent."""
        face = Ink.font(font_name, size)
        ascent, descent = face.getmetrics()
        width = face.getlength(text)
        picture = Image.new("L", (int(np.ceil(width)) + 2 * Ink.TEXT_PAD, ascent + descent + 2 * Ink.TEXT_PAD), 0)
        ImageDraw.Draw(picture).text((Ink.TEXT_PAD, Ink.TEXT_PAD + ascent), text, font=face, fill=255, anchor="ls")

        return np.asarray(picture, np.float32) / 255, width, ascent

    @staticmethod
    @cache
    def font(font_name: str, size: int) -> ImageFont.FreeTypeFont:
        """Returns one of the package's fonts at ``size``."""
        return ImageFont.truetype(str(Ink.FONTS_DIR / f"{font_name}.ttf"), size)
