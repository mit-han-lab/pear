from __future__ import annotations

import ctypes
from io import BytesIO

# Wand before cv2: ImageMagick's gobject needs a newer glib than opencv-python's, and loaded after cv2 it fails;
# loaded first, conda-forge's glib serves both (LiberoPlusEnv._bind_rendering)
from wand.api import library as wandlibrary
from wand.image import Image as WandImage

# isort: split
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import zoom as scizoom
from skimage.filters import gaussian


class SensorNoise:
    """One of LIBERO-plus's sensor noises on one square RGB frame, drawn from the frame's own generator at the
    benchmark's severity tables."""

    # the frame size the benchmark's tables were tuned at: blur sigmas and radii and the glass shift scale with the
    # frame's size over it; the zoom factors and the fog map are relative to the frame already
    REFERENCE_SIZE = 256
    LEVELS_PER_TYPE = 10  # severities per noise type: the benchmark's ids 1-10 are the first type's, 11-20 the next's
    MOTION_ANGLE_DEG = 45  # the motion blur's direction lies within this of the horizontal, either way
    FOG_WIBBLE = 100.0  # the random amplitude of the fog map's first, coarsest pass
    # (radius, sigma) of the motion blur per level
    MOTION_BLUR_LEVELS = [(5, 2), (8, 3), (10, 4), (12, 5), (15, 6), (18, 8), (20, 10), (25, 12), (30, 15), (35, 20)]
    GAUSSIAN_BLUR_LEVELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # the Gaussian blur's sigma per level
    # the zoom factors the zoom blur averages, per level
    ZOOM_BLUR_LEVELS = [
        np.arange(1, 1.11, 0.01),
        np.arange(1, 1.16, 0.01),
        np.arange(1, 1.21, 0.02),
        np.arange(1, 1.26, 0.02),
        np.arange(1, 1.31, 0.03),
        np.arange(1, 1.36, 0.01),
        np.arange(1, 1.41, 0.01),
        np.arange(1, 1.46, 0.02),
        np.arange(1, 1.51, 0.02),
        np.arange(1, 1.56, 0.03),
    ]
    # (strength, wibble decay) of the fog per level
    FOG_LEVELS = [
        (0.5, 3),
        (1.0, 2.8),
        (1.5, 2.5),
        (2.0, 2.2),
        (2.5, 2.0),
        (3.0, 1.8),
        (3.5, 1.6),
        (4.0, 1.5),
        (4.5, 1.4),
        (5.0, 1.3),
    ]
    # (sigma, max shift, iterations) of the glass blur per level
    GLASS_BLUR_LEVELS = [
        (0.5, 1, 3),
        (0.7, 1, 3),
        (0.9, 2, 3),
        (1.0, 2, 2),
        (1.1, 3, 2),
        (1.3, 3, 2),
        (1.5, 4, 2),
        (1.8, 4, 2),
        (2.2, 5, 1),
        (2.5, 5, 1),
    ]

    def __init__(self, noise_id: int, frame: np.ndarray, generator: np.random.Generator) -> None:
        # the types in the order of the benchmark's noise ids
        noise_types = (self._motion_blur, self._gaussian_blur, self._zoom_blur, self._fog, self._glass_blur)
        assert 1 <= noise_id <= len(noise_types) * self.LEVELS_PER_TYPE, noise_id

        self.noise_type = noise_types[(noise_id - 1) // self.LEVELS_PER_TYPE]
        self.level = (noise_id - 1) % self.LEVELS_PER_TYPE  # the row of the type's table
        self.frame = frame  # H x W x 3 uint8, square
        self.scale = frame.shape[0] / self.REFERENCE_SIZE
        self.generator = generator

    def apply(self) -> np.ndarray:
        """The frame with the noise applied, uint8."""
        return np.clip(self.noise_type(), 0, 255).astype(np.uint8)

    def _motion_blur(self) -> np.ndarray:
        # ImageMagick's kernel through Wand, as the benchmark, with the generator's angle
        wandlibrary.MagickMotionBlurImage.argtypes = (
            ctypes.c_void_p,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
        )
        radius, sigma = self.MOTION_BLUR_LEVELS[self.level]
        png = BytesIO()
        Image.fromarray(self.frame).save(png, format="PNG")

        with WandImage(blob=png.getvalue()) as wand_image:
            angle = float(self.generator.uniform(-self.MOTION_ANGLE_DEG, self.MOTION_ANGLE_DEG))
            wandlibrary.MagickMotionBlurImage(wand_image.wand, radius * self.scale, sigma * self.scale, angle)
            blurred_blob = wand_image.make_blob()
        blurred_frame = cv2.imdecode(np.frombuffer(blurred_blob, np.uint8), cv2.IMREAD_UNCHANGED)

        # BGR (or grey) -> RGB
        return blurred_frame[..., [2, 1, 0]] if blurred_frame.ndim == 3 else np.stack([blurred_frame] * 3, axis=-1)

    def _gaussian_blur(self) -> np.ndarray:
        return (
            gaussian(self.frame / 255.0, sigma=self.GAUSSIAN_BLUR_LEVELS[self.level] * self.scale, channel_axis=-1)
            * 255
        )

    def _zoom_blur(self) -> np.ndarray:
        frame = (self.frame / 255.0).astype(np.float32)
        factors = self.ZOOM_BLUR_LEVELS[self.level]
        zoom_sum = np.zeros_like(frame)

        for factor in factors:
            zoom_sum += self._clipped_zoom(frame, factor)

        return (frame + zoom_sum) / (len(factors) + 1) * 255

    @staticmethod
    def _clipped_zoom(frame: np.ndarray, factor: float) -> np.ndarray:
        """The frame's center zoomed by ``factor``, at the frame's size."""
        size = frame.shape[0]
        crop_size = int(np.ceil(size / float(factor)))
        top = (size - crop_size) // 2
        zoomed_frame = scizoom(frame[top : top + crop_size, top : top + crop_size], (factor, factor, 1), order=1)
        trim = (zoomed_frame.shape[0] - size) // 2

        return zoomed_frame[trim : trim + size, trim : trim + size]

    def _fog(self) -> np.ndarray:
        strength, wibble_decay = self.FOG_LEVELS[self.level]
        frame = self.frame / 255.0
        brightest_value = frame.max()
        # a map as large as the frame: the benchmark's fixed 256 crashes on a larger frame
        map_size = 1 << int(np.ceil(np.log2(max(frame.shape[:2]))))
        fog_map = self._plasma_fractal(map_size, wibble_decay)[: frame.shape[0], : frame.shape[1], None]
        foggy_frame = frame + strength * fog_map

        return np.clip(foggy_frame * brightest_value / (brightest_value + strength), 0, 1) * 255

    def _plasma_fractal(self, map_size: int, wibble_decay: float) -> np.ndarray:
        """The benchmark's diamond-square heightmap in [0, 1], ``map_size`` (a power of two) per side."""
        heights = np.empty((map_size, map_size), dtype=np.float64)
        heights[0, 0] = 0
        step_size = map_size
        wibble = self.FOG_WIBBLE

        while step_size >= 2:
            half_step = step_size // 2
            # the squares' corners; their centers once the first pass has set them (two views of the map)
            corners = heights[0:map_size:step_size, 0:map_size:step_size]
            centers = heights[half_step:map_size:step_size, half_step:map_size:step_size]
            square_sums = corners + np.roll(corners, shift=-1, axis=0)
            square_sums += np.roll(square_sums, shift=-1, axis=1)
            heights[half_step:map_size:step_size, half_step:map_size:step_size] = self._wibbled_mean(
                square_sums, wibble
            )
            row_sums = centers + np.roll(centers, 1, axis=0) + corners + np.roll(corners, -1, axis=1)
            heights[0:map_size:step_size, half_step:map_size:step_size] = self._wibbled_mean(row_sums, wibble)
            column_sums = centers + np.roll(centers, 1, axis=1) + corners + np.roll(corners, -1, axis=0)
            heights[half_step:map_size:step_size, 0:map_size:step_size] = self._wibbled_mean(column_sums, wibble)
            step_size //= 2
            wibble /= wibble_decay

        heights -= heights.min()

        return heights / heights.max()

    def _wibbled_mean(self, neighbour_sums: np.ndarray, wibble: float) -> np.ndarray:
        """The mean of each cell's four neighbours, from their sum, plus ``wibble`` times a draw within ``wibble`` of
        zero."""
        return neighbour_sums / 4 + wibble * self.generator.uniform(-wibble, wibble, neighbour_sums.shape)

    def _glass_blur(self) -> np.ndarray:
        sigma, max_shift, iteration_n = self.GLASS_BLUR_LEVELS[self.level]
        sigma, max_shift = sigma * self.scale, max(1, int(round(max_shift * self.scale)))
        frame = np.uint8(gaussian(self.frame / 255.0, sigma=sigma, channel_axis=-1) * 255)
        height, width = frame.shape[:2]
        rows, columns = np.mgrid[max_shift : height - max_shift, max_shift : width - max_shift]

        # every interior pixel takes a neighbour's value within the shift (the benchmark swaps them one at a time)
        for _ in range(iteration_n):
            row_shifts = self.generator.integers(-max_shift, max_shift, size=rows.shape)
            column_shifts = self.generator.integers(-max_shift, max_shift, size=columns.shape)
            frame[max_shift : height - max_shift, max_shift : width - max_shift] = frame[
                rows + row_shifts, columns + column_shifts
            ]

        return np.clip(gaussian(frame / 255.0, sigma=sigma, channel_axis=-1), 0, 1) * 255
