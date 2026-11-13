import io
import json
import shutil
import threading
import time
import warnings
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image, ImageOps, UnidentifiedImageError

from pear.deploy.station.agent_loop.messages import JPEG_QUALITY
from pear.deploy.station.records.task_files import TaskFiles


class ImageUploads:
    """The draft images a browser uploads for its next task, and their copies in the folder of the task that takes
    them."""

    MAX_BYTES = 10 * 1024 * 1024  # of one uploaded image; the page's service bounds its request bodies by it too
    MAX_PIXELS = 40_000_000  # of a decoded image, against decompression bombs
    MAX_SIDE = 2048  # the longest side of a stored image, in pixels
    FORMATS = ("JPEG", "PNG", "WEBP")  # the formats an upload may have
    DRAFT_LIFETIME_S = 3600  # a draft expires this long after its upload
    MAX_DRAFTS = 16  # drafts one browser may keep
    MAX_TASK_IMAGES = 4  # images one task may take
    MAX_NAME_CHARS = 255  # of an upload's file name that are kept
    IMAGE_SUFFIX = ".jpg"  # a draft's stored image is <its id>.jpg
    DESCRIPTION_SUFFIX = ".json"  # beside the image, its description: name, size, upload time and owner

    def __init__(self, root: Path) -> None:
        self.root = root  # the folder of the drafts
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()  # held by every change of the drafts
        self.cleanup()

    def add(self, owner: str, filename: str, image_bytes: bytes) -> dict[str, Any]:
        """Stores one draft image of ``owner``, re-encoded upright as JPEG, and returns its description without
        the owner."""
        jpeg_bytes, (width, height) = self.normalize(image_bytes)

        with self.lock:
            self.cleanup()
            drafts = [json.loads(path.read_text()) for path in self.root.glob("*" + self.DESCRIPTION_SUFFIX)]
            if sum(draft["owner"] == owner for draft in drafts) >= self.MAX_DRAFTS:
                raise ValueError("Too many draft images; remove unused images first")

            upload_id = uuid4().hex
            draft = {
                "id": upload_id,
                "name": Path(filename).name[: self.MAX_NAME_CHARS],
                "width": width,
                "height": height,
                "created": time.time(),
                "owner": owner,
            }
            self._image_path(upload_id).write_bytes(jpeg_bytes)
            TaskFiles.write_json(self._description_path(upload_id), draft)

            return {key: value for key, value in draft.items() if key != "owner"}

    def attach(self, upload_ids: object, owner: str, files: TaskFiles) -> list[dict[str, Any]]:
        """Copies drafts of ``owner`` into a task's folder, in the given order, and returns their descriptions."""
        if (
            not isinstance(upload_ids, list)
            or len(upload_ids) > self.MAX_TASK_IMAGES
            or any(not isinstance(upload_id, str) for upload_id in upload_ids)
            or len(set(upload_ids)) != len(upload_ids)
        ):
            raise ValueError("Choose up to four distinct images")

        with self.lock:
            drafts = [self._owned_draft(upload_id, owner) for upload_id in upload_ids]

            attached_images = []
            for draft in drafts:
                image_path = files.reference_image_path(draft["id"])
                shutil.copyfile(self._image_path(draft["id"]), image_path)
                attached_images.append(
                    {
                        "id": draft["id"],
                        "name": draft["name"],
                        "file": image_path.name,
                        "width": draft["width"],
                        "height": draft["height"],
                    }
                )

            return attached_images

    def delete(self, upload_id: str, owner: str) -> None:
        """Removes one draft of ``owner``."""
        with self.lock:
            self._owned_draft(upload_id, owner)
            self._description_path(upload_id).unlink(missing_ok=True)
            self._image_path(upload_id).unlink(missing_ok=True)

    def cleanup(self) -> None:
        """Removes the drafts that expired or cannot be read, and the images without a description."""
        with self.lock:
            for path in self.root.glob("*" + self.DESCRIPTION_SUFFIX):
                try:
                    is_expired = time.time() - json.loads(path.read_text())["created"] >= self.DRAFT_LIFETIME_S
                except (ValueError, KeyError, OSError):
                    is_expired = True
                if is_expired:
                    path.with_suffix(self.IMAGE_SUFFIX).unlink(missing_ok=True)
                    path.unlink(missing_ok=True)

            for path in self.root.glob("*" + self.IMAGE_SUFFIX):
                if not path.with_suffix(self.DESCRIPTION_SUFFIX).exists():
                    path.unlink(missing_ok=True)

    @classmethod
    def normalize(cls, image_bytes: bytes) -> tuple[bytes, tuple[int, int]]:
        """Returns an upload re-encoded as an upright RGB JPEG on white, at most MAX_SIDE on a side, with its
        size."""
        if not image_bytes or len(image_bytes) > cls.MAX_BYTES:
            raise ValueError(f"Each image must be at most {cls.MAX_BYTES >> 20} MiB")

        try:
            # Pillow's warning of a decompression bomb refuses the image like its error
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(image_bytes)) as image:
                    if image.format not in cls.FORMATS:
                        raise ValueError("Use JPEG, PNG or static WebP images")
                    if getattr(image, "n_frames", 1) != 1:
                        raise ValueError("Animated images are not supported")
                    if image.width * image.height > cls.MAX_PIXELS:
                        raise ValueError(f"Image exceeds {cls.MAX_PIXELS // 1_000_000} million pixels")

                    image.load()
                    upright_image = ImageOps.exif_transpose(image)
                    upright_image.thumbnail((cls.MAX_SIDE, cls.MAX_SIDE), Image.Resampling.LANCZOS)
                    rgba_image = upright_image.convert("RGBA")
                    rgb_image = Image.new("RGB", rgba_image.size, "white")
                    rgb_image.paste(rgba_image, mask=rgba_image.getchannel("A"))

                    # the task's copy of the file is what a model is sent, as it is
                    jpeg_file = io.BytesIO()
                    rgb_image.save(jpeg_file, "JPEG", quality=JPEG_QUALITY)

                    return jpeg_file.getvalue(), rgb_image.size
        except (
            UnidentifiedImageError,
            OSError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ) as error:
            raise ValueError("Invalid or oversized image") from error

    def _owned_draft(self, upload_id: str, owner: str) -> dict[str, Any]:
        """Returns the description of a draft ``owner`` may use, and refuses an id, an owner or a draft that is not
        valid."""
        if not isinstance(upload_id, str) or not TaskFiles.HEX_ID_PATTERN.fullmatch(upload_id):
            raise ValueError("Invalid image ID")

        try:
            draft = json.loads(self._description_path(upload_id).read_text())
        except (OSError, ValueError) as error:
            raise ValueError("Image expired or unavailable; upload it again") from error

        if draft["owner"] != owner:
            raise ValueError("Image belongs to another browser")
        if time.time() - draft["created"] >= self.DRAFT_LIFETIME_S:
            raise ValueError("Image expired; upload it again")

        return draft

    def _image_path(self, upload_id: str) -> Path:
        """Returns the stored image of the draft ``upload_id``."""
        return self.root / (upload_id + self.IMAGE_SUFFIX)

    def _description_path(self, upload_id: str) -> Path:
        """Returns the description file of the draft ``upload_id``."""
        return self.root / (upload_id + self.DESCRIPTION_SUFFIX)
