import argparse
import shutil
import zipfile
from pathlib import Path


class AssetUnpacker:
    """Unpacks the archive of LIBERO-plus's scene assets into the package's assets directory, without the prefix its
    entries sit under."""

    PREFIX_END = "/assets/"  # an entry is <prefix>/assets/<the file's path in the assets directory>

    def __init__(self, args: argparse.Namespace) -> None:
        self.archive = Path(args.archive)
        self.destination = Path(args.destination)  # the assets directory of the libero package

    @classmethod
    def from_command_line(cls) -> "AssetUnpacker":
        parser = argparse.ArgumentParser(description=cls.__doc__)
        parser.add_argument("--archive", required=True, help="the assets.zip of the LIBERO-plus dataset")
        parser.add_argument("--destination", required=True, help="the assets directory of the libero package")
        return cls(parser.parse_args())

    def run(self) -> None:
        with zipfile.ZipFile(self.archive) as archive:
            for entry in archive.infolist():
                _, prefix_end, relative_path = entry.filename.partition(self.PREFIX_END)
                if not prefix_end or not relative_path or entry.is_dir():
                    continue

                path = self.destination / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as source, path.open("wb") as target:
                    shutil.copyfileobj(source, target)


if __name__ == "__main__":
    AssetUnpacker.from_command_line().run()
