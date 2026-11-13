import argparse
import hashlib
import shutil
from pathlib import Path


class TaskPlacer:
    """Checks the files of LIBERO-PRO's perturbed tasks in a download of its dataset against their checksums and places
    their task sets where LIBERO-PRO's registry looks for them."""

    READ_SIZE = 1 << 20  # bytes hashed at a time

    def __init__(self, args: argparse.Namespace) -> None:
        self.download = Path(args.download)
        self.destination = Path(args.destination)  # the libero package of the checkout
        # a line: "<sha256>  <bddl_files | init_files>/<task set>/<file>"
        self.checksum_by_file = dict(
            reversed(line.split(maxsplit=1)) for line in Path(args.checksums).read_text().splitlines()
        )

    @classmethod
    def from_command_line(cls) -> "TaskPlacer":
        parser = argparse.ArgumentParser(description=cls.__doc__)
        parser.add_argument("--download", required=True, help="the dataset's download")
        parser.add_argument("--checksums", required=True, help="the SHA-256 of every file of the perturbed tasks")
        parser.add_argument("--destination", required=True, help="the libero package of the LIBERO-PRO checkout")
        return cls(parser.parse_args())

    def run(self) -> None:
        other_files = [name for name, checksum in self.checksum_by_file.items() if self.sha256(name) != checksum]
        if other_files:
            raise SystemExit(
                f"{len(other_files)} files of {self.download} are missing or not the pinned ones, the first: "
                f"{other_files[0]}"
            )

        # a task set is a directory the checksums name: only those are placed
        task_set_dirs = sorted({str(Path(name).parent) for name in self.checksum_by_file})
        for task_set_dir in task_set_dirs:
            task_set_destination = self.destination / task_set_dir
            if task_set_destination.exists():
                shutil.rmtree(task_set_destination)
            shutil.copytree(self.download / task_set_dir, task_set_destination)
        print(f"{len(task_set_dirs)} directories of perturbed tasks in {self.destination}")

    def sha256(self, name: str) -> str | None:
        """The SHA-256 of a file of the download, None when it is not there."""
        path = self.download / name
        if not path.is_file():
            return None

        digest = hashlib.sha256()
        with path.open("rb") as file:
            while chunk := file.read(self.READ_SIZE):
                digest.update(chunk)
        return digest.hexdigest()


if __name__ == "__main__":
    TaskPlacer.from_command_line().run()
