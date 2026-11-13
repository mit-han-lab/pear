from pathlib import Path

REPO = Path(__file__).resolve().parents[1]  # the repository the tests are of


class Repository:
    """The repository's files and packages, as a test that reads every one of a kind finds them."""

    NPM_DIRECTORY = "node_modules"  # what npm installs beside the robot station's page: none of it is PEAR's

    @classmethod
    def files(cls, pattern: str) -> list[Path]:
        """Returns the files of the repository that match ``pattern``, sorted, without what npm installed."""
        return sorted(path for path in REPO.glob(pattern) if path.is_file() and cls.NPM_DIRECTORY not in path.parts)

    @staticmethod
    def packages(root: Path) -> list[Path]:
        """Returns the packages directly under ``root``, sorted."""
        return sorted(path for path in root.iterdir() if (path / "__init__.py").exists())
