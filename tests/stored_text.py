import os
from pathlib import Path


class StoredText:
    """A text a test's output must equal byte for byte, kept in a file that a test writes again only on request."""

    @staticmethod
    def assert_matches(path: Path, text: str, update_variable: str) -> None:
        """Fails the test unless ``text`` is the text stored at ``path``, which it first writes when the environment
        variable ``update_variable`` is set."""
        if os.environ.get(update_variable):
            path.write_text(text)

        assert text == path.read_text()
