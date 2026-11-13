import argparse
import hashlib
import re
import subprocess
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit


class RemoteAssetDownloader:
    """Downloads the remote assets of RoboLab's scenes (scripts/install/sim/robolab/remote_assets.txt) under a directory, with
    the textures their MDL modules read, each checked against its SHA-256, and writes the omni.client aliases that
    serve them from there."""

    # what an MDL module reads besides itself: its textures, by paths relative to the module
    TEXTURE_PATTERN = re.compile(rb'texture_2d\(\s*"([^"]+)"')
    DOWNLOAD_SCRIPT = Path(__file__).parents[2] / "download.sh"  # the installer's one download command
    # a scene may name an asset by plain http; the buckets serve both, and a download takes the encrypted one
    PLAIN_SCHEME, DOWNLOAD_SCHEME = "http", "https"

    def __init__(self, args: argparse.Namespace) -> None:
        self.urls = Path(args.urls).read_text().split()
        self.destination = Path(args.destination)  # a file's place: <destination>/<bucket>/<path>
        self.config = Path(args.config)  # the omniverse.toml the Apps name in OMNI_CONFIG_PATH
        # a line: "<sha256>  <bucket>/<path>", a file's place under the destination
        self.checksum_by_file = dict(
            reversed(line.split(maxsplit=1)) for line in Path(args.checksums).read_text().splitlines()
        )

    @classmethod
    def from_command_line(cls) -> "RemoteAssetDownloader":
        parser = argparse.ArgumentParser(description=cls.__doc__)
        parser.add_argument("--urls", required=True, help="the remote assets' URLs, one per line")
        parser.add_argument("--checksums", required=True, help="the SHA-256 of every file the URLs lead to")
        parser.add_argument("--destination", required=True, help="the directory to download into")
        parser.add_argument("--config", required=True, help="the omni.client config to write (its aliases)")
        return cls(parser.parse_args())

    def run(self) -> None:
        queue, seen, prefixes = list(self.urls), set(), set()
        while queue:
            url = queue.pop()
            if url in seen:
                continue
            seen.add(url)
            parts = urlsplit(url)
            prefixes.add(f"{parts.scheme}://{parts.netloc}/")
            data = self.fetch(url, self.local_path(url))
            if url.endswith(".mdl"):
                queue += [urljoin(url, reference.decode()) for reference in self.TEXTURE_PATTERN.findall(data)]

        # every scheme and host form of a bucket the scenes name, to the bucket's one copy
        aliases = [f'"{prefix}" = "file://{self.destination / self.bucket(prefix)}/"' for prefix in sorted(prefixes)]
        self.config.parent.mkdir(parents=True, exist_ok=True)
        self.config.write_text("[aliases]\n" + "\n".join(aliases) + "\n")
        print(f"{len(seen)} URLs served from {self.destination}; aliases in {self.config}:\n" + "\n".join(aliases))

    def local_path(self, url: str) -> Path:
        parts = urlsplit(url)
        return self.destination / self.bucket(f"{parts.scheme}://{parts.netloc}/") / parts.path.lstrip("/")

    def fetch(self, url: str, path: Path) -> bytes:
        """The file of ``url`` with the SHA-256 pinned for ``path``, downloaded unless an earlier run did."""
        partial = path.with_name(path.name + ".partial")  # a file present under its own name is a whole one
        if not path.is_file():
            self.download(url, partial)

        data = (path if path.is_file() else partial).read_bytes()
        checksum = self.checksum_by_file.get(str(path.relative_to(self.destination)))
        if hashlib.sha256(data).hexdigest() != checksum:
            partial.unlink(missing_ok=True)
            raise SystemExit(f"{url} is not the file of SHA-256 {checksum} ({path})")
        if not path.is_file():
            partial.rename(path)

        return data

    def download(self, url: str, destination: Path) -> None:
        parts = urlsplit(url)
        if parts.scheme == self.PLAIN_SCHEME:
            parts = parts._replace(scheme=self.DOWNLOAD_SCHEME)
        download_url = urlunsplit(parts)

        # the command prints its failed attempts with the addresses they reached
        if subprocess.run(["bash", str(self.DOWNLOAD_SCRIPT), download_url, str(destination)]).returncode:
            raise SystemExit(f"{download_url} could not be downloaded")

    @staticmethod
    def bucket(prefix: str) -> str:
        """The S3 bucket of a URL prefix (its host's first label): the dotted and dashed endpoints serve one bucket."""
        return urlsplit(prefix).netloc.split(".")[0]


if __name__ == "__main__":
    RemoteAssetDownloader.from_command_line().run()
