import argparse
import hashlib
import importlib.util
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from repository import REPO

# the program the installer downloads the remote assets of RoboLab's scenes with
PROGRAM = REPO / "scripts/install/sim/robolab/download_remote_assets.py"
# a scene names its material by plain http; the module reads one texture next to it
MATERIAL_URL = "http://content.s3.us-west-2.amazonaws.com/Materials/Oak.mdl"
BUCKET_FILES = {
    "/Materials/Oak.mdl": b'mdl 1.6; export material Oak() = (texture_2d("./Textures/oak.png"));',
    "/Materials/Textures/oak.png": b"png",
}


class Bucket:
    """Stands in for the installer's download command: writes the bucket's files and keeps the URLs it was asked for."""

    def __init__(self) -> None:
        self.requested_urls = []

    def __call__(self, command: list[str]) -> subprocess.CompletedProcess:
        url, destination = command[-2:]
        self.requested_urls.append(url)
        path = urlsplit(url).path
        if path not in BUCKET_FILES:
            return subprocess.CompletedProcess(command, 1)

        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_bytes(BUCKET_FILES[path])
        return subprocess.CompletedProcess(command, 0)


class Download:
    """The downloader on a directory of its own, with the URL list and the checksums it is given."""

    def __init__(self, directory: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        spec = importlib.util.spec_from_file_location("download_remote_assets", PROGRAM)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.bucket = Bucket()
        monkeypatch.setattr(self.module.subprocess, "run", self.bucket)

        self.destination = directory / "omni_content"
        self.arguments = argparse.Namespace(
            urls=directory / "remote_assets.txt",
            checksums=directory / "remote_assets.sha256",
            destination=self.destination,
            config=directory / "omni_config/omniverse.toml",
        )
        self.arguments.urls.write_text(MATERIAL_URL + "\n")
        self.write_checksums(BUCKET_FILES)

    def write_checksums(self, files: dict[str, bytes]) -> None:
        lines = [f"{hashlib.sha256(data).hexdigest()}  content{path}\n" for path, data in files.items()]
        self.arguments.checksums.write_text("".join(lines))

    def run(self) -> None:
        self.module.RemoteAssetDownloader(self.arguments).run()

    def files(self) -> list[str]:
        return sorted(str(path.relative_to(self.destination)) for path in self.destination.rglob("*") if path.is_file())


@pytest.fixture
def download(tmp_path, monkeypatch):
    return Download(tmp_path, monkeypatch)


def test_an_asset_and_the_textures_its_module_reads_are_downloaded_over_https(download):
    download.run()

    assert download.files() == ["content/Materials/Oak.mdl", "content/Materials/Textures/oak.png"]
    assert all(url.startswith("https://") for url in download.bucket.requested_urls)
    # the alias keeps the scheme the scene names the asset by
    assert f'"http://content.s3.us-west-2.amazonaws.com/" = "file://{download.destination}/content/"' in (
        download.arguments.config.read_text()
    )


def test_a_second_run_downloads_nothing(download):
    download.run()
    first_run_url_n = len(download.bucket.requested_urls)

    download.run()

    assert len(download.bucket.requested_urls) == first_run_url_n


@pytest.mark.parametrize("pinned_files", [{"/Materials/Oak.mdl": b"another module"}, {}])
def test_a_file_without_its_pinned_checksum_ends_the_download_and_is_not_kept(download, pinned_files):
    download.write_checksums(pinned_files)

    with pytest.raises(SystemExit, match="Oak.mdl is not the file of SHA-256"):
        download.run()

    assert download.files() == []


def test_a_download_that_fails_names_its_url(download):
    download.arguments.urls.write_text("https://content.s3.us-west-2.amazonaws.com/Materials/Missing.mdl\n")

    with pytest.raises(SystemExit, match="Materials/Missing.mdl could not be downloaded"):
        download.run()
