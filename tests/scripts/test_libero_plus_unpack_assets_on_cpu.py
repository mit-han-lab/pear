import subprocess
import sys
import zipfile

from repository import REPO

# the program the installer unpacks LIBERO-plus's assets with
PROGRAM = REPO / "scripts/install/sim/libero_plus/unpack_assets.py"
PREFIX = "release/LIBERO-plus/libero/libero"  # the archive's entries sit under a prefix of several directories


def test_the_archive_s_assets_are_unpacked_without_their_prefix(tmp_path):
    archive = tmp_path / "assets.zip"
    assets = tmp_path / "assets"
    with zipfile.ZipFile(archive, "w") as archive_file:
        archive_file.writestr(f"{PREFIX}/assets/scenes/kitchen.xml", "<mujoco/>")
        archive_file.writestr(f"{PREFIX}/assets/textures/wood/oak.png", b"png")
        archive_file.writestr(f"{PREFIX}/assets/empty/", "")
        archive_file.writestr(f"{PREFIX}/README.md", "outside the assets")

    subprocess.run([sys.executable, PROGRAM, "--archive", archive, "--destination", assets], check=True)

    unpacked_files = sorted(str(path.relative_to(assets)) for path in assets.rglob("*") if path.is_file())
    assert unpacked_files == ["scenes/kitchen.xml", "textures/wood/oak.png"]
    assert (assets / "scenes/kitchen.xml").read_text() == "<mujoco/>"
