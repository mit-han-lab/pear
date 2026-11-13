import pytest

from pear.utils.file_lock import FileLock


def test_a_lock_has_one_holder_until_it_is_closed(tmp_path):
    path = tmp_path / "bus.lock"
    lock = FileLock(path)

    with pytest.raises(BlockingIOError):
        FileLock(path)

    lock.close()
    FileLock(path).close()
