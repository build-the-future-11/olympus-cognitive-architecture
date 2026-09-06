import io
import tarfile
import zipfile

import pytest

from scripts.run_external_study import copy_task


def test_zip_path_traversal_is_rejected(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="unsafe ZIP"):
        copy_task(archive, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_tar_link_is_rejected(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        handle.addfile(info, io.BytesIO())
    with pytest.raises(ValueError, match="link/device"):
        copy_task(archive, tmp_path / "out")
