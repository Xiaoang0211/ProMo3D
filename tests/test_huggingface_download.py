import hashlib
from pathlib import Path
import zipfile

import pytest

from isdynamic.release.download import destination_file, download, extract_archive, install_file


def test_dry_run_does_not_download_or_create_paths(tmp_path, capsys):
    download("all", ["pi3x"], tmp_path / "models", tmp_path / "data", dry_run=True)
    assert "[plan] all" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


def test_install_verifies_and_never_overwrites(tmp_path):
    source, target = tmp_path / "checkpoint.pt", tmp_path / "release/checkpoint.pt"
    source.write_bytes(b"original probe")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    install_file(source, target, expected)
    install_file(source, target, expected)
    target.write_bytes(b"different probe")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        install_file(source, target, expected)
    assert target.read_bytes() == b"different probe"


def test_archive_extracts_only_manifest_members_and_preserves_bytes(tmp_path):
    archive = tmp_path / "rgb.zip"
    data = b"original RGB file bytes"
    member = "JPEGImages/sequence/00000.jpg"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(member, data)
    expected = {member: hashlib.sha256(data).hexdigest()}
    extract_archive(archive, tmp_path / "data", expected)
    extract_archive(archive, tmp_path / "data", expected)
    assert (tmp_path / "data" / member).read_bytes() == data
    with pytest.raises(ValueError, match="membership differs"):
        extract_archive(archive, tmp_path / "data", {})


def test_archive_cannot_write_outside_destination(tmp_path):
    with pytest.raises(ValueError, match="Unsafe release path"):
        destination_file(tmp_path / "data", "../outside.jpg")
