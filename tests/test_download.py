"""Tests for tracking.data.download.

Network is fully mocked — these tests never hit the wire. We exercise:
- the .part-then-rename pattern (atomicity)
- skip-if-exists idempotency at file and dataset level
- end-to-end extraction with a real (tiny) zip on disk
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import pytest

from tracking.data.download import (
    KittiArchive,
    download_file,
    download_kitti,
    extract_zip,
)


def _boom(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("urlretrieve should not have been called")


def test_download_file_skips_existing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dest = tmp_path / "x.zip"
    dest.write_bytes(b"already here")
    monkeypatch.setattr(urllib.request, "urlretrieve", _boom)

    download_file("http://example/x.zip", dest)

    assert dest.read_bytes() == b"already here"


def test_download_file_writes_via_part(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dest = tmp_path / "x.zip"
    seen_paths: list[str] = []

    def fake(url: str, filename: str, reporthook: object = None) -> tuple[str, None]:
        seen_paths.append(str(filename))
        Path(filename).write_bytes(b"downloaded")
        return filename, None

    monkeypatch.setattr(urllib.request, "urlretrieve", fake)

    download_file("http://example/x.zip", dest)

    # Wrote to .part, not directly to dest.
    assert seen_paths == [str(dest.with_suffix(".zip.part"))]
    # Renamed cleanly: dest exists, .part is gone.
    assert dest.read_bytes() == b"downloaded"
    assert not dest.with_suffix(".zip.part").exists()


def test_download_file_cleans_stale_part(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dest = tmp_path / "x.zip"
    part = dest.with_suffix(".zip.part")
    part.write_bytes(b"stale half-download")

    def fake(url: str, filename: str, reporthook: object = None) -> tuple[str, None]:
        # The stale .part must have been removed by now.
        assert not Path(filename).exists() or Path(filename).read_bytes() != b"stale half-download"
        Path(filename).write_bytes(b"fresh")
        return filename, None

    monkeypatch.setattr(urllib.request, "urlretrieve", fake)

    download_file("http://example/x.zip", dest)

    assert dest.read_bytes() == b"fresh"


def test_extract_zip_creates_dir_and_unpacks(tmp_path: Path) -> None:
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("a/b.txt", "hello")
        zf.writestr("a/c.txt", "world")

    out_dir = tmp_path / "extracted"
    extract_zip(zip_path, out_dir)

    assert (out_dir / "a" / "b.txt").read_text() == "hello"
    assert (out_dir / "a" / "c.txt").read_text() == "world"


def test_download_kitti_skips_when_markers_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 15-GB-saver: pre-extracted dataset must trigger zero downloads."""
    archives = (
        KittiArchive("a.zip", "fake archive a", Path("training/image_02")),
        KittiArchive("b.zip", "fake archive b", Path("training/label_02")),
    )
    for archive in archives:
        marker = tmp_path / archive.marker
        marker.mkdir(parents=True)
        (marker / "placeholder.txt").write_text("")

    monkeypatch.setattr(urllib.request, "urlretrieve", _boom)

    download_kitti(tmp_path, archives=archives)


def test_download_kitti_downloads_when_marker_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end on a fake archive: download, extract, marker appears."""
    # Build a real zip in memory whose contents land at the expected marker.
    fake_zip_path = tmp_path / "_source.zip"
    with zipfile.ZipFile(fake_zip_path, "w") as zf:
        zf.writestr("training/image_02/0000/000000.png", "")
    fake_zip_bytes = fake_zip_path.read_bytes()
    fake_zip_path.unlink()

    def fake(url: str, filename: str, reporthook: object = None) -> tuple[str, None]:
        Path(filename).write_bytes(fake_zip_bytes)
        return filename, None

    monkeypatch.setattr(urllib.request, "urlretrieve", fake)

    target = tmp_path / "out"
    archives = (KittiArchive("a.zip", "fake", Path("training/image_02")),)

    download_kitti(target, archives=archives)

    assert (target / "training" / "image_02" / "0000" / "000000.png").exists()
    # Zip stayed on disk (intentional — re-running is then a pure marker-check skip).
    assert (target / "a.zip").exists()
