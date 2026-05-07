"""Cross-platform KITTI tracking dataset downloader.

Stdlib-only replacement for the original ``wget`` + ``unzip`` shell script,
so ``make download`` works on Windows, Linux, and macOS without extra
tooling.

Idempotency
-----------
If the marker directory for an archive already exists and is non-empty,
both download and extraction are skipped. If the zip is on disk but the
marker is missing, only the download is skipped.

Resilience
----------
Downloads write to ``<file>.part`` and are renamed on completion, so an
interrupted download leaves a ``.part`` file behind. The next run cleans
up that ``.part`` and re-downloads — a partial file never gets silently
treated as complete.
"""

from __future__ import annotations

import logging
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

KITTI_BASE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti"
DEFAULT_TARGET_DIR = Path("data/kitti_tracking")


@dataclass(frozen=True)
class KittiArchive:
    """One downloadable KITTI archive."""

    filename: str
    description: str
    marker: Path
    """Path relative to ``target_dir``; if it exists with content, the
    archive is already set up and we skip both download and extraction."""


KITTI_ARCHIVES: tuple[KittiArchive, ...] = (
    KittiArchive(
        filename="data_tracking_image_2.zip",
        description="left color images (training + testing), ~15 GB",
        marker=Path("training/image_02"),
    ),
    KittiArchive(
        filename="data_tracking_label_2.zip",
        description="GT labels (training only), ~9 MB",
        marker=Path("training/label_02"),
    ),
)


def _format_size(num_bytes: float) -> str:
    """Pretty-print a byte count: 1234 -> '1.2 KB', etc."""
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """``urlretrieve`` reporthook printing roughly every 5%."""
    if total_size <= 0:
        return
    downloaded = min(block_num * block_size, total_size)
    threshold = max(1, total_size // (block_size * 20))
    if block_num % threshold == 0 or downloaded >= total_size:
        pct = downloaded / total_size * 100
        sys.stdout.write(
            f"\r  {pct:5.1f}%  ({_format_size(downloaded)} / {_format_size(total_size)})"
        )
        sys.stdout.flush()
        if downloaded >= total_size:
            sys.stdout.write("\n")


def download_file(url: str, dest: Path) -> None:
    """Atomically download ``url`` to ``dest`` via a ``.part`` suffix.

    Skips entirely if ``dest`` already exists. A leftover ``.part`` from
    an earlier interrupted run is removed before retrying.
    """
    if dest.exists():
        logger.info("[skip] %s already on disk", dest.name)
        return

    part = dest.with_suffix(dest.suffix + ".part")
    if part.exists():
        logger.info("[clean] removing stale %s", part.name)
        part.unlink()

    logger.info("[get ] %s", url)
    urllib.request.urlretrieve(url, part, reporthook=_progress_hook)
    part.rename(dest)


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract ``zip_path`` into ``dest_dir`` (created if needed)."""
    logger.info("[unzip] %s", zip_path.name)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def download_kitti(
    target_dir: Path = DEFAULT_TARGET_DIR,
    archives: tuple[KittiArchive, ...] = KITTI_ARCHIVES,
) -> None:
    """Download and extract every archive into ``target_dir``.

    Idempotent: if a marker directory already exists with content, both
    download and extraction are skipped for that archive.
    """
    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    for archive in archives:
        marker = target_dir / archive.marker
        if marker.is_dir() and any(marker.iterdir()):
            logger.info(
                "[skip] %s already extracted (marker: %s)", archive.filename, archive.marker
            )
            continue

        zip_path = target_dir / archive.filename
        download_file(f"{KITTI_BASE_URL}/{archive.filename}", zip_path)
        extract_zip(zip_path, target_dir)

    logger.info("[done] KITTI tracking data ready at %s", target_dir)
