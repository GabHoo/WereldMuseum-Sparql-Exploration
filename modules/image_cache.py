"""
ImageCache — thumbnails and placeholder detection for collection images.

The museum's image proxy only serves full-size originals (often 0.5–3 MB),
which makes paging through batches slow. ImageCache downloads each original
once, stores a small JPEG thumbnail on disk, and serves that instead. The same
download is used to detect the collection's "image not available" placeholder
(perceptual hash compared to a reference image), so placeholder images can be
skipped in favour of an artifact's next image.

Cache layout (persists across restarts):
    <cache_dir>/<sha1(url)>.jpg           thumbnail
    <cache_dir>/<sha1(url)>.placeholder   marker: this URL is a placeholder

Only URLs on allowed hosts are fetched, so /thumb cannot be used as an open proxy.

Requires Pillow; placeholder detection additionally needs imagehash.
"""
import hashlib
import io
import pathlib
import threading
import urllib.parse
import urllib.request
from collections import defaultdict

from PIL import Image

try:
    import imagehash
except ImportError:  # placeholder detection disabled
    imagehash = None

_HASH_THRESHOLD = 5      # max Hamming distance to count as the placeholder
_FETCH_TIMEOUT = 15      # seconds per original download


class ImageCache:
    def __init__(
        self,
        cache_dir: str,
        allowed_hosts: list[str],
        size: int = 480,
        placeholder_file: str = "",
    ):
        self._dir = pathlib.Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._hosts = {h.lower() for h in allowed_hosts if h}
        self._size = size
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._locks_guard = threading.Lock()
        self._ref_hash = None
        if imagehash and placeholder_file and pathlib.Path(placeholder_file).exists():
            self._ref_hash = imagehash.average_hash(Image.open(placeholder_file))

    # ── Public ───────────────────────────────────────────────────────────────

    def is_allowed(self, url: str) -> bool:
        parts = urllib.parse.urlsplit(url)
        return parts.scheme in ("http", "https") and (parts.hostname or "").lower() in self._hosts

    def is_placeholder(self, url: str) -> bool:
        """True if url is already known to be a placeholder (never downloads)."""
        return self._marker(url).exists()

    def thumbnail(self, url: str) -> tuple[pathlib.Path | None, bool]:
        """
        Return (thumbnail_path, is_placeholder) for url, downloading and
        resizing the original on first use. thumbnail_path is None when the
        image is a placeholder or could not be fetched/decoded.
        """
        if self.is_placeholder(url):
            return None, True
        thumb = self._thumb(url)
        if thumb.exists():
            return thumb, False

        with self._lock_for(url):  # one download per URL, even under concurrent requests
            if self.is_placeholder(url):
                return None, True
            if thumb.exists():
                return thumb, False
            try:
                with urllib.request.urlopen(url, timeout=_FETCH_TIMEOUT) as resp:
                    data = resp.read()
                img = Image.open(io.BytesIO(data))
                img.draft("RGB", (self._size * 2, self._size * 2))  # fast JPEG downscale on decode
                img = img.convert("RGB")
                img.thumbnail((self._size, self._size))
            except Exception:
                return None, False

            if self._ref_hash is not None and (imagehash.average_hash(img) - self._ref_hash) <= _HASH_THRESHOLD:
                self._marker(url).touch()
                return None, True

            tmp = thumb.with_suffix(".tmp")
            img.save(tmp, "JPEG", quality=82, optimize=True)
            tmp.replace(thumb)  # atomic: readers never see a half-written file
            return thumb, False

    # ── Private ──────────────────────────────────────────────────────────────

    def _key(self, url: str) -> str:
        return hashlib.sha1(url.encode("utf-8")).hexdigest()

    def _thumb(self, url: str) -> pathlib.Path:
        return self._dir / f"{self._key(url)}.jpg"

    def _marker(self, url: str) -> pathlib.Path:
        return self._dir / f"{self._key(url)}.placeholder"

    def _lock_for(self, url: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks[url]
