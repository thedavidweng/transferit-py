"""
Pure helpers used by the download flow.

API calls (``fetch_transfer``, ``fetch_transfer_info``, ``get_download_url``)
are methods on :class:`MegaAPI`; this module only contains helpers that
don't touch the wire.
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx
from Cryptodome.Cipher import AES
from Cryptodome.Util import Counter

from ._api import MegaAPIError
from ._crypto import a32_to_bytes, attr_key


def _remove_partial(out_path: Path) -> None:
    """Best-effort delete of a half-written file.

    A partial file left behind would be treated as "already downloaded" on a
    re-run (``out_path.exists()`` → skip).  Removing it on failure ensures
    callers retry cleanly.
    """
    try:
        out_path.unlink(missing_ok=True)
    except OSError:
        pass


def stream_decrypt_to_file(
    url: str,
    out_path: Path,
    key_a32: list[int],
    size: int,
    on_progress=None,
    *,
    retries: int = 4,
) -> None:
    """
    Stream encrypted bytes from ``url``, AES-CTR-decrypt on the fly, and
    write to ``out_path``.

        key   = attr_key(filekey_a32)          # XOR-reduced node key
        nonce = filekey_a32[4:6]               # 64-bit
        counter starts at 0 (16-byte blocks)

    The file is only kept if the full ``size`` bytes were received and
    decrypted.  On any failure the partial file is removed so that a later
    call (or a re-run of the CLI) retries cleanly instead of skipping a
    corrupt half-written file.

    Transient network errors (``httpx.TransportError`` / ``httpx.StreamError``)
    are retried up to ``retries`` times with a 1, 2, 3, … second backoff.  All
    other errors (bad status, OS errors, size mismatch, …) propagate
    immediately after cleaning up the partial file.
    """
    aes_key = attr_key(key_a32)
    nonce = a32_to_bytes(key_a32[4:6])

    for attempt in range(retries + 1):
        try:
            _stream_once(url, out_path, aes_key, nonce, size, on_progress)
            return
        except (httpx.TransportError, httpx.StreamError):
            _remove_partial(out_path)
            if attempt >= retries:
                raise
            time.sleep(1 + attempt)
        except Exception:
            _remove_partial(out_path)
            raise


def _stream_once(
    url: str,
    out_path: Path,
    aes_key: bytes,
    nonce: bytes,
    size: int,
    on_progress,
) -> None:
    """One attempt at streaming + decrypting + writing ``url``."""
    ctr = Counter.new(64, prefix=nonce, initial_value=0)
    cipher = AES.new(aes_key, AES.MODE_CTR, counter=ctr)
    written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, timeout=httpx.Timeout(None, connect=30.0)) as resp:
        resp.raise_for_status()
        with out_path.open("wb") as fh:
            for chunk in resp.iter_bytes(1024 * 1024):
                if not chunk:
                    continue
                fh.write(cipher.decrypt(chunk))
                written += len(chunk)
                if on_progress:
                    on_progress(written, size)
    if written != size:
        raise MegaAPIError(
            f"incomplete download: received {written:,} of {size:,} bytes"
        )


def compute_folder_paths(nodes: list[dict], root_handle: str) -> dict[str, str]:
    """Build a ``folder_handle → posix-relative-path`` map, root = ``""``."""
    paths: dict[str, str] = {root_handle: ""}
    pending = [n for n in nodes if n["t"] == 1 and n["h"] != root_handle]
    while pending:
        made = False
        for n in list(pending):
            if n["p"] in paths:
                parent = paths[n["p"]]
                paths[n["h"]] = (f"{parent}/" if parent else "") + (n["name"] or n["h"])
                pending.remove(n)
                made = True
        if not made:
            break
    return paths
