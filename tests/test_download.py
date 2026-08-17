"""
Tests for the download stream helper (``stream_decrypt_to_file``) and the
CLI download error surface.

Network I/O is replaced with fake ``httpx.stream`` responses so these tests
are fully offline.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
from typing import TypeAlias

import httpx
import pytest
from Cryptodome.Cipher import AES
from Cryptodome.Util import Counter

from transferit._api import MegaAPIError
from transferit._crypto import a32_to_bytes, attr_key
from transferit._download import stream_decrypt_to_file

IterableBytes: TypeAlias = Iterable[bytes]

# A deterministic 8-element file key (4 key words + 4 MAC words).
KEY_A32 = [
    0x01020304,
    0x05060708,
    0x0A0B0C0D,
    0x0E0F1011,
    0x11121314,
    0x15161718,
    0x191A1B1C,
    0x1D1E1F20,
]


def _encrypt(key_a32: list[int], plain: bytes) -> bytes:
    """AES-CTR-encrypt ``plain`` with the same scheme the downloader decrypts."""
    aes_key = attr_key(key_a32)
    nonce = a32_to_bytes(key_a32[4:6])
    ctr = Counter.new(64, prefix=nonce, initial_value=0)
    return AES.new(aes_key, AES.MODE_CTR, counter=ctr).encrypt(plain)


class _FakeResponse:
    """Stand-in for the object ``httpx.stream`` yields."""

    def __init__(
        self,
        chunks: list[bytes],
        *,
        status_code: int = 200,
        fail_with: Exception | None = None,
    ) -> None:
        self._chunks = chunks
        self.status_code = status_code
        self._fail_with = fail_with

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://example.invalid")
            raise httpx.HTTPStatusError(
                f"Server error '{self.status_code}'",
                request=request,
                response=httpx.Response(self.status_code, request=request),
            )

    def iter_bytes(self, chunk_size: int) -> IterableBytes:
        def _gen() -> IterableBytes:
            for chunk in self._chunks:
                yield chunk
            if self._fail_with is not None:
                raise self._fail_with

        return _gen()


def _patch_stream(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[_FakeResponse | Exception],
) -> None:
    """Replace ``httpx.stream`` with one that walks through ``responses``."""
    import transferit._download as mod

    calls = {"n": 0}

    def _stream(method: str, url: str, **kwargs):
        @contextmanager
        def _cm():
            item = responses[min(calls["n"], len(responses) - 1)]
            calls["n"] += 1
            if isinstance(item, Exception):
                raise item
            yield item

        return _cm()

    monkeypatch.setattr(mod.httpx, "stream", _stream)
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)


# ---------------------------------------------------------------------------
# stream_decrypt_to_file
# ---------------------------------------------------------------------------


class TestStreamDecryptToFile:
    def test_stream_error_removes_partial_file(self, tmp_path, monkeypatch) -> None:
        """A mid-stream network drop must not leave a half-written file behind."""
        enc = _encrypt(KEY_A32, b"x" * 200)
        err = httpx.RemoteProtocolError("server disconnected")
        _patch_stream(
            monkeypatch,
            [_FakeResponse([enc[:50]], fail_with=err)],
        )

        out = tmp_path / "f.bin"
        with pytest.raises(httpx.RemoteProtocolError):
            stream_decrypt_to_file(
                "https://example.invalid/dl/xyz", out, KEY_A32, 200, retries=0
            )

        assert not out.exists()

    def test_transient_error_is_retried(self, tmp_path, monkeypatch) -> None:
        plain = b"hello world! " * 500
        enc = _encrypt(KEY_A32, plain)
        _patch_stream(
            monkeypatch,
            [httpx.RemoteProtocolError("flaky"), _FakeResponse([enc])],
        )

        out = tmp_path / "f.bin"
        stream_decrypt_to_file(
            "https://example.invalid/dl/xyz", out, KEY_A32, len(plain), retries=1
        )

        assert out.read_bytes() == plain

    def test_retries_exhausted_reraises(self, tmp_path, monkeypatch) -> None:
        _patch_stream(monkeypatch, [httpx.ReadError("still down")])

        out = tmp_path / "f.bin"
        with pytest.raises(httpx.ReadError):
            stream_decrypt_to_file(
                "https://example.invalid/dl/xyz", out, KEY_A32, 10, retries=2
            )

        assert not out.exists()

    def test_size_mismatch_raises_and_cleans_up(self, tmp_path, monkeypatch) -> None:
        enc = _encrypt(KEY_A32, b"short")
        _patch_stream(monkeypatch, [_FakeResponse([enc])])

        out = tmp_path / "f.bin"
        with pytest.raises(MegaAPIError, match="incomplete download"):
            stream_decrypt_to_file(
                "https://example.invalid/dl/xyz", out, KEY_A32, 1000, retries=0
            )

        assert not out.exists()

    def test_bad_status_creates_no_file(self, tmp_path, monkeypatch) -> None:
        _patch_stream(monkeypatch, [_FakeResponse([], status_code=503)])

        out = tmp_path / "f.bin"
        with pytest.raises(httpx.HTTPStatusError):
            stream_decrypt_to_file(
                "https://example.invalid/dl/xyz", out, KEY_A32, 10, retries=0
            )

        assert not out.exists()


# ---------------------------------------------------------------------------
# CLI download error surface
# ---------------------------------------------------------------------------


class TestCliDownloadErrors:
    def test_httpx_error_is_wrapped_without_traceback(
        self, tmp_path, monkeypatch
    ) -> None:
        """A transient network error must surface as a clean CLI failure —
        never a silent exit or a raw traceback."""
        from click.testing import CliRunner

        import transferit_cli._download as mod

        class _BoomTransferit:
            def __enter__(self):
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def download(self, *args, **kwargs):
                raise httpx.ReadError("connection reset by peer")

        monkeypatch.setattr(mod, "Transferit", _BoomTransferit)
        runner = CliRunner()
        result = runner.invoke(mod.cmd_download, ["abcdefghijkl", "-o", str(tmp_path)])

        assert result.exit_code == 1
        output = (result.output or "") + (result.stderr or "")
        assert "download failed" in output
        assert "connection reset by peer" in output

    def test_error_names_the_file_in_flight(self, tmp_path, monkeypatch) -> None:
        """The failure message should name the file that was being written."""
        from click.testing import CliRunner

        import transferit_cli._download as mod
        from transferit import TransferNode

        class _BoomTransferit:
            def __enter__(self):
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def download(
                self,
                url_or_xh,
                output_dir,
                *,
                password=None,
                force=False,
                on_start=None,
                on_file_start=None,
                **kwargs,
            ):
                node = TransferNode(
                    handle="h1",
                    parent="",
                    kind=0,
                    name="Tall el-Zaatar 1977.mp4",
                    size=100,
                    timestamp=None,
                )
                if on_start:
                    on_start([node], 100)
                if on_file_start:
                    on_file_start(node, tmp_path / "Tall el-Zaatar 1977.mp4")
                raise httpx.RemoteProtocolError("server disconnected")

        monkeypatch.setattr(mod, "Transferit", _BoomTransferit)
        runner = CliRunner()
        result = runner.invoke(mod.cmd_download, ["abcdefghijkl", "-o", str(tmp_path)])

        assert result.exit_code == 1
        output = (result.output or "") + (result.stderr or "")
        assert "Tall el-Zaatar 1977.mp4" in output
        assert "re-run to retry" in output

    def test_main_returns_one_and_logs_unexpected_errors(self, monkeypatch) -> None:
        """main() must never let an unexpected exception escape silently."""
        import transferit_cli

        calls: list[str] = []

        class _Console:
            def print(self, msg, **kwargs):  # noqa: ANN001
                calls.append(msg)

        monkeypatch.setattr(transferit_cli, "CONSOLE", _Console())

        def _boom(standalone_mode):  # noqa: ARG001
            raise RuntimeError("internal boom")

        monkeypatch.setattr(transferit_cli, "cli", _boom)
        assert transferit_cli.main() == 1
        assert any("RuntimeError" in c and "internal boom" in c for c in calls)

    def test_main_logs_network_errors(self, monkeypatch) -> None:
        import transferit_cli

        calls: list[str] = []

        class _Console:
            def print(self, msg, **kwargs):  # noqa: ANN001
                calls.append(msg)

        monkeypatch.setattr(transferit_cli, "CONSOLE", _Console())

        def _boom(standalone_mode):  # noqa: ARG001
            raise httpx.ReadTimeout("server slow")

        monkeypatch.setattr(transferit_cli, "cli", _boom)
        assert transferit_cli.main() == 1
        assert any("network error" in c and "server slow" in c for c in calls)
