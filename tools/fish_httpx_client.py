from __future__ import annotations

import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Optional, Sequence

import httpx

API_PREFIX = "/fishspeech"
ADMIN_SHUTDOWN_HEADER = "X-Fish-Speech-Admin"
ADMIN_SHUTDOWN_VALUE = "shutdown"
DEFAULT_SERVER_URL = "http://127.0.0.1:8002"


def _form_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


class FishSpeechClientError(RuntimeError):
    pass


class FishSpeechHttpxClient:
    def __init__(
        self,
        server_url: str = DEFAULT_SERVER_URL,
        timeout: float = 600.0,
        client: Optional[httpx.Client] = None,
    ) -> None:
        normalized_server_url = server_url.rstrip("/")
        if normalized_server_url.endswith(API_PREFIX):
            normalized_server_url = normalized_server_url[: -len(API_PREFIX)]
        self.server_url = normalized_server_url
        self._client = client or httpx.Client(base_url=self.server_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "FishSpeechHttpxClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _json(self, response: httpx.Response) -> dict[str, Any]:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail: Any
            try:
                detail = response.json()
            except ValueError:
                detail = response.text
            raise FishSpeechClientError(
                f"{response.status_code} {response.reason_phrase} for "
                f"{response.request.method} {response.request.url}\n{detail}"
            ) from exc
        return response.json()

    def health(self) -> dict[str, Any]:
        return self._json(self._client.get(f"{API_PREFIX}/health"))

    def shutdown(
        self,
        *,
        wait: bool = False,
        wait_timeout: float = 30.0,
    ) -> dict[str, Any]:
        payload = self._json(
            self._client.post(
                f"{API_PREFIX}/admin/shutdown",
                headers={
                    ADMIN_SHUTDOWN_HEADER: ADMIN_SHUTDOWN_VALUE,
                    "Connection": "close",
                },
            )
        )
        if wait:
            self.wait_until_stopped(timeout=wait_timeout)
            payload["server_stopped"] = True
        return payload

    def wait_until_stopped(self, timeout: float = 30.0) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be greater than 0")

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            try:
                response = self._client.get(
                    f"{API_PREFIX}/health",
                    headers={"Connection": "close"},
                    timeout=max(min(remaining, 1.0), 0.05),
                )
                if response.status_code != 200:
                    return
                try:
                    if response.json().get("status") != "ok":
                        return
                except (AttributeError, ValueError):
                    return
            except (httpx.ConnectError, httpx.RemoteProtocolError):
                return
            except httpx.TimeoutException:
                pass
            time.sleep(min(0.2, max(remaining, 0)))

        raise FishSpeechClientError(
            f"Server still responds after waiting {timeout:.1f} seconds for shutdown."
        )

    def download_url(self, url: str, output_path: str | Path) -> Path:
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._client.stream("GET", url) as response:
            response.raise_for_status()
            with target.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
        return target

    def voice_clone(
        self,
        *,
        ref_audio_path: str | Path,
        text: Optional[str] = None,
        text_file: Optional[str | Path] = None,
        ref_text: Optional[str] = None,
        ref_text_file: Optional[str | Path] = None,
        output_name: Optional[str] = None,
        audio_format: str = "wav",
        download_to: Optional[str | Path] = None,
        **generation_kwargs: Any,
    ) -> dict[str, Any]:
        with ExitStack() as stack:
            data: dict[str, str] = {"format": audio_format}
            files: dict[str, tuple[str, Any, str]] = {}

            if text is not None:
                data["text"] = text
            if ref_text is not None:
                data["ref_text"] = ref_text
            if output_name is not None:
                data["output_name"] = output_name
            for key, value in generation_kwargs.items():
                if value is not None:
                    data[key] = _form_value(value)

            ref_audio = Path(ref_audio_path).expanduser().resolve()
            files["ref_audio"] = (
                ref_audio.name,
                stack.enter_context(ref_audio.open("rb")),
                "audio/*",
            )

            if text_file is not None:
                path = Path(text_file).expanduser().resolve()
                files["text_file"] = (
                    path.name,
                    stack.enter_context(path.open("rb")),
                    "text/plain",
                )

            if ref_text_file is not None:
                path = Path(ref_text_file).expanduser().resolve()
                files["ref_text_file"] = (
                    path.name,
                    stack.enter_context(path.open("rb")),
                    "text/plain",
                )

            response = self._client.post(
                f"{API_PREFIX}/tts/voice_clone", data=data, files=files
            )

        payload = self._json(response)
        if download_to is not None:
            self.download_url(payload["audio"]["url"], download_to)
        return payload

    def voice_clone_batch_file(
        self,
        *,
        ref_audio_path: str | Path,
        text_file: Optional[str | Path] = None,
        texts: Optional[Sequence[str]] = None,
        ref_text: Optional[str] = None,
        ref_text_file: Optional[str | Path] = None,
        output_prefix: Optional[str] = None,
        audio_format: str = "wav",
        download_dir: Optional[str | Path] = None,
        **generation_kwargs: Any,
    ) -> dict[str, Any]:
        if text_file is not None and texts:
            raise ValueError("Provide either text_file or texts, not both.")
        if text_file is None and not texts:
            raise ValueError("Either text_file or texts must be provided.")

        with ExitStack() as stack:
            data: dict[str, Any] = {"format": audio_format}
            files: dict[str, tuple[str, Any, str]] = {}

            if ref_text is not None:
                data["ref_text"] = ref_text
            if output_prefix is not None:
                data["output_prefix"] = output_prefix
            if texts:
                data["text"] = list(texts)
            for key, value in generation_kwargs.items():
                if value is not None:
                    data[key] = _form_value(value)

            ref_audio = Path(ref_audio_path).expanduser().resolve()
            files["ref_audio"] = (
                ref_audio.name,
                stack.enter_context(ref_audio.open("rb")),
                "audio/*",
            )

            if text_file is not None:
                path = Path(text_file).expanduser().resolve()
                files["text_file"] = (
                    path.name,
                    stack.enter_context(path.open("rb")),
                    "text/plain",
                )

            if ref_text_file is not None:
                path = Path(ref_text_file).expanduser().resolve()
                files["ref_text_file"] = (
                    path.name,
                    stack.enter_context(path.open("rb")),
                    "text/plain",
                )

            response = self._client.post(
                f"{API_PREFIX}/tts/voice_clone_batch_file", data=data, files=files
            )

        payload = self._json(response)
        if download_dir is not None:
            target_dir = Path(download_dir).expanduser().resolve()
            target_dir.mkdir(parents=True, exist_ok=True)
            for item in payload["audio_paths"]:
                self.download_url(item["url"], target_dir / item["filename"])
        return payload
