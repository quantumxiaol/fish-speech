"""Shared API-key validation and early authentication for native service routes."""

import hmac
from collections.abc import Mapping

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


def api_key_matches(headers: Mapping[str, str], api_key: str | None) -> bool:
    if not api_key:
        return True
    scheme, _, token = headers.get("authorization", "").partition(" ")
    candidates = [headers.get("api-key", "")]
    if scheme.lower() == "bearer":
        candidates.append(token.strip())
    return any(
        hmac.compare_digest(value.encode(), api_key.encode()) for value in candidates
    )


class FishSpeechAPIKeyMiddleware:
    """Reject unauthorized uploads before multipart parsing or route execution.

    Use pure ASGI middleware so OpenAI streaming responses pass through unchanged.
    The /v1 router uses the same validator with its own OpenAI error envelope.
    """

    def __init__(self, app: ASGIApp, api_key: str | None):
        self.app = app
        self.api_key = api_key

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "")
        root_path = scope.get("root_path", "").rstrip("/")
        if root_path and (path == root_path or path.startswith(root_path + "/")):
            path = path[len(root_path) :]
        if (
            scope["type"] == "http"
            and (path == "/fishspeech" or path.startswith("/fishspeech/"))
            and not api_key_matches(Headers(scope=scope), self.api_key)
        ):
            response = JSONResponse(
                {"detail": "Invalid or missing API key."},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)
