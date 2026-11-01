import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tools import fish_client
from tools.api_auth import FishSpeechAPIKeyMiddleware
from tools.fastapi_service import ServiceSettings, create_app
from tools.fish_httpx_client import FishSpeechClientError, FishSpeechHttpxClient


class FakeManager:
    loaded, device, dtype = False, "cpu", "float32"

    def __init__(self):
        self.requests = []

    def infer(self, request):
        self.requests.append(request)
        return 44100, np.zeros(128, dtype=np.float32)


class ServiceAuthTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manager = FakeManager()
        self.shutdowns = []
        self.settings = ServiceSettings(
            self.root,
            self.root / "model",
            self.root / "codec.pth",
            "modded_dac_vq",
            "cpu",
            "float32",
            False,
            4096,
            openai_api_key="test-key",
        )
        self.app = create_app(
            self.settings, self.manager, lambda: self.shutdowns.append(True)
        )
        self.client = TestClient(self.app, client=("127.0.0.1", 50000))
        self.addCleanup(self.client.close)
        self.file = self.root / "outputs" / "test" / "audio.wav"
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.file.write_bytes(b"test audio")

    def test_every_native_route_requires_key_even_for_loopback(self):
        routes = (
            ("GET", "/fishspeech/health"),
            ("GET", "/fishspeech/health/"),
            ("GET", "/%66ishspeech/health"),
            ("GET", "/fishspeech/files/outputs/test/audio.wav"),
            ("POST", "/fishspeech/tts/voice_clone"),
            ("POST", "/fishspeech/tts/voice_clone_batch_file"),
            ("POST", "/fishspeech/admin/shutdown"),
        )
        for method, path in routes:
            for auth in ({}, {"Authorization": "Bearer wrong"}, {"api-key": "wrong"}):
                with self.subTest(method=method, path=path, auth=auth):
                    response = self.client.request(
                        method,
                        path,
                        headers={
                            **auth,
                            "X-Fish-Speech-Admin": "shutdown",
                        },
                    )
                    self.assertEqual(response.status_code, 401)
                    self.assertEqual(response.headers["WWW-Authenticate"], "Bearer")
                    self.assertNotIn("test-key", response.text)
        self.assertEqual(self.manager.requests, [])
        self.assertEqual(self.shutdowns, [])

    def test_valid_header_forms_allow_health_download_and_models(self):
        for headers in (
            {"Authorization": "Bearer test-key"},
            {"Authorization": "bearer test-key"},
            {"api-key": "test-key"},
        ):
            for path in (
                "/fishspeech/health",
                "/fishspeech/files/outputs/test/audio.wav",
                "/v1/models",
            ):
                with self.subTest(path=path, headers=headers):
                    self.assertEqual(
                        self.client.get(path, headers=headers).status_code, 200
                    )

    def test_mounted_path_prefix_does_not_bypass_authentication(self):
        parent = FastAPI()
        parent.mount("/tts", self.app)
        with TestClient(parent) as client:
            self.assertEqual(client.get("/tts/fishspeech/health").status_code, 401)
            self.assertEqual(
                client.get(
                    "/tts/fishspeech/health", headers={"api-key": "test-key"}
                ).status_code,
                200,
            )

    def test_shutdown_still_requires_confirmation_and_loopback_after_auth(self):
        auth = {"api-key": "test-key"}
        self.assertEqual(
            self.client.post("/fishspeech/admin/shutdown", headers=auth).status_code,
            403,
        )
        headers = {**auth, "X-Fish-Speech-Admin": "shutdown"}
        with TestClient(self.app, client=("192.0.2.1", 50000)) as remote:
            self.assertEqual(
                remote.post("/fishspeech/admin/shutdown", headers=headers).status_code,
                403,
            )
        self.assertEqual(self.shutdowns, [])
        self.assertEqual(
            self.client.post("/fishspeech/admin/shutdown", headers=headers).status_code,
            202,
        )
        self.assertEqual(self.shutdowns, [True])

    def test_unset_or_empty_key_preserves_local_unauthenticated_use(self):
        for key in (None, ""):
            with TestClient(
                create_app(replace(self.settings, openai_api_key=key), self.manager)
            ) as client:
                self.assertEqual(client.get("/fishspeech/health").status_code, 200)
                self.assertEqual(client.get("/v1/models").status_code, 200)
                response = client.post(
                    "/fishspeech/tts/voice_clone",
                    data={"text": "hello", "ref_text": "reference"},
                    files={"ref_audio": ("ref.wav", b"reference audio", "audio/wav")},
                )
                self.assertEqual(response.status_code, 200)

    def test_environment_key_is_loaded_and_not_in_settings_repr(self):
        with (
            patch.dict("os.environ", {"FISH_TTS_API_KEY": "env-test-secret"}),
            patch("tools.fastapi_service.load_project_env"),
        ):
            config = ServiceSettings.from_env()
        self.assertEqual(config.openai_api_key, "env-test-secret")
        self.assertNotIn("env-test-secret", repr(config))

    def test_authenticated_client_can_synthesize_and_download_single_and_batch(self):
        reference = self.root / "ref.wav"
        reference.write_bytes(b"reference audio")
        with FishSpeechHttpxClient(client=self.client, api_key="test-key") as client:
            self.assertEqual(client.health()["status"], "ok")
            output = self.root / "download.wav"
            client.voice_clone(
                ref_audio_path=reference,
                text="hello",
                ref_text="reference",
                download_to=output,
            )
            self.assertGreater(output.stat().st_size, 0)
            batch = client.voice_clone_batch_file(
                ref_audio_path=reference,
                texts=["first", "second"],
                ref_text="reference",
                download_dir=self.root / "batch",
            )
            self.assertEqual(len(batch["audio_paths"]), 2)
            self.assertEqual(len(list((self.root / "batch").glob("*.wav"))), 2)
            client.shutdown()
        self.assertEqual(len(self.manager.requests), 3)
        self.assertEqual(self.shutdowns, [True])


class AuthMiddlewareTest(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_upload_without_reading_request_body(self):
        app = AsyncMock()
        receive = AsyncMock(side_effect=AssertionError("body must not be read"))
        send = AsyncMock()
        middleware = FishSpeechAPIKeyMiddleware(app, api_key="test-key")
        await middleware(
            {
                "type": "http",
                "method": "POST",
                "path": "/fishspeech/tts/voice_clone",
                "headers": [(b"content-type", b"multipart/form-data; boundary=test")],
            },
            receive,
            send,
        )
        app.assert_not_called()
        receive.assert_not_called()
        self.assertEqual(send.call_args_list[0].args[0]["status"], 401)


class ClientAuthTest(unittest.TestCase):
    def test_environment_and_explicit_key_precedence_and_external_download(self):
        for explicit, expected in (
            (None, "env-key"),
            ("explicit", "explicit"),
            ("", None),
        ):
            seen = []

            def handler(request):
                seen.append(request)
                return httpx.Response(200, json={"status": "ok"})

            with (
                self.subTest(explicit=explicit),
                patch("tools.fish_httpx_client.load_project_env"),
                patch.dict("os.environ", {"FISH_TTS_API_KEY": "env-key"}),
                tempfile.TemporaryDirectory() as tmp,
                FishSpeechHttpxClient(
                    api_key=explicit,
                    client=httpx.Client(
                        base_url="https://tts.example",
                        transport=httpx.MockTransport(handler),
                    ),
                ) as client,
            ):
                client.health()
                client.download_url(
                    "https://tts.example/fishspeech/files/test.wav",
                    Path(tmp) / "same.wav",
                )
                client.download_url(
                    "https://external.example/audio.wav", Path(tmp) / "external.wav"
                )
                client.download_url(
                    "http://tts.example/audio.wav", Path(tmp) / "downgrade.wav"
                )
            for request in seen[:2]:
                self.assertEqual(
                    request.headers.get("Authorization"),
                    f"Bearer {expected}" if expected else None,
                )
            for request in seen[2:]:
                self.assertNotIn("Authorization", request.headers)

    def test_shutdown_wait_auth_failure_is_not_reported_as_stopped(self):
        seen = []

        def handler(request):
            seen.append(request)
            return httpx.Response(401, json={"detail": "Invalid or missing API key."})

        with FishSpeechHttpxClient(
            api_key="test-key",
            client=httpx.Client(
                base_url="http://127.0.0.1:8002", transport=httpx.MockTransport(handler)
            ),
        ) as client:
            with self.assertRaises(FishSpeechClientError):
                client.wait_until_stopped(timeout=1)
        self.assertEqual(seen[0].headers["Authorization"], "Bearer test-key")

    def test_cli_passes_explicit_key_without_printing_it(self):
        with (
            patch("sys.argv", ["fish-tts-client", "--api-key", "cli-secret", "health"]),
            patch.object(fish_client, "FishSpeechHttpxClient") as client,
            patch.object(fish_client, "_print_payload"),
        ):
            fish_client.main()
        self.assertEqual(client.call_args.kwargs["api_key"], "cli-secret")


if __name__ == "__main__":
    unittest.main()
