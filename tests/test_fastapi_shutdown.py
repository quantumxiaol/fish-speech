import tempfile
import unittest
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from tools.fastapi_service import (
    ADMIN_SHUTDOWN_HEADER,
    ADMIN_SHUTDOWN_VALUE,
    ServerShutdownController,
    ServiceSettings,
    create_app,
)
from tools.fish_httpx_client import FishSpeechHttpxClient


class FakeModelManager:
    loaded = False
    device = "mps"
    dtype = "float16"

    def infer(self, request):
        raise AssertionError("Shutdown tests must not run inference.")


def make_settings(root: Path) -> ServiceSettings:
    return ServiceSettings(
        storage_root=root,
        llama_checkpoint_path=root / "llama",
        decoder_checkpoint_path=root / "codec.pth",
        decoder_config_name="modded_dac_vq",
        device="mps",
        dtype="float16",
        compile=False,
        max_seq_len=4096,
    )


class ServerShutdownControllerTest(unittest.TestCase):
    def test_shutdown_drains_all_active_tts_requests_and_runs_once(self) -> None:
        callbacks: list[str] = []
        controller = ServerShutdownController(
            shutdown_callback=lambda: callbacks.append("shutdown"),
        )

        self.assertTrue(controller.begin_tts_request())
        self.assertTrue(controller.begin_tts_request())
        self.assertTrue(controller.request_shutdown("admin_request"))
        self.assertFalse(controller.begin_tts_request())
        self.assertFalse(controller.execute_shutdown())

        controller.end_tts_request()
        self.assertEqual(callbacks, [])
        controller.end_tts_request()
        self.assertEqual(callbacks, ["shutdown"])
        self.assertFalse(controller.execute_shutdown())
        self.assertFalse(controller.request_shutdown("admin_request"))


class FastAPIShutdownEndpointTest(unittest.TestCase):
    def _make_app(self, root: Path, callbacks: list[str], configured: bool = True):
        callback = (lambda: callbacks.append("shutdown")) if configured else None
        return create_app(
            settings=make_settings(root),
            model_manager=FakeModelManager(),
            shutdown_callback=callback,
        )

    def test_loopback_request_is_accepted_and_callback_runs_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            callbacks: list[str] = []
            app = self._make_app(Path(tmp), callbacks)
            headers = {ADMIN_SHUTDOWN_HEADER: ADMIN_SHUTDOWN_VALUE}
            with TestClient(app, client=("127.0.0.1", 50000)) as client:
                first = client.post("/fishspeech/admin/shutdown", headers=headers)
                second = client.post("/fishspeech/admin/shutdown", headers=headers)

            self.assertEqual(first.status_code, 202)
            self.assertEqual(first.json()["status"], "accepted")
            self.assertEqual(second.status_code, 202)
            self.assertEqual(second.json()["status"], "already_pending")
            self.assertEqual(callbacks, ["shutdown"])

    def test_health_metrics_fall_back_cleanly_when_accelerator_is_unavailable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = self._make_app(Path(tmp), [])
            with TestClient(app, client=("127.0.0.1", 50000)) as client:
                response = client.get("/fishspeech/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["mps_profile"])
        self.assertIsNone(payload["mps_tensor_gib"])
        self.assertIsNone(payload["cuda_allocated_gib"])

    def test_ipv6_loopback_and_ipv4_mapped_loopback_are_accepted(self) -> None:
        headers = {ADMIN_SHUTDOWN_HEADER: ADMIN_SHUTDOWN_VALUE}
        for host in ("::1", "::ffff:127.0.0.1"):
            with self.subTest(host=host), tempfile.TemporaryDirectory() as tmp:
                callbacks: list[str] = []
                app = self._make_app(Path(tmp), callbacks)
                with TestClient(app, client=(host, 50000)) as client:
                    response = client.post(
                        "/fishspeech/admin/shutdown", headers=headers
                    )
                self.assertEqual(response.status_code, 202)
                self.assertEqual(callbacks, ["shutdown"])

    def test_remote_client_cannot_spoof_loopback_with_forwarded_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            callbacks: list[str] = []
            app = self._make_app(Path(tmp), callbacks)
            headers = {
                ADMIN_SHUTDOWN_HEADER: ADMIN_SHUTDOWN_VALUE,
                "X-Forwarded-For": "127.0.0.1",
            }
            with TestClient(app, client=("192.168.1.50", 50000)) as client:
                response = client.post("/fishspeech/admin/shutdown", headers=headers)

            self.assertEqual(response.status_code, 403)
            self.assertEqual(callbacks, [])

    def test_confirmation_header_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            callbacks: list[str] = []
            app = self._make_app(Path(tmp), callbacks)
            with TestClient(app, client=("127.0.0.1", 50000)) as client:
                response = client.post("/fishspeech/admin/shutdown")

            self.assertEqual(response.status_code, 403)
            self.assertEqual(callbacks, [])

    def test_missing_server_callback_returns_service_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = self._make_app(Path(tmp), [], configured=False)
            headers = {ADMIN_SHUTDOWN_HEADER: ADMIN_SHUTDOWN_VALUE}
            with TestClient(app, client=("127.0.0.1", 50000)) as client:
                response = client.post("/fishspeech/admin/shutdown", headers=headers)

            self.assertEqual(response.status_code, 503)

    def test_pending_shutdown_rejects_new_tts_before_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            callbacks: list[str] = []
            app = self._make_app(Path(tmp), callbacks)
            controller = app.state.shutdown_controller
            self.assertTrue(controller.request_shutdown("test"))

            with TestClient(app, client=("127.0.0.1", 50000)) as client:
                response = client.post(
                    "/fishspeech/tts/voice_clone",
                    data={"text": "test", "ref_text": "reference"},
                    files={"ref_audio": ("ref.wav", b"not-audio", "audio/wav")},
                )

            self.assertEqual(response.status_code, 503)
            self.assertEqual(callbacks, [])


class ShutdownHttpClientTest(unittest.TestCase):
    def test_client_sends_confirmation_header_and_can_wait_for_exit(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.url.path == "/fishspeech/admin/shutdown":
                return httpx.Response(
                    202,
                    json={"status": "accepted", "reason": "admin_request"},
                )
            raise httpx.ConnectError("server stopped", request=request)

        http_client = httpx.Client(
            base_url="http://127.0.0.1:8002",
            transport=httpx.MockTransport(handler),
        )
        client = FishSpeechHttpxClient(client=http_client)
        try:
            payload = client.shutdown(wait=True, wait_timeout=1)
        finally:
            client.close()

        self.assertTrue(payload["server_stopped"])
        self.assertEqual(requests[0].url.path, "/fishspeech/admin/shutdown")
        self.assertEqual(
            requests[0].headers[ADMIN_SHUTDOWN_HEADER], ADMIN_SHUTDOWN_VALUE
        )
        self.assertEqual(requests[0].headers["Connection"], "close")

    def test_wait_treats_non_health_response_as_server_stopped(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/fishspeech/admin/shutdown":
                return httpx.Response(
                    202,
                    json={"status": "accepted", "reason": "admin_request"},
                )
            return httpx.Response(502, text="listener is gone")

        http_client = httpx.Client(
            base_url="http://127.0.0.1:8002",
            transport=httpx.MockTransport(handler),
        )
        client = FishSpeechHttpxClient(client=http_client)
        try:
            payload = client.shutdown(wait=True, wait_timeout=1)
        finally:
            client.close()

        self.assertTrue(payload["server_stopped"])


if __name__ == "__main__":
    unittest.main()
