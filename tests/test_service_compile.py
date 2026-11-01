import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from fish_speech.models.text2semantic.inference import resolve_compile_backend
from tools import fish_server
from tools.fastapi_service import ModelManager, ServiceSettings, create_app


def settings(root: Path) -> ServiceSettings:
    return ServiceSettings(
        storage_root=root,
        llama_checkpoint_path=root / "s2-pro",
        decoder_checkpoint_path=root / "codec.pth",
        decoder_config_name="modded_dac_vq",
        device="cpu",
        dtype="float32",
        compile=False,
        max_seq_len=4096,
    )


class ServiceCompileTest(unittest.TestCase):
    def test_auto_backend_uses_requested_device_even_when_cuda_is_available(self):
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(resolve_compile_backend("mps"), "aot_eager")
            self.assertEqual(resolve_compile_backend("cpu"), "aot_eager")
            self.assertEqual(resolve_compile_backend("cuda:0"), "inductor")
        self.assertEqual(resolve_compile_backend("mps", "inductor"), "inductor")
        with self.assertRaises(ValueError):
            resolve_compile_backend("mps", "typo")

    def test_cli_overrides_compile_environment_in_both_directions(self):
        for enabled, args in (
            (True, ["--compile", "--compile-backend", "inductor"]),
            (False, ["--no-compile"]),
        ):
            with (
                self.subTest(enabled=enabled),
                patch("sys.argv", ["fish-tts-server", *args]),
                patch.object(
                    ServiceSettings,
                    "from_env",
                    return_value=replace(
                        settings(Path("/tmp/test")), compile=not enabled
                    ),
                ),
                patch.object(fish_server, "create_app") as create,
                patch.object(fish_server.uvicorn, "Server"),
            ):
                fish_server.main()
                actual = create.call_args.kwargs["settings"]
                self.assertEqual(actual.compile, enabled)
                self.assertEqual(
                    actual.compile_backend, "inductor" if enabled else "auto"
                )

    def test_manager_forwards_backend_to_worker_and_health(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = replace(
                settings(Path(tmp)), compile=True, compile_backend="aot_eager"
            )
            manager = ModelManager(config)
            with (
                patch("tools.fastapi_service.launch_thread_safe_queue") as launch,
                patch("tools.fastapi_service.load_decoder_model"),
                patch("tools.fastapi_service.TTSInferenceEngine"),
            ):
                manager.get_engine()
                self.assertTrue(launch.call_args.kwargs["compile"])
                self.assertEqual(
                    launch.call_args.kwargs["compile_backend"], "aot_eager"
                )
            with TestClient(create_app(config, manager)) as client:
                health = client.get("/fishspeech/health").json()
                self.assertTrue(health["compile"])
                self.assertEqual(health["compile_backend"], "aot_eager")

    def test_compile_rejects_synchronized_diagnostics_before_loading_models(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            ModelManager(
                replace(settings(Path("/tmp/test")), compile=True, perf_detail=True)
            )


if __name__ == "__main__":
    unittest.main()
