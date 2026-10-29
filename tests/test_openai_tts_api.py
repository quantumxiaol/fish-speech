import asyncio
import base64
import importlib.util
import io
import json
import shutil
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from tools.fastapi_service import ServerShutdownController, ServiceSettings, create_app
from fish_speech.inference_engine.utils import InferenceResult
from fish_speech.utils.schema import ServeTTSRequest
from tools.openai_tts_api import chat_stream, encode_audio


def settings(root):
    return ServiceSettings(
        root,
        root / "s2-pro",
        root / "codec.pth",
        "modded_dac_vq",
        "cpu",
        "float32",
        False,
        4096,
    )


class FakeManager:
    loaded, device, dtype = True, "cpu", "float32"

    def __init__(self):
        self.requests = []
        self.audio = (0.2 * np.sin(2 * np.pi * 440 * np.arange(22050) / 44100)).astype(
            np.float32
        )
        self.finish_reason = "stop"
        self.error = None

    def infer_result(self, request):
        self.requests.append(request)
        if self.error:
            raise self.error
        return InferenceResult("final", (44100, self.audio), None, self.finish_reason)

    def infer(self, request):
        return self.infer_result(request).audio

    def iter_inference(self, request):
        self.requests.append(request)
        if self.error:
            yield InferenceResult("error", None, self.error)
            return
        yield InferenceResult("header", (44100, np.zeros(44, dtype=np.uint8)), None)
        for segment in np.split(self.audio, 2):
            yield InferenceResult("segment", (44100, segment), None)
        yield InferenceResult("final", (44100, self.audio), None, self.finish_reason)


class OpenAITTSTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manager = FakeManager()
        self.app = create_app(settings(self.root), self.manager)
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)

    def speech(self, **changes):
        return self.client.post(
            "/v1/audio/speech",
            json={
                "model": "s2-pro",
                "voice": "default",
                "input": "你好。",
                "response_format": "wav",
                **changes,
            },
        )

    def chat(self, **changes):
        return self.client.post(
            "/v1/chat/completions",
            json={
                "model": "s2-pro",
                "messages": [{"role": "assistant", "content": "你好。"}],
                **changes,
            },
        )

    def voice(self, name="my_voice"):
        folder = self.root / "voices" / name
        folder.mkdir(parents=True)
        sf.write(folder / "ref.wav", self.manager.audio, 44100)
        (folder / "ref.lab").write_text("参考音频的文字。", encoding="utf-8")
        return folder

    def test_model_discovery_and_wav_response(self):
        self.assertEqual(
            self.client.get("/v1/models").json()["data"][0]["id"], "s2-pro"
        )
        response = self.speech()
        self.assertEqual(response.status_code, 200, response.text[:200])
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertTrue(response.headers["x-request-id"].startswith("req_"))
        audio, rate = sf.read(io.BytesIO(response.content))
        self.assertEqual(rate, 44100)
        self.assertEqual(len(audio), len(self.manager.audio))
        self.assertEqual(self.manager.requests[0].text, "你好。")

    def test_default_format_is_mp3(self):
        response = self.client.post(
            "/v1/audio/speech",
            json={
                "model": "s2-pro",
                "voice": "default",
                "input": "hello",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/mpeg")
        self.assertEqual(sf.info(io.BytesIO(response.content)).format, "MP3")

    def test_formats_and_pcm_wire_format(self):
        for fmt, expected in (("wav", "WAV"), ("flac", "FLAC"), ("opus", "OGG")):
            with self.subTest(format=fmt):
                response = self.speech(response_format=fmt)
                self.assertEqual(response.status_code, 200)
                info = sf.info(io.BytesIO(response.content))
                self.assertEqual(info.format, expected)
                self.assertEqual(info.samplerate, 48000 if fmt == "opus" else 44100)
        pcm = self.speech(response_format="pcm")
        self.assertEqual(pcm.headers["content-type"], "audio/pcm")
        self.assertEqual(len(pcm.content), 24000)  # 0.5 s, 24 kHz, int16 mono
        self.assertLess(np.abs(np.frombuffer(pcm.content, dtype="<i2")).max(), 7000)
        self.assertEqual(
            encode_audio(np.array([-1.0, 0.0, 1.0]), 24000, "pcm"),
            b"\x00\x80\x00\x00\xff\x7f",
        )

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is needed for AAC and tempo")
    def test_aac_and_pitch_preserving_speed(self):
        aac = self.speech(response_format="aac")
        self.assertEqual(aac.status_code, 200)
        self.assertEqual(aac.headers["content-type"], "audio/aac")
        self.assertEqual(aac.content[0], 255)
        faster = self.speech(speed=2.0)
        audio, rate = sf.read(io.BytesIO(faster.content))
        self.assertAlmostEqual(len(audio) / rate, 0.25, delta=0.06)
        peak = np.argmax(np.abs(np.fft.rfft(audio))) * rate / len(audio)
        self.assertAlmostEqual(peak, 440, delta=15)

    def test_voice_registry_and_generation_options(self):
        self.voice()
        response = self.speech(
            voice={"id": "my_voice"},
            fish_speech={
                "seed": 7,
                "repetition_penalty": 1.25,
                "iterative_prompt": False,
            },
        )
        self.assertEqual(response.status_code, 200, response.text[:100])
        request = self.manager.requests[-1]
        self.assertEqual(request.references[0].text, "参考音频的文字。")
        self.assertEqual(request.use_memory_cache, "on")
        self.assertEqual(request.seed, 7)
        self.assertEqual(request.repetition_penalty, 1.25)
        self.assertFalse(request.iterative_prompt)

    def test_custom_model_and_voices_directory(self):
        self.voice()
        app = create_app(
            replace(
                settings(self.root),
                openai_model_id="fish-local",
                voices_dir=self.root / "voices",
            ),
            self.manager,
        )
        with TestClient(app) as client:
            self.assertEqual(
                client.get("/v1/models").json()["data"][0]["id"], "fish-local"
            )
            result = client.post(
                "/v1/audio/speech",
                json={
                    "model": "fish-local",
                    "voice": "my_voice",
                    "input": "Hello",
                },
            )
            self.assertEqual(result.status_code, 200)

    def test_bad_requests_fail_before_inference(self):
        cases = [
            {"model": "tts-1"},
            {"voice": "missing"},
            {"voice": "../secret"},
            {"input": " "},
            {"input": "x" * 4097},
            {"speed": 0.1},
            {"instructions": "A new voice"},
            {"stream_format": "sse"},
            {"stream": True},
            {"fish_speech": {"top_p": 0.01}},
        ]
        for changes in cases:
            with self.subTest(changes=changes):
                response = self.speech(**changes)
                self.assertEqual(response.status_code, 400)
                self.assertIn("param", response.json()["error"])
        self.assertEqual(self.manager.requests, [])
        bad_json = self.client.post(
            "/v1/audio/speech",
            content="{",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(bad_json.status_code, 400)
        self.assertIn("error", bad_json.json())

    def test_reference_directory_and_file_symlinks_cannot_escape(self):
        outside = self.root / "outside"
        outside.mkdir()
        (self.root / "voices").mkdir()
        (self.root / "voices" / "escape").symlink_to(outside, target_is_directory=True)
        self.assertEqual(self.speech(voice="escape").status_code, 400)
        folder = self.voice()
        (folder / "ref.lab").unlink()
        (outside / "text.lab").write_text("private", encoding="utf-8")
        (folder / "ref.lab").symlink_to(outside / "text.lab")
        self.assertEqual(self.speech(voice="my_voice").status_code, 400)
        self.assertEqual(self.manager.requests, [])

    def test_bearer_and_api_key_auth_only_apply_to_new_routes(self):
        app = create_app(
            replace(settings(self.root), openai_api_key="test-key"), self.manager
        )
        with TestClient(app) as client:
            self.assertEqual(client.get("/v1/models").status_code, 401)
            self.assertEqual(client.get("/fishspeech/health").status_code, 200)
            for headers in (
                {"Authorization": "Bearer test-key"},
                {"api-key": "test-key"},
            ):
                self.assertEqual(
                    client.get("/v1/models", headers=headers).status_code, 200
                )
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "s2-pro",
                    "messages": [{"role": "assistant", "content": "hi"}],
                },
            )
            self.assertEqual(response.status_code, 401)
            self.assertEqual(response.json()["error"]["code"], "invalid_api_key")
        self.assertEqual(self.manager.requests, [])

    def test_chat_base64_response_and_finish_reason(self):
        self.manager.finish_reason = "length"
        response = self.chat()
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["object"], "chat.completion")
        self.assertIsNone(payload["usage"])
        choice = payload["choices"][0]
        self.assertEqual(choice["finish_reason"], "length")
        self.assertEqual(choice["message"]["audio"]["transcript"], "你好。")
        raw = base64.b64decode(choice["message"]["audio"]["data"])
        self.assertEqual(sf.info(io.BytesIO(raw)).format, "WAV")

    def test_chat_inline_voice_requires_and_uses_transcript(self):
        buffer = io.BytesIO()
        sf.write(buffer, self.manager.audio, 44100, format="WAV")
        voice = base64.b64encode(buffer.getvalue()).decode()
        self.assertEqual(self.chat(audio={"voice": voice}).status_code, 400)
        response = self.chat(
            audio={"voice": voice, "format": "mp3"},
            fish_speech={"reference_text": "This is the reference."},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            self.manager.requests[0].references[0].audio, buffer.getvalue()
        )
        self.assertEqual(
            self.manager.requests[0].references[0].text, "This is the reference."
        )

    def test_chat_rejects_unsupported_features(self):
        for changes in (
            {"audio": {"optimize_text_preview": True}},
            {"messages": [{"role": "user", "content": "design a voice"}]},
            {"messages": [{"role": "assistant", "content": "a"}] * 2},
            {"stream": True, "audio": {"format": "wav"}},
            {"modalities": ["text"]},
            {"model": "mimo-v2.5-tts-voicedesign"},
        ):
            with self.subTest(changes=changes):
                self.assertEqual(self.chat(**changes).status_code, 400)
        self.assertFalse(self.manager.requests)

    def test_streaming_chat_pcm_deltas_and_done(self):
        self.manager.finish_reason = "length"
        response = self.chat(stream=True, audio={"format": "pcm16"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        lines = [
            line[6:] for line in response.text.splitlines() if line.startswith("data: ")
        ]
        self.assertEqual(lines[-1], "[DONE]")
        chunks = [json.loads(line) for line in lines[:-1]]
        self.assertEqual(len({chunk["id"] for chunk in chunks}), 1)
        self.assertEqual(len({chunk["created"] for chunk in chunks}), 1)
        self.assertEqual(chunks[0]["choices"][0]["delta"], {"role": "assistant"})
        deltas = [
            c["choices"][0]["delta"]["audio"]
            for c in chunks
            if "audio" in c["choices"][0]["delta"]
        ]
        self.assertEqual(len(deltas), 2)
        self.assertEqual(sum(len(base64.b64decode(d["data"])) for d in deltas), 24000)
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "length")
        self.assertTrue(self.manager.requests[-1].streaming)

    def test_errors_and_shutdown_rejection_are_openai_shaped(self):
        self.manager.error = RuntimeError("internal private detail")
        response = self.speech()
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("private", response.text)
        streamed = self.chat(stream=True)
        self.assertIn('"error"', streamed.text)
        self.assertNotIn('"finish_reason": "stop"', streamed.text)
        callbacks = []
        controller = self.app.state.shutdown_controller
        controller.set_shutdown_callback(lambda: callbacks.append(True))
        controller.request_shutdown("test")
        self.assertTrue(controller.execute_shutdown())
        self.assertEqual(callbacks, [True])
        self.assertEqual(self.speech().status_code, 503)
        self.assertEqual(self.chat(stream=True).status_code, 503)

    @unittest.skipUnless(importlib.util.find_spec("openai"), "OpenAI SDK not installed")
    def test_real_openai_sdk_speech_chat_and_stream(self):
        from openai import OpenAI

        sdk = OpenAI(
            base_url="http://testserver/v1",
            api_key="local",
            http_client=self.client,
            max_retries=0,
        )
        self.assertEqual(sdk.models.list().data[0].id, "s2-pro")
        path = self.root / "sdk.wav"
        with sdk.audio.speech.with_streaming_response.create(
            model="s2-pro", voice="default", input="hello", response_format="wav"
        ) as response:
            response.stream_to_file(path)
        self.assertEqual(sf.info(path).format, "WAV")
        args = dict(
            model="s2-pro", messages=[{"role": "assistant", "content": "hello"}]
        )
        completion = sdk.chat.completions.create(
            **args, audio={"voice": "default", "format": "wav"}
        )
        self.assertTrue(completion.choices[0].message.audio.data)
        chunks = list(
            sdk.chat.completions.create(
                **args, stream=True, audio={"voice": "default", "format": "pcm16"}
            )
        )
        self.assertEqual(chunks[-1].choices[0].finish_reason, "stop")


class StreamDrainTest(unittest.IsolatedAsyncioTestCase):
    async def test_disconnect_drains_inference_before_shutdown(self):
        release, drained = threading.Event(), threading.Event()
        callbacks = []
        controller = ServerShutdownController(lambda: callbacks.append("shutdown"))

        class Manager:
            def iter_inference(self, request):
                try:
                    yield InferenceResult(
                        "segment", (24000, np.zeros(240, np.float32)), None
                    )
                    if not release.wait(timeout=5):
                        raise RuntimeError("test timed out")
                    # More chunks than the queue capacity: disconnect must drain
                    # without waiting on a consumer that is no longer present.
                    for _ in range(8):
                        yield InferenceResult(
                            "segment", (24000, np.zeros(240, np.float32)), None
                        )
                    yield InferenceResult(
                        "final", (24000, np.zeros(240, np.float32)), None, "stop"
                    )
                finally:
                    drained.set()

        response = chat_stream(
            Manager(), controller, ServeTTSRequest(text="hi", streaming=True), "s2-pro"
        )
        iterator = response.body_iterator
        try:
            await asyncio.wait_for(anext(iterator), 2)  # role arrives immediately
            first_audio = await asyncio.wait_for(anext(iterator), 2)
            self.assertIn(b'"audio"', first_audio)
            self.assertFalse(drained.is_set())
            controller.request_shutdown("test")
            self.assertFalse(controller.execute_shutdown())
            self.assertFalse(controller.begin_tts_request())
            await iterator.aclose()  # simulated client disconnect
            self.assertEqual(callbacks, [])
        finally:
            response.stop.set()
            release.set()
        self.assertTrue(await asyncio.to_thread(drained.wait, 2))
        deadline = asyncio.get_running_loop().time() + 2
        while not callbacks and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)
        self.assertEqual(callbacks, ["shutdown"])


if __name__ == "__main__":
    unittest.main()
