"""OpenAI Speech and MiMo-style audio Chat adapters for the local TTS engine."""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.routing import APIRoute
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from fish_speech.models.text2semantic.errors import InvalidInputError
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest
from tools.api_auth import api_key_matches

AudioFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]
Text = Annotated[StrictStr, Field(min_length=1, max_length=4096)]
VOICE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
MAX_REFERENCE_BYTES = 25 * 1024 * 1024
MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}


class APIError(Exception):
    def __init__(self, message, param=None, *, status=400, code="invalid_value"):
        super().__init__(message)
        self.param, self.status, self.code = param, status, code

    def payload(self):
        kind = "invalid_request_error"
        if self.status == 401:
            kind = "authentication_error"
        elif self.status >= 500:
            kind = "server_error"
        return {
            "error": {
                "message": str(self),
                "type": kind,
                "param": self.param,
                "code": self.code,
            }
        }


class OpenAIRoute(APIRoute):
    """Keep OpenAI error envelopes confined to these routes."""

    def get_route_handler(self):
        handler = super().get_route_handler()

        async def wrapped(request: Request):
            request.state.openai_request_id = f"req_{uuid.uuid4().hex}"
            try:
                response = await handler(request)
            except RequestValidationError as exc:
                error = exc.errors()[0]
                param = ".".join(str(p) for p in error["loc"] if p != "body")
                failure = APIError(error["msg"], param or None, code="validation_error")
                response = JSONResponse(failure.payload(), status_code=400)
            except APIError as exc:
                response = JSONResponse(exc.payload(), status_code=exc.status)
            except InvalidInputError as exc:
                failure = APIError(str(exc), "input")
                response = JSONResponse(failure.payload(), status_code=400)
            except Exception as exc:
                logger.error(
                    "TTS API failed: request_id={} error={}",
                    request.state.openai_request_id,
                    exc,
                )
                failure = APIError(
                    "Speech generation failed.",
                    status=500,
                    code="speech_generation_failed",
                )
                response = JSONResponse(failure.payload(), status_code=500)
            response.headers["X-Request-ID"] = request.state.openai_request_id
            return response

        return wrapped


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VoiceID(StrictModel):
    id: StrictStr


class FishOptions(StrictModel):
    """Optional local generation settings, sent via SDK extra_body."""

    max_new_tokens: Annotated[int, Field(ge=1, le=4096, strict=True)] = 1024
    chunk_length: Annotated[int, Field(ge=100, le=1000, strict=True)] = 300
    iterative_prompt: bool = True
    top_p: Annotated[float, Field(ge=0.1, le=1)] = 0.8
    temperature: Annotated[float, Field(ge=0.1, le=1)] = 0.8
    repetition_penalty: Annotated[float, Field(ge=0.9, le=2)] = 1.1
    seed: int | None = None
    use_memory_cache: Literal["on", "off"] = "on"
    reference_text: Text | None = None


class SpeechRequest(StrictModel):
    model: StrictStr
    input: Text
    voice: StrictStr | VoiceID
    response_format: AudioFormat = "mp3"
    speed: Annotated[float, Field(ge=0.25, le=4, strict=True)] = 1.0
    instructions: StrictStr | None = None
    stream_format: Literal["audio", "sse"] = "audio"
    fish_speech: FishOptions = Field(default_factory=FishOptions)


class ChatMessage(StrictModel):
    role: Literal["user", "assistant", "system"]
    content: Text


class ChatAudio(StrictModel):
    format: Literal["wav", "mp3", "pcm", "pcm16"] | None = None
    voice: StrictStr = "default"
    optimize_text_preview: bool = False


class ChatRequest(StrictModel):
    model: StrictStr
    messages: Annotated[list[ChatMessage], Field(min_length=1, max_length=32)]
    audio: ChatAudio = Field(default_factory=ChatAudio)
    stream: bool = False
    modalities: list[Literal["text", "audio"]] | None = None
    fish_speech: FishOptions = Field(default_factory=FishOptions)


def _check_audio(raw: bytes, param: str) -> bytes:
    if not raw or len(raw) > MAX_REFERENCE_BYTES:
        raise APIError("Reference audio must be between 1 byte and 25 MiB.", param)
    try:
        info = sf.info(io.BytesIO(raw))
    except (RuntimeError, ValueError) as exc:
        raise APIError(
            "Reference audio must be a valid WAV or MP3 file.", param
        ) from exc
    if info.format not in {"WAV", "WAVEX", "MP3"} or info.frames == 0:
        raise APIError("Reference audio must be a non-empty WAV or MP3 file.", param)
    return raw


def load_voice(root: Path, voice: str, options: FishOptions, *, inline=False):
    param = "audio.voice" if inline else "voice"
    if inline and not VOICE_ID.fullmatch(voice):
        if len(voice) > (MAX_REFERENCE_BYTES + 2) // 3 * 4:
            raise APIError("Reference audio exceeds 25 MiB.", param)
        try:
            raw = base64.b64decode(voice, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise APIError(
                "Expected a local voice ID or Base64 WAV/MP3 audio.", param
            ) from exc
        raw = _check_audio(raw, param)
        if not options.reference_text or not options.reference_text.strip():
            raise APIError(
                "Fish Speech voice cloning requires fish_speech.reference_text.",
                "fish_speech.reference_text",
            )
        return [ServeReferenceAudio(audio=raw, text=options.reference_text)]
    if options.reference_text is not None:
        raise APIError(
            "reference_text is only used with Base64 audio.voice.",
            "fish_speech.reference_text",
        )
    if voice == "default":
        return []
    if not VOICE_ID.fullmatch(voice):
        raise APIError(
            "Voice IDs may contain ASCII letters, digits, '_' and '-'.", param
        )
    root = root.resolve()
    folder = (root / voice).resolve()
    if not folder.is_relative_to(root) or not folder.is_dir():
        raise APIError(f"Unknown local voice: {voice}.", param, code="voice_not_found")
    audios = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".wav", ".mp3"})
    if len(audios) != 1:
        raise APIError(
            "A voice directory must contain exactly one WAV/MP3 and its .lab transcript.",
            param,
        )
    audio_path, text_path = audios[0].resolve(), audios[0].with_suffix(".lab").resolve()
    if not all(
        p.is_relative_to(folder) and p.is_file() for p in (audio_path, text_path)
    ):
        raise APIError(
            "Voice files must remain inside their directory and include a .lab transcript.",
            param,
        )
    if audio_path.stat().st_size > MAX_REFERENCE_BYTES:
        raise APIError("Reference audio exceeds 25 MiB.", param)
    try:
        text = text_path.read_text(encoding="utf-8").strip()
    except UnicodeError as exc:
        raise APIError("The voice transcript must be UTF-8 text.", param) from exc
    if not text:
        raise APIError("The voice transcript cannot be empty.", param)
    return [
        ServeReferenceAudio(
            audio=_check_audio(audio_path.read_bytes(), param), text=text
        )
    ]


def check_encoding(audio_format: str, speed: float = 1.0):
    if audio_format == "aac" or speed != 1.0:
        if shutil.which("ffmpeg") is None:
            raise APIError(
                "AAC and speed adjustment require ffmpeg on the server.",
                "response_format" if audio_format == "aac" else "speed",
                code="unsupported_feature",
            )
    formats = {
        "mp3": ("MP3", None),
        "opus": ("OGG", "OPUS"),
        "wav": ("WAV", "PCM_16"),
        "flac": ("FLAC", "PCM_16"),
    }
    if audio_format in formats and not sf.check_format(*formats[audio_format]):
        raise APIError(
            f"The server's libsndfile cannot encode {audio_format}.",
            "response_format",
            code="unsupported_format",
        )


def _tempo_filters(speed: float) -> str:
    factors = []
    while speed < 0.5:
        factors.append(0.5)
        speed /= 0.5
    while speed > 2:
        factors.append(2.0)
        speed /= 2
    factors.append(speed)
    return ",".join(f"atempo={factor}" for factor in factors)


def _ffmpeg(audio, sample_rate, speed, *, aac=False):
    command = [
        shutil.which("ffmpeg") or "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
    ]
    if speed != 1.0:
        command += ["-af", _tempo_filters(speed)]
    command += ["-c:a", "aac", "-f", "adts"] if aac else ["-f", "f32le"]
    result = subprocess.run(
        command + ["pipe:1"],
        input=audio.astype("<f4").tobytes(),
        capture_output=True,
        check=True,
        timeout=120,
    )
    return result.stdout


def encode_audio(audio, sample_rate: int, audio_format: str, speed=1.0) -> bytes:
    audio = np.asarray(audio, dtype=np.float32)
    if (
        sample_rate <= 0
        or audio.ndim != 1
        or not audio.size
        or not np.isfinite(audio).all()
    ):
        raise RuntimeError("The inference engine returned invalid audio.")
    if audio_format == "aac":
        return _ffmpeg(audio, sample_rate, speed, aac=True)
    if speed != 1.0:
        audio = np.frombuffer(_ffmpeg(audio, sample_rate, speed), dtype="<f4").copy()
    target_rate = (
        24000
        if audio_format == "pcm"
        else 48000
        if audio_format == "opus"
        else sample_rate
    )
    if target_rate != sample_rate:
        audio = AF.resample(
            torch.from_numpy(audio.copy()), sample_rate, target_rate
        ).numpy()
    if audio_format == "pcm":
        # OpenAI PCM: mono, 24 kHz, signed 16-bit little-endian, no header.
        return np.clip(np.rint(audio * 32768), -32768, 32767).astype("<i2").tobytes()
    buffer = io.BytesIO()
    fmt = {"wav": "WAV", "flac": "FLAC", "mp3": "MP3", "opus": "OGG"}[audio_format]
    subtype = (
        "OPUS"
        if audio_format == "opus"
        else "PCM_16"
        if audio_format in {"wav", "flac"}
        else None
    )
    sf.write(buffer, audio, target_rate, format=fmt, subtype=subtype)
    return buffer.getvalue()


@contextmanager
def active_request(controller):
    if not controller.begin_tts_request():
        raise APIError(
            "Server shutdown is pending; new TTS requests are not accepted.",
            status=503,
            code="server_shutting_down",
        )
    try:
        yield
    finally:
        controller.end_tts_request()


def _sse(payload):
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


class DrainingStreamResponse(StreamingResponse):
    """A disconnect stops network output, while the producer drains inference."""

    def __init__(self, content, stop):
        super().__init__(
            content,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
        self.stop = stop

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.stop.set()


def chat_stream(manager, controller, tts_request, model):
    # Acquire before sending HTTP headers, so draining servers return HTTP 503.
    if not controller.begin_tts_request():
        raise APIError(
            "Server shutdown is pending; new TTS requests are not accepted.",
            status=503,
            code="server_shutting_down",
        )
    output = queue.Queue(maxsize=4)
    stop = threading.Event()
    stream_id, audio_id = f"chatcmpl-{uuid.uuid4().hex}", f"audio_{uuid.uuid4().hex}"
    created = int(time.time())

    def emit(item):
        while not stop.is_set():
            try:
                output.put(item, timeout=0.1)
                return
            except queue.Full:
                pass

    def producer():
        finish_reason, failure, saw_audio = "stop", None, False
        try:
            # Always exhaust the engine, including after a client disconnect or
            # encoding failure. This keeps the shared GPU lock until cleanup.
            for result in manager.iter_inference(tts_request):
                if result.code == "error":
                    failure = result.error or RuntimeError("Speech generation failed.")
                elif result.code == "final":
                    finish_reason = result.finish_reason or "stop"
                elif result.code == "segment" and result.audio is not None:
                    saw_audio = True
                    if not stop.is_set() and failure is None:
                        try:
                            sample_rate, audio = result.audio
                            emit(("audio", encode_audio(audio, sample_rate, "pcm")))
                        except Exception as exc:
                            failure = exc
            if failure is not None:
                raise failure
            if not saw_audio:
                raise RuntimeError("No audio generated.")
            emit(("done", finish_reason))
        except Exception as exc:
            logger.error("Chat audio stream failed: id={} error={}", stream_id, exc)
            error = (
                APIError(str(exc), "messages")
                if isinstance(exc, InvalidInputError)
                else APIError(
                    "Speech generation failed.",
                    status=500,
                    code="speech_generation_failed",
                )
            )
            emit(("error", error.payload()))
        finally:
            controller.end_tts_request()

    def chunk(delta, finish_reason=None):
        return {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            "usage": None,
        }

    async def events():
        try:
            yield _sse(chunk({"role": "assistant"}))
            while True:
                try:
                    kind, value = await asyncio.to_thread(output.get, True, 0.2)
                except queue.Empty:
                    continue
                if kind == "audio":
                    yield _sse(
                        chunk(
                            {
                                "audio": {
                                    "id": audio_id,
                                    "data": base64.b64encode(value).decode("ascii"),
                                    "expires_at": None,
                                    "transcript": None,
                                }
                            }
                        )
                    )
                elif kind == "done":
                    yield _sse(chunk({}, value))
                    break
                else:
                    yield _sse(value)
                    break
            yield b"data: [DONE]\n\n"
        finally:
            stop.set()

    try:
        threading.Thread(target=producer, name="fish-chat-audio", daemon=True).start()
    except Exception:
        controller.end_tts_request()
        raise
    return DrainingStreamResponse(events(), stop)


def create_router(manager, settings, controller):
    model_id = settings.openai_model_id
    voices_root = settings.voices_dir or settings.storage_root / "voices"
    model_info = {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "local",
    }

    def authenticate(request: Request):
        if not api_key_matches(request.headers, settings.openai_api_key):
            raise APIError(
                "Invalid or missing API key.", status=401, code="invalid_api_key"
            )

    router = APIRouter(
        prefix="/v1",
        route_class=OpenAIRoute,
        dependencies=[Depends(authenticate)],
        tags=["OpenAI-compatible TTS"],
    )

    def check_model(value):
        if value != model_id:
            raise APIError(
                f"Unknown model. This server provides '{model_id}'.",
                "model",
                code="model_not_found",
            )

    def make_request(text, voice, options, *, inline=False, streaming=False):
        if not text.strip():
            raise APIError(
                "Speech input cannot be blank.", "messages" if inline else "input"
            )
        references = load_voice(voices_root, voice, options, inline=inline)
        return ServeTTSRequest(
            text=text,
            references=references,
            format="wav",
            streaming=streaming,
            **options.model_dump(exclude={"reference_text"}),
        )

    @router.get("/models")
    def models():
        return {"object": "list", "data": [model_info]}

    @router.post("/audio/speech", response_class=Response)
    def speech(body: SpeechRequest):
        check_model(body.model)
        if body.instructions:
            raise APIError(
                "S2-Pro does not support free-form instructions; use supported tags in input.",
                "instructions",
                code="unsupported_feature",
            )
        if body.stream_format != "audio":
            raise APIError(
                "Speech supports stream_format='audio'; use chat/completions for SSE.",
                "stream_format",
                code="unsupported_feature",
            )
        check_encoding(body.response_format, body.speed)
        voice = body.voice.id if isinstance(body.voice, VoiceID) else body.voice
        tts_request = make_request(body.input, voice, body.fish_speech)
        with active_request(controller):
            sample_rate, audio = manager.infer(tts_request)
            encoded = encode_audio(audio, sample_rate, body.response_format, body.speed)
        return Response(
            encoded,
            media_type=MEDIA_TYPES[body.response_format],
            headers={
                "Content-Disposition": f'attachment; filename="speech.{body.response_format}"'
            },
        )

    @router.post("/chat/completions")
    def chat(body: ChatRequest):
        check_model(body.model)
        if body.audio.optimize_text_preview:
            raise APIError(
                "Text optimization and voice design are not supported by S2-Pro.",
                "audio.optimize_text_preview",
                code="unsupported_feature",
            )
        if body.modalities is not None and "audio" not in body.modalities:
            raise APIError(
                "This endpoint synthesizes speech and requires the audio modality.",
                "modalities",
            )
        if any(message.role != "assistant" for message in body.messages):
            raise APIError(
                "This TTS endpoint reads an assistant message verbatim; user/system instructions and voice design are unsupported.",
                "messages",
                code="unsupported_feature",
            )
        if len(body.messages) != 1:
            raise APIError(
                "Provide exactly one assistant message containing the text to speak.",
                "messages",
            )
        text = body.messages[0].content
        audio_format = body.audio.format or ("pcm" if body.stream else "wav")
        if audio_format == "pcm16":
            audio_format = "pcm"
        if body.stream and audio_format != "pcm":
            raise APIError(
                "Streaming chat audio requires pcm or pcm16 (24 kHz mono).",
                "audio.format",
            )
        check_encoding(audio_format)
        tts_request = make_request(
            text, body.audio.voice, body.fish_speech, inline=True, streaming=body.stream
        )
        if body.stream:
            return chat_stream(manager, controller, tts_request, model_id)
        with active_request(controller):
            result = manager.infer_result(tts_request)
            sample_rate, audio = result.audio
            encoded = encode_audio(audio, sample_rate, audio_format)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "usage": None,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": result.finish_reason or "stop",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "audio": {
                            "id": f"audio_{uuid.uuid4().hex}",
                            "data": base64.b64encode(encoded).decode("ascii"),
                            "expires_at": None,
                            "transcript": text,
                        },
                    },
                }
            ],
        }

    return router
