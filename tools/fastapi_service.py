from __future__ import annotations

import argparse
import ipaddress
import os
import platform
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Callable, Optional

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")

if platform.system() == "Darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.55")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.75")

_MPL_CACHE = Path(tempfile.gettempdir()) / "fish_speech_matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
_XDG_CACHE = Path(tempfile.gettempdir()) / "fish_speech_cache"
_XDG_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE))

import numpy as np
import pyrootutils
import soundfile as sf
import torch
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from fish_speech.env_config import (  # noqa: E402
    checkpoint_path,
    decoder_checkpoint_path,
    default_device,
    load_project_env,
)
from fish_speech.inference_engine import TTSInferenceEngine  # noqa: E402
from fish_speech.models.dac.inference import load_model as load_decoder_model  # noqa: E402
from fish_speech.models.text2semantic.inference import (  # noqa: E402
    launch_thread_safe_queue,
)
from fish_speech.utils.file import audio_to_bytes  # noqa: E402
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest  # noqa: E402

API_PREFIX = "/fishspeech"
ADMIN_SHUTDOWN_HEADER = "X-Fish-Speech-Admin"
ADMIN_SHUTDOWN_VALUE = "shutdown"
DEFAULT_STORAGE_ROOT = Path("storage") / "fish_speech_service"
SUPPORTED_FORMATS = {"wav", "flac", "mp3", "opus", "pcm"}


def _safe_stem(value: str, fallback: str) -> str:
    stem = Path(value or "").stem
    cleaned = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem
    ).strip("._")
    return cleaned or fallback


def _safe_audio_filename(
    value: Optional[str], fallback_prefix: str = "audio", audio_format: str = "wav"
) -> str:
    ext = audio_format.lower()
    if value:
        return f"{_safe_stem(Path(value).name, fallback_prefix)}.{ext}"
    return f"{fallback_prefix}_{uuid.uuid4().hex}.{ext}"


def _resolve_torch_dtype(name: Optional[str], device_name: str) -> torch.dtype:
    if name:
        normalized = name.strip().lower()
        mapping = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        if normalized not in mapping:
            raise ValueError(f"Unsupported dtype: {name}")
        return mapping[normalized]

    if device_name.startswith("cuda"):
        return torch.bfloat16
    return torch.float32


def _choose_device(device_override: Optional[str]) -> str:
    if device_override:
        return device_override
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def _validate_audio_format(audio_format: str) -> str:
    normalized = audio_format.strip().lower()
    if normalized not in SUPPORTED_FORMATS:
        allowed = ", ".join(sorted(SUPPORTED_FORMATS))
        raise HTTPException(status_code=422, detail=f"Unsupported format: {audio_format}. Allowed: {allowed}")
    return normalized


def _is_loopback_request(request: Request) -> bool:
    if request.client is None:
        return False
    try:
        address = ipaddress.ip_address(request.client.host)
    except ValueError:
        return False
    if address.is_loopback:
        return True
    mapped = getattr(address, "ipv4_mapped", None)
    return mapped is not None and mapped.is_loopback


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int, audio_format: str) -> None:
    audio_format = _validate_audio_format(audio_format)
    path.parent.mkdir(parents=True, exist_ok=True)

    if audio_format == "pcm":
        pcm = np.clip(audio, -1.0, 1.0)
        path.write_bytes((pcm * 32767.0).astype(np.int16).tobytes())
        return

    sf_format = {
        "wav": "WAV",
        "flac": "FLAC",
        "mp3": "MP3",
        "opus": "OGG",
    }[audio_format]
    kwargs = {"format": sf_format}
    if audio_format == "opus":
        kwargs["subtype"] = "OPUS"
    sf.write(path, audio, sample_rate, **kwargs)


def _content_type(audio_format: str) -> str:
    return {
        "wav": "audio/wav",
        "flac": "audio/flac",
        "mp3": "audio/mpeg",
        "opus": "audio/ogg",
        "pcm": "application/octet-stream",
    }.get(audio_format, "application/octet-stream")


def _optional_int(value: Optional[str]) -> Optional[int]:
    if value is None or value.strip() == "":
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


@dataclass
class ServiceSettings:
    storage_root: Path
    llama_checkpoint_path: Path
    decoder_checkpoint_path: Path
    decoder_config_name: str
    device: Optional[str]
    dtype: Optional[str]
    compile: bool
    max_seq_len: Optional[int]

    @classmethod
    def from_env(cls) -> "ServiceSettings":
        load_project_env()
        return cls(
            storage_root=Path(
                os.getenv("FISH_TTS_STORAGE_ROOT", str(DEFAULT_STORAGE_ROOT))
            )
            .expanduser()
            .resolve(),
            llama_checkpoint_path=checkpoint_path(),
            decoder_checkpoint_path=decoder_checkpoint_path(),
            decoder_config_name=os.getenv(
                "FISH_TTS_DECODER_CONFIG_NAME", "modded_dac_vq"
            ),
            device=default_device(fallback="") or None,
            dtype=os.getenv("FISH_TTS_DTYPE") or None,
            compile=os.getenv("FISH_TTS_COMPILE", "").lower() in {"1", "true", "yes"},
            max_seq_len=_optional_int(os.getenv("FISH_TTS_MAX_SEQ_LEN", "4096")),
        )


class StoredFile(BaseModel):
    filename: str
    path: str
    url: str


class VoiceCloneResponse(BaseModel):
    status: str
    request_id: str
    audio: StoredFile
    prompt_audio: StoredFile
    prompt_text: StoredFile
    synthesis_text: StoredFile
    sample_rate: int
    format: str


class VoiceCloneBatchResponse(BaseModel):
    status: str
    request_id: str
    audio_paths: list[StoredFile]
    prompt_audio: StoredFile
    prompt_text: StoredFile
    text_file: StoredFile
    sample_rate: int
    format: str


class HealthResponse(BaseModel):
    status: str
    storage_root: str
    loaded: bool
    device: str
    dtype: str
    max_seq_len: Optional[int]
    llama_checkpoint_path: str
    decoder_checkpoint_path: str
    shutdown_pending: bool


class ShutdownResponse(BaseModel):
    status: str
    reason: str


class FileStore:
    def __init__(self, root: Path):
        self.root = root
        self.upload_root = root / "uploads"
        self.output_root = root / "outputs"
        self.upload_root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def request_upload_dir(self, request_id: str) -> Path:
        path = self.upload_root / request_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def request_output_dir(self, request_id: str) -> Path:
        path = self.output_root / request_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def save_upload(
        self, upload: UploadFile, request_id: str, fallback_prefix: str
    ) -> Path:
        suffix = Path(upload.filename or "").suffix.lower() or ".bin"
        filename = (
            f"{_safe_stem(upload.filename or fallback_prefix, fallback_prefix)}_"
            f"{uuid.uuid4().hex[:8]}{suffix}"
        )
        target = self.request_upload_dir(request_id) / filename
        with target.open("wb") as handle:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        await upload.close()
        return target.resolve()

    def save_text(self, text: str, request_id: str, filename: str) -> Path:
        target = self.request_upload_dir(request_id) / filename
        target.write_text(text, encoding="utf-8")
        return target.resolve()

    def build_output_path(
        self, request_id: str, output_name: Optional[str], prefix: str, audio_format: str
    ) -> Path:
        filename = _safe_audio_filename(output_name, prefix, audio_format)
        return (self.request_output_dir(request_id) / filename).resolve()

    def resolve_public_file(self, category: str, request_id: str, filename: str) -> Path:
        if category == "uploads":
            root = self.upload_root
        elif category == "outputs":
            root = self.output_root
        else:
            raise HTTPException(status_code=404, detail=f"Unknown file category: {category}")

        candidate = (root / request_id / filename).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="Invalid file path.") from exc
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail="File not found.")
        return candidate


class ModelManager:
    def __init__(self, settings: ServiceSettings):
        self.settings = settings
        self.device = _choose_device(settings.device)
        self.dtype = _resolve_torch_dtype(settings.dtype, self.device)
        self._engine: Optional[TTSInferenceEngine] = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._engine is not None

    def get_engine(self) -> TTSInferenceEngine:
        if self._engine is not None:
            return self._engine

        with self._load_lock:
            if self._engine is not None:
                return self._engine

            logger.info(
                "Loading Fish Speech models: llama={} decoder={} device={} dtype={} max_seq_len={}",
                self.settings.llama_checkpoint_path,
                self.settings.decoder_checkpoint_path,
                self.device,
                self.dtype,
                self.settings.max_seq_len,
            )
            llama_queue = launch_thread_safe_queue(
                checkpoint_path=self.settings.llama_checkpoint_path,
                device=self.device,
                precision=self.dtype,
                compile=self.settings.compile,
                max_seq_len=self.settings.max_seq_len,
            )
            decoder_model = load_decoder_model(
                config_name=self.settings.decoder_config_name,
                checkpoint_path=self.settings.decoder_checkpoint_path,
                device=self.device,
            )
            self._engine = TTSInferenceEngine(
                llama_queue=llama_queue,
                decoder_model=decoder_model,
                precision=self.dtype,
                compile=self.settings.compile,
            )
            return self._engine

    def infer(self, request: ServeTTSRequest) -> tuple[int, np.ndarray]:
        engine = self.get_engine()
        with self._inference_lock:
            for result in engine.inference(request):
                if result.code == "error":
                    raise RuntimeError(str(result.error))
                if result.code == "final" and isinstance(result.audio, tuple):
                    sample_rate, audio = result.audio
                    return int(sample_rate), audio
        raise RuntimeError("No audio generated, please check the input text.")


class ServerShutdownController:
    """Coordinate graceful shutdown without touching model code."""

    def __init__(
        self,
        shutdown_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        self._shutdown_callback = shutdown_callback
        self._lock = threading.Lock()
        self._active_tts_requests = 0
        self._shutdown_pending = False
        self._shutdown_callback_called = False
        self._shutdown_reason: Optional[str] = None

    @property
    def configured(self) -> bool:
        with self._lock:
            return self._shutdown_callback is not None

    @property
    def shutdown_pending(self) -> bool:
        with self._lock:
            return self._shutdown_pending

    @property
    def shutdown_reason(self) -> Optional[str]:
        with self._lock:
            return self._shutdown_reason

    def set_shutdown_callback(self, callback: Callable[[], None]) -> None:
        with self._lock:
            if self._shutdown_pending:
                raise RuntimeError(
                    "Cannot replace shutdown callback after shutdown is pending."
                )
            self._shutdown_callback = callback

    def begin_tts_request(self) -> bool:
        with self._lock:
            if self._shutdown_pending:
                return False
            self._active_tts_requests += 1
            return True

    def end_tts_request(self) -> None:
        with self._lock:
            if self._active_tts_requests <= 0:
                raise RuntimeError("TTS request activity counter is unbalanced.")
            self._active_tts_requests -= 1
            should_execute = (
                self._active_tts_requests == 0 and self._shutdown_pending
            )

        if should_execute:
            self.execute_shutdown()

    def request_shutdown(self, reason: str) -> bool:
        with self._lock:
            if self._shutdown_pending:
                return False
            self._shutdown_pending = True
            self._shutdown_reason = reason
            return True

    def execute_shutdown(self) -> bool:
        with self._lock:
            if (
                not self._shutdown_pending
                or self._active_tts_requests > 0
                or self._shutdown_callback_called
            ):
                return False
            callback = self._shutdown_callback
            if callback is None:
                logger.error(
                    "Shutdown requested, but no server shutdown callback is configured."
                )
                return False
            self._shutdown_callback_called = True
            reason = self._shutdown_reason

        logger.warning("Graceful server shutdown requested: reason={}", reason)
        callback()
        return True


def _stored_file_from_path(request: Request, store: FileStore, path: Path) -> StoredFile:
    relative = path.resolve().relative_to(store.root.resolve())
    parts = relative.parts
    if len(parts) < 3:
        raise ValueError(f"Unexpected stored path layout: {path}")
    url = str(
        request.url_for(
            "download_file",
            category=parts[0],
            request_id=parts[1],
            filename="/".join(parts[2:]),
        )
    )
    return StoredFile(filename=path.name, path=str(path.resolve()), url=url)


async def _resolve_required_text(
    store: FileStore,
    request_id: str,
    text: Optional[str],
    text_file: Optional[UploadFile],
    default_filename: str,
) -> tuple[str, Path]:
    if text_file is not None:
        saved_path = await store.save_upload(
            text_file, request_id, fallback_prefix=Path(default_filename).stem
        )
        normalized = saved_path.read_text(encoding="utf-8").strip()
        if not normalized:
            raise HTTPException(status_code=422, detail=f"{default_filename} is empty.")
        return normalized, saved_path

    if text is not None and text.strip():
        normalized = text.strip()
        return normalized, store.save_text(normalized, request_id, default_filename)

    raise HTTPException(status_code=422, detail="Either text or text_file must be provided.")


async def _resolve_required_text_lines(
    store: FileStore,
    request_id: str,
    text_file: Optional[UploadFile],
    text: Optional[list[str]],
    default_filename: str,
) -> tuple[Path, list[tuple[int, str]]]:
    if text_file is not None and text:
        raise HTTPException(status_code=422, detail="Provide either text_file or text, not both.")

    if text_file is not None:
        saved_path = await store.save_upload(
            text_file, request_id, fallback_prefix=Path(default_filename).stem
        )
        lines: list[tuple[int, str]] = []
        with saved_path.open("r", encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                normalized = raw_line.strip()
                if normalized:
                    lines.append((line_no, normalized))
        if not lines:
            raise HTTPException(
                status_code=422,
                detail=f"{default_filename} does not contain any non-empty lines.",
            )
        return saved_path, lines

    lines = []
    raw_lines = text or []
    for line_no, raw_line in enumerate(raw_lines, start=1):
        normalized = raw_line.strip()
        if normalized:
            lines.append((line_no, normalized))
    if not lines:
        raise HTTPException(
            status_code=422,
            detail="Either text_file or at least one non-empty text value must be provided.",
        )

    content = "\n".join(raw_lines)
    if not content.endswith("\n"):
        content += "\n"
    return store.save_text(content, request_id, default_filename), lines


def _build_tts_request(
    *,
    text: str,
    reference_audio: bytes,
    reference_text: str,
    audio_format: str,
    max_new_tokens: int,
    chunk_length: int,
    iterative_prompt: bool,
    top_p: float,
    repetition_penalty: float,
    temperature: float,
    seed: Optional[int],
    use_memory_cache: str,
) -> ServeTTSRequest:
    return ServeTTSRequest(
        text=text,
        references=[ServeReferenceAudio(audio=reference_audio, text=reference_text)],
        reference_id=None,
        format=audio_format,
        max_new_tokens=max_new_tokens,
        chunk_length=chunk_length,
        iterative_prompt=iterative_prompt,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        seed=seed,
        use_memory_cache=use_memory_cache,
    )


def create_app(
    settings: Optional[ServiceSettings] = None,
    model_manager: Optional[ModelManager] = None,
    shutdown_callback: Optional[Callable[[], None]] = None,
) -> FastAPI:
    app_settings = settings or ServiceSettings.from_env()
    manager = model_manager or ModelManager(app_settings)
    store = FileStore(app_settings.storage_root)
    shutdown_controller = ServerShutdownController(
        shutdown_callback=shutdown_callback,
    )

    app = FastAPI(title="Fish Speech FastAPI Service")
    app.state.settings = app_settings
    app.state.model_manager = manager
    app.state.file_store = store
    app.state.shutdown_controller = shutdown_controller

    def tts_request_activity():
        if not shutdown_controller.begin_tts_request():
            raise HTTPException(
                status_code=503,
                detail="Server shutdown is pending; new TTS requests are not accepted.",
            )
        try:
            yield
        finally:
            shutdown_controller.end_tts_request()

    @app.get(f"{API_PREFIX}/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            storage_root=str(store.root),
            loaded=manager.loaded,
            device=manager.device,
            dtype=str(manager.dtype).replace("torch.", ""),
            max_seq_len=app_settings.max_seq_len,
            llama_checkpoint_path=str(app_settings.llama_checkpoint_path),
            decoder_checkpoint_path=str(app_settings.decoder_checkpoint_path),
            shutdown_pending=shutdown_controller.shutdown_pending,
        )

    @app.post(
        f"{API_PREFIX}/admin/shutdown",
        response_model=ShutdownResponse,
        status_code=202,
    )
    def shutdown(request: Request, background_tasks: BackgroundTasks) -> ShutdownResponse:
        if not _is_loopback_request(request):
            raise HTTPException(
                status_code=403, detail="Server shutdown is restricted to loopback clients."
            )
        if request.headers.get(ADMIN_SHUTDOWN_HEADER) != ADMIN_SHUTDOWN_VALUE:
            raise HTTPException(
                status_code=403,
                detail=f"Missing or invalid {ADMIN_SHUTDOWN_HEADER} confirmation header.",
            )
        if not shutdown_controller.configured:
            raise HTTPException(
                status_code=503,
                detail="Server shutdown callback is not configured.",
            )

        accepted = shutdown_controller.request_shutdown("admin_request")
        if accepted:
            background_tasks.add_task(shutdown_controller.execute_shutdown)
        return ShutdownResponse(
            status="accepted" if accepted else "already_pending",
            reason=shutdown_controller.shutdown_reason or "admin_request",
        )

    @app.get(f"{API_PREFIX}/files/{{category}}/{{request_id}}/{{filename:path}}", name="download_file")
    def download_file(category: str, request_id: str, filename: str) -> FileResponse:
        target = store.resolve_public_file(category, request_id, filename)
        return FileResponse(
            target, media_type=_content_type(target.suffix.lstrip(".").lower())
        )

    @app.post(f"{API_PREFIX}/tts/voice_clone", response_model=VoiceCloneResponse)
    async def tts_voice_clone(
        request: Request,
        ref_audio: Annotated[UploadFile, File(...)],
        _activity: None = Depends(tts_request_activity),
        text: Annotated[Optional[str], Form()] = None,
        text_file: Annotated[Optional[UploadFile], File()] = None,
        ref_text: Annotated[Optional[str], Form()] = None,
        ref_text_file: Annotated[Optional[UploadFile], File()] = None,
        output_name: Annotated[Optional[str], Form()] = None,
        format: Annotated[str, Form()] = "wav",
        max_new_tokens: Annotated[int, Form()] = 1024,
        chunk_length: Annotated[int, Form()] = 300,
        iterative_prompt: Annotated[bool, Form()] = True,
        top_p: Annotated[float, Form()] = 0.8,
        repetition_penalty: Annotated[float, Form()] = 1.1,
        temperature: Annotated[float, Form()] = 0.8,
        seed: Annotated[Optional[int], Form()] = None,
        use_memory_cache: Annotated[str, Form()] = "off",
    ) -> VoiceCloneResponse:
        audio_format = _validate_audio_format(format)
        request_id = uuid.uuid4().hex
        synthesis_text, synthesis_text_path = await _resolve_required_text(
            store, request_id, text, text_file, "synthesis_text.txt"
        )
        reference_text, reference_text_path = await _resolve_required_text(
            store, request_id, ref_text, ref_text_file, "reference_text.txt"
        )
        prompt_audio_path = await store.save_upload(
            ref_audio, request_id, fallback_prefix="prompt_audio"
        )
        output_path = store.build_output_path(
            request_id, output_name, "voice_clone", audio_format
        )

        tts_request = _build_tts_request(
            text=synthesis_text,
            reference_audio=audio_to_bytes(str(prompt_audio_path)),
            reference_text=reference_text,
            audio_format=audio_format,
            max_new_tokens=max_new_tokens,
            chunk_length=chunk_length,
            iterative_prompt=iterative_prompt,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            seed=seed,
            use_memory_cache=use_memory_cache,
        )

        try:
            sample_rate, audio = manager.infer(tts_request)
            _write_audio(output_path, audio, sample_rate, audio_format)
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            logger.exception("voice_clone request_id={} failed", request_id)
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return VoiceCloneResponse(
            status="success",
            request_id=request_id,
            audio=_stored_file_from_path(request, store, output_path),
            prompt_audio=_stored_file_from_path(request, store, prompt_audio_path),
            prompt_text=_stored_file_from_path(request, store, reference_text_path),
            synthesis_text=_stored_file_from_path(request, store, synthesis_text_path),
            sample_rate=sample_rate,
            format=audio_format,
        )

    @app.post(f"{API_PREFIX}/tts/voice_clone_batch_file", response_model=VoiceCloneBatchResponse)
    async def tts_voice_clone_batch_file(
        request: Request,
        ref_audio: Annotated[UploadFile, File(...)],
        _activity: None = Depends(tts_request_activity),
        text_file: Annotated[Optional[UploadFile], File()] = None,
        text: Annotated[Optional[list[str]], Form()] = None,
        ref_text: Annotated[Optional[str], Form()] = None,
        ref_text_file: Annotated[Optional[UploadFile], File()] = None,
        output_prefix: Annotated[Optional[str], Form()] = None,
        format: Annotated[str, Form()] = "wav",
        max_new_tokens: Annotated[int, Form()] = 1024,
        chunk_length: Annotated[int, Form()] = 300,
        iterative_prompt: Annotated[bool, Form()] = True,
        top_p: Annotated[float, Form()] = 0.8,
        repetition_penalty: Annotated[float, Form()] = 1.1,
        temperature: Annotated[float, Form()] = 0.8,
        seed: Annotated[Optional[int], Form()] = None,
        use_memory_cache: Annotated[str, Form()] = "on",
    ) -> VoiceCloneBatchResponse:
        audio_format = _validate_audio_format(format)
        request_id = uuid.uuid4().hex
        synthesis_text_path, lines = await _resolve_required_text_lines(
            store, request_id, text_file, text, "synthesis_text.txt"
        )
        reference_text, reference_text_path = await _resolve_required_text(
            store, request_id, ref_text, ref_text_file, "reference_text.txt"
        )
        prompt_audio_path = await store.save_upload(
            ref_audio, request_id, fallback_prefix="prompt_audio"
        )
        reference_audio = audio_to_bytes(str(prompt_audio_path))
        output_base = _safe_stem(
            output_prefix or synthesis_text_path.name,
            "voice_clone",
        )

        audio_paths: list[Path] = []
        sample_rate = 0
        for line_no, synthesis_text in lines:
            output_path = store.build_output_path(
                request_id,
                f"{output_base}_{line_no}",
                "voice_clone",
                audio_format,
            )
            tts_request = _build_tts_request(
                text=synthesis_text,
                reference_audio=reference_audio,
                reference_text=reference_text,
                audio_format=audio_format,
                max_new_tokens=max_new_tokens,
                chunk_length=chunk_length,
                iterative_prompt=iterative_prompt,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                seed=seed,
                use_memory_cache=use_memory_cache,
            )
            try:
                sample_rate, audio = manager.infer(tts_request)
                _write_audio(output_path, audio, sample_rate, audio_format)
            except (ValueError, RuntimeError, FileNotFoundError) as exc:
                logger.exception(
                    "voice_clone_batch request_id={} line_no={} failed",
                    request_id,
                    line_no,
                )
                raise HTTPException(
                    status_code=422, detail=f"Line {line_no}: {exc}"
                ) from exc
            audio_paths.append(output_path)

        return VoiceCloneBatchResponse(
            status="success",
            request_id=request_id,
            audio_paths=[
                _stored_file_from_path(request, store, path) for path in audio_paths
            ],
            prompt_audio=_stored_file_from_path(request, store, prompt_audio_path),
            prompt_text=_stored_file_from_path(request, store, reference_text_path),
            text_file=_stored_file_from_path(request, store, synthesis_text_path),
            sample_rate=sample_rate,
            format=audio_format,
        )

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fish-tts-server",
        description="Launch a FastAPI server for Fish Speech voice clone.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--storage-root", default=None)
    parser.add_argument("--llama-checkpoint-path", default=None)
    parser.add_argument("--decoder-checkpoint-path", default=None)
    parser.add_argument("--decoder-config-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"],
    )
    parser.add_argument("--compile", action="store_true")
    return parser
