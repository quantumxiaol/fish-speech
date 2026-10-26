import os
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_MODEL_REPO = "fishaudio/s2-pro"
DEFAULT_MODEL_NAME = "s2-pro"
DEFAULT_WEIGHTS_ROOT = "modelsweights"

_ENV_LOADED = False


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_project_env() -> None:
    global _ENV_LOADED

    if _ENV_LOADED:
        return

    load_dotenv(project_root() / ".env", override=False)
    _ENV_LOADED = True


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return project_root() / path


def default_device(fallback: str = "cuda") -> str:
    load_project_env()
    return os.environ.get("DEVICE", fallback)


def weights_root() -> Path:
    load_project_env()
    return resolve_project_path(os.environ.get("HF_HOME", DEFAULT_WEIGHTS_ROOT))


def model_repo_id() -> str:
    load_project_env()
    return os.environ.get("FISH_SPEECH_MODEL_ID", DEFAULT_MODEL_REPO)


def model_name(repo_id: str | None = None) -> str:
    load_project_env()
    if name := os.environ.get("FISH_SPEECH_MODEL_NAME"):
        return name

    repo_id = repo_id or model_repo_id()
    return repo_id.rstrip("/").split("/")[-1]


def checkpoint_path(repo_id: str | None = None) -> Path:
    load_project_env()
    explicit_path = os.environ.get("FISH_SPEECH_CHECKPOINT_PATH") or os.environ.get(
        "FISH_SPEECH_MODEL_DIR"
    )
    if explicit_path:
        return resolve_project_path(explicit_path)

    return weights_root() / model_name(repo_id)


def decoder_checkpoint_path(repo_id: str | None = None) -> Path:
    load_project_env()
    if explicit_path := os.environ.get("FISH_SPEECH_DECODER_CHECKPOINT_PATH"):
        return resolve_project_path(explicit_path)

    return checkpoint_path(repo_id) / "codec.pth"
