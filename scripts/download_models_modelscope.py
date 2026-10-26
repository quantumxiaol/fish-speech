#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fish_speech.env_config import (  # noqa: E402
    DEFAULT_MODEL_REPO,
    checkpoint_path,
    load_project_env,
    resolve_project_path,
    weights_root,
)
from scripts.download_models import validate_download  # noqa: E402


def default_modelscope_model_id() -> str:
    load_project_env()
    return os.environ.get("FISH_SPEECH_MODELSCOPE_MODEL_ID", DEFAULT_MODEL_REPO)


def parse_args() -> argparse.Namespace:
    load_project_env()

    parser = argparse.ArgumentParser(
        description="Download Fish Speech model weights from ModelScope.",
    )
    parser.add_argument(
        "--model-id",
        default=default_modelscope_model_id(),
        help="ModelScope model id to download. Default: fishaudio/s2-pro",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help=(
            "Local model directory. Defaults to "
            "$FISH_SPEECH_CHECKPOINT_PATH, $FISH_SPEECH_MODEL_DIR, "
            "or $HF_HOME/<model-name>."
        ),
    )
    parser.add_argument(
        "--revision",
        default=os.environ.get("FISH_SPEECH_MODEL_REVISION"),
        help="Optional ModelScope revision, branch, or tag.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("MODELSCOPE_TOKEN"),
        help="Optional ModelScope access token.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved paths without downloading.",
    )

    return parser.parse_args()


def download_snapshot(
    model_id: str,
    local_dir: Path,
    revision: str | None,
    token: str | None,
) -> Path:
    from modelscope.hub.snapshot_download import snapshot_download

    kwargs = {
        "model_id": model_id,
        "local_dir": str(local_dir),
    }
    if revision:
        kwargs["revision"] = revision
    if token:
        os.environ["MODELSCOPE_API_TOKEN"] = token

    return Path(snapshot_download(**kwargs))


def main() -> None:
    args = parse_args()

    weights_dir = weights_root()
    os.environ["MODELSCOPE_CACHE"] = str(weights_dir)

    local_dir = (
        resolve_project_path(args.local_dir)
        if args.local_dir is not None
        else checkpoint_path(args.model_id)
    )
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"MODELSCOPE_CACHE={weights_dir}")
    print(f"model_id={args.model_id}")
    print(f"local_dir={local_dir}")

    if args.dry_run:
        return

    download_snapshot(
        model_id=args.model_id,
        local_dir=local_dir,
        revision=args.revision,
        token=args.token,
    )

    missing = validate_download(local_dir)
    if missing:
        missing_list = ", ".join(missing)
        raise SystemExit(f"Downloaded snapshot is missing required files: {missing_list}")

    print(f"Model weights are ready at: {local_dir}")


if __name__ == "__main__":
    main()
