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
    model_repo_id,
    resolve_project_path,
    weights_root,
)


def parse_args() -> argparse.Namespace:
    load_project_env()

    parser = argparse.ArgumentParser(
        description="Download Fish Speech model weights into the local weights folder.",
    )
    parser.add_argument(
        "--repo-id",
        default=model_repo_id(),
        help=f"Hugging Face repo id to download. Default: {DEFAULT_MODEL_REPO}",
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
        help="Optional Hugging Face revision, branch, or commit hash.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Optional Hugging Face access token.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=int(os.environ.get("HF_HUB_DOWNLOAD_THREADS", "4")),
        help="Number of parallel download workers.",
    )
    parser.add_argument(
        "--enable-xet",
        action="store_true",
        help="Use Hugging Face Xet downloads. Disabled by default for slow links.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved paths without downloading.",
    )

    return parser.parse_args()


def validate_download(local_dir: Path) -> list[str]:
    missing = []

    if not (local_dir / "config.json").is_file():
        missing.append("config.json")

    weight_files = (
        local_dir / "model.safetensors",
        local_dir / "model.safetensors.index.json",
        local_dir / "model.pth",
    )
    if not any(path.is_file() for path in weight_files):
        missing.append("model.safetensors, model.safetensors.index.json, or model.pth")

    if not (local_dir / "codec.pth").is_file():
        missing.append("codec.pth")

    return missing


def download_snapshot(
    repo_id: str,
    local_dir: Path,
    revision: str | None,
    token: str | None,
    max_workers: int,
) -> Path:
    from huggingface_hub import snapshot_download

    kwargs = {
        "repo_id": repo_id,
        "local_dir": str(local_dir),
        "max_workers": max_workers,
    }
    if revision:
        kwargs["revision"] = revision
    if token:
        kwargs["token"] = token

    return Path(snapshot_download(**kwargs))


def main() -> None:
    args = parse_args()

    hf_home = weights_root()
    os.environ["HF_HOME"] = str(hf_home)
    if args.enable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "0"
    else:
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    local_dir = (
        resolve_project_path(args.local_dir)
        if args.local_dir is not None
        else checkpoint_path(args.repo_id)
    )
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"HF_HOME={hf_home}")
    print(f"HF_HUB_DISABLE_XET={os.environ['HF_HUB_DISABLE_XET']}")
    print(f"repo_id={args.repo_id}")
    print(f"local_dir={local_dir}")

    if args.dry_run:
        return

    download_snapshot(
        repo_id=args.repo_id,
        local_dir=local_dir,
        revision=args.revision,
        token=args.token,
        max_workers=args.max_workers,
    )

    missing = validate_download(local_dir)
    if missing:
        missing_list = ", ".join(missing)
        raise SystemExit(f"Downloaded snapshot is missing required files: {missing_list}")

    print(f"Model weights are ready at: {local_dir}")


if __name__ == "__main__":
    main()
