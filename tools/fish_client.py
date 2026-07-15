from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from tools.fish_httpx_client import (  # noqa: E402
    DEFAULT_SERVER_URL,
    FishSpeechClientError,
    FishSpeechHttpxClient,
)


def _add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--chunk-length", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use-memory-cache", choices=["on", "off"], default=None)


def _collect_generation_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    mapping = {
        "max_new_tokens": args.max_new_tokens,
        "chunk_length": args.chunk_length,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "seed": args.seed,
        "use_memory_cache": args.use_memory_cache,
    }
    return {key: value for key, value in mapping.items() if value is not None}


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a finite number greater than 0")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fish-tts-client",
        description="HTTP client for the Fish Speech FastAPI service.",
    )
    parser.add_argument(
        "--server-url",
        default=DEFAULT_SERVER_URL,
        help=f"Server origin without API prefix (default: {DEFAULT_SERVER_URL}).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("health", help="Query service health.")
    shutdown_parser = subparsers.add_parser(
        "shutdown",
        help="Gracefully stop a local Fish Speech server.",
    )
    shutdown_parser.add_argument(
        "--wait-timeout",
        type=_positive_float,
        default=30.0,
        help="Seconds to wait for the server port to stop responding.",
    )
    shutdown_parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Return after shutdown is accepted instead of waiting for exit.",
    )

    download_parser = subparsers.add_parser("download", help="Download a file by URL.")
    download_parser.add_argument("--url", required=True)
    download_parser.add_argument("--output", required=True)

    voice_clone_parser = subparsers.add_parser(
        "voice-clone", help="Call /fishspeech/tts/voice_clone."
    )
    voice_clone_parser.add_argument("--ref-audio", required=True)
    voice_clone_parser.add_argument("--text", default=None)
    voice_clone_parser.add_argument("--text-file", default=None)
    voice_clone_parser.add_argument("--ref-text", default=None)
    voice_clone_parser.add_argument("--ref-text-file", default=None)
    voice_clone_parser.add_argument("--output-name", default=None)
    voice_clone_parser.add_argument("--format", default="wav")
    voice_clone_parser.add_argument("--download-to", default=None)
    _add_generation_args(voice_clone_parser)

    batch_parser = subparsers.add_parser(
        "voice-clone-batch-file",
        aliases=["voice-clone-batch"],
        help="Call /fishspeech/tts/voice_clone_batch_file.",
    )
    batch_parser.add_argument("--ref-audio", required=True)
    batch_parser.add_argument("--text-file", default=None)
    batch_parser.add_argument(
        "--text",
        action="append",
        default=None,
        help="Batch synthesis text. Repeat for multiple lines.",
    )
    batch_parser.add_argument("--ref-text", default=None)
    batch_parser.add_argument("--ref-text-file", default=None)
    batch_parser.add_argument("--output-prefix", default=None)
    batch_parser.add_argument("--format", default="wav")
    batch_parser.add_argument("--download-dir", default=None)
    _add_generation_args(batch_parser)

    return parser


def _print_payload(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        with FishSpeechHttpxClient(server_url=args.server_url) as client:
            if args.command == "health":
                _print_payload(client.health())
                return

            if args.command == "download":
                target = client.download_url(args.url, args.output)
                _print_payload({"status": "success", "output_path": str(Path(target))})
                return

            if args.command == "shutdown":
                _print_payload(
                    client.shutdown(
                        wait=not args.no_wait,
                        wait_timeout=args.wait_timeout,
                    )
                )
                return

            generation_kwargs = _collect_generation_kwargs(args)

            if args.command == "voice-clone":
                _print_payload(
                    client.voice_clone(
                        ref_audio_path=args.ref_audio,
                        text=args.text,
                        text_file=args.text_file,
                        ref_text=args.ref_text,
                        ref_text_file=args.ref_text_file,
                        output_name=args.output_name,
                        audio_format=args.format,
                        download_to=args.download_to,
                        **generation_kwargs,
                    )
                )
                return

            if args.command in {"voice-clone-batch-file", "voice-clone-batch"}:
                if args.text_file and args.text:
                    parser.error(
                        "voice-clone-batch accepts either --text-file or repeated --text values, not both."
                    )
                if not args.text_file and not args.text:
                    parser.error(
                        "voice-clone-batch requires --text-file or at least one --text value."
                    )
                _print_payload(
                    client.voice_clone_batch_file(
                        ref_audio_path=args.ref_audio,
                        text_file=args.text_file,
                        texts=args.text,
                        ref_text=args.ref_text,
                        ref_text_file=args.ref_text_file,
                        output_prefix=args.output_prefix,
                        audio_format=args.format,
                        download_dir=args.download_dir,
                        **generation_kwargs,
                    )
                )
                return

            raise SystemExit(f"Unsupported command: {args.command}")
    except FishSpeechClientError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
