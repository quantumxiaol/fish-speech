from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pyrootutils
import uvicorn

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from tools.fastapi_service import ServiceSettings, build_parser, create_app  # noqa: E402


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    settings = ServiceSettings.from_env()
    if args.storage_root:
        settings = replace(
            settings, storage_root=Path(args.storage_root).expanduser().resolve()
        )
    if args.llama_checkpoint_path:
        settings = replace(
            settings,
            llama_checkpoint_path=Path(args.llama_checkpoint_path).expanduser().resolve(),
        )
    if args.decoder_checkpoint_path:
        settings = replace(
            settings,
            decoder_checkpoint_path=Path(args.decoder_checkpoint_path)
            .expanduser()
            .resolve(),
        )
    if args.decoder_config_name:
        settings = replace(settings, decoder_config_name=args.decoder_config_name)
    if args.device:
        settings = replace(settings, device=args.device)
    if args.max_seq_len:
        settings = replace(settings, max_seq_len=args.max_seq_len)
    if args.dtype:
        settings = replace(settings, dtype=args.dtype)
    if args.compile:
        settings = replace(settings, compile=True)
    if args.mps_profile:
        settings = replace(settings, mps_profile=True)

    app = create_app(settings=settings)
    config = uvicorn.Config(
        app=app,
        host=args.host,
        port=args.port,
        proxy_headers=False,
    )
    server = uvicorn.Server(config)

    def request_graceful_shutdown() -> None:
        server.should_exit = True

    app.state.shutdown_controller.set_shutdown_callback(request_graceful_shutdown)
    server.run()


if __name__ == "__main__":
    main()
