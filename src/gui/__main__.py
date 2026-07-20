from __future__ import annotations

import argparse

import uvicorn

from .app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Локальный GUI anki-pipeline")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7999)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
