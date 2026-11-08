from __future__ import annotations

import argparse
from typing import Sequence

import uvicorn

from fruitmatch_ml.training import train_profile


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fruitmatch-ml")
    commands = parser.add_subparsers(dest="command", required=True)

    serve_parser = commands.add_parser("serve", help="Run the FastAPI recommender.")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8001)

    train_parser = commands.add_parser("train", help="Train one saved user profile.")
    train_parser.add_argument("--profile-id", required=True)

    arguments = parser.parse_args(argv)
    if arguments.command == "serve":
        uvicorn.run(
            "fruitmatch_ml.api:app",
            host=arguments.host,
            port=arguments.port,
            reload=False,
        )
        return 0

    result = train_profile(arguments.profile_id)
    if not result["trained"]:
        print("No saved feedback is available for this profile yet.")
        return 1

    print(
        f"CleanRL PPO trained from {result['feedback_count']} saved responses."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())