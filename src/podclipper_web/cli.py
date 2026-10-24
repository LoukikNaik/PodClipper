"""Separate entry points for the web API and the single processing worker."""

import argparse


def main():
    parser = argparse.ArgumentParser(prog="podclipper-web")
    parser.add_argument("command", choices=["api", "worker"])
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.command == "api":
        import uvicorn
        uvicorn.run("podclipper_web.api:create_app", factory=True, host="127.0.0.1", port=args.port,
                    proxy_headers=False)
    else:
        from .worker import main as worker
        worker()
