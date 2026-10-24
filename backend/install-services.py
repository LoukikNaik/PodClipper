"""Install this user's API and worker LaunchAgents on the always-on Mac."""

from pathlib import Path
import os
import plistlib
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parent.parent
binary = root / ".venv/bin/podclipper-web"
if not binary.exists() or not (root / ".env").is_file():
    raise SystemExit("Install podclipper[web] in .venv and configure the private .env first")
logs = root / "web-data/logs"
logs.mkdir(parents=True, exist_ok=True, mode=0o700)
agents = Path.home() / "Library/LaunchAgents"
agents.mkdir(parents=True, exist_ok=True)
for command in ("api", "worker"):
    label = f"dev.podclipper.web-{command}"
    path = agents / f"{label}.plist"
    program = [str(binary), command]
    if command == "worker":
        program = ["/usr/bin/caffeinate", "-i", *program]
    config = {"Label": label, "ProgramArguments": program, "WorkingDirectory": str(root),
              "EnvironmentVariables": {"PATH": f"{root}/.venv/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
                                       "PODCLIPPER_WEB_ENV": str(root / ".env")},
              "RunAtLoad": True, "KeepAlive": True, "ThrottleInterval": 15,
              "StandardOutPath": str(logs / f"{command}.log"),
              "StandardErrorPath": str(logs / f"{command}-error.log")}
    subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}/{label}"], capture_output=True)
    path.write_bytes(plistlib.dumps(config))
    subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)], check=True)
    print(f"Installed {label}")

tunnel_config = root / "backend/tunnel.yml"
cloudflared = shutil.which("cloudflared") or "/opt/homebrew/bin/cloudflared"
if tunnel_config.is_file() and Path(cloudflared).is_file():
    label = "dev.podclipper.web-tunnel"
    path = agents / f"{label}.plist"
    config = {"Label": label, "ProgramArguments": [cloudflared, "tunnel", "--config", str(tunnel_config), "run"],
              "WorkingDirectory": str(root), "RunAtLoad": True, "KeepAlive": True,
              "ThrottleInterval": 15, "StandardOutPath": str(logs / "tunnel.log"),
              "StandardErrorPath": str(logs / "tunnel-error.log")}
    subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}/{label}"], capture_output=True)
    path.write_bytes(plistlib.dumps(config))
    subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)], check=True)
    print(f"Installed {label}")
