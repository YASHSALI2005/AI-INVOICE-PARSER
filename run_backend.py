#!/usr/bin/env python3
"""
Start the FastAPI backend on the first available port (8000, 8001, 8002).
Updates frontend/.env with the chosen port so the frontend can reach the API.
"""
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTEND_ENV = ROOT.parent / "frontend" / ".env"
PORTS = [8000, 8001, 8002, 8003, 8004, 8005]


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def main():
    port = None
    for p in PORTS:
        if port_free(p):
            port = p
            break
    if port is None:
        print("No free port. Free one with: lsof -ti :8000 | xargs kill  (or :8001, :8002, ...)", file=sys.stderr)
        sys.exit(1)

    url = f"http://localhost:{port}"
    # Update frontend .env so Vite uses this API URL
    if FRONTEND_ENV.exists():
        content = FRONTEND_ENV.read_text()
        if "VITE_API_URL=" in content:
            lines = []
            for line in content.splitlines():
                if line.strip().startswith("VITE_API_URL="):
                    lines.append(f"VITE_API_URL={url}")
                else:
                    lines.append(line)
            FRONTEND_ENV.write_text("\n".join(lines) + "\n")
        else:
            FRONTEND_ENV.write_text(content.rstrip() + f"\nVITE_API_URL={url}\n")
    else:
        FRONTEND_ENV.parent.mkdir(parents=True, exist_ok=True)
        FRONTEND_ENV.write_text(f"VITE_API_URL={url}\n")

    print(f"Starting backend on {url}")
    print(f"Frontend .env set to VITE_API_URL={url} (restart frontend dev server if it's running)")
    print(f"Docs: {url}/docs  Health: {url}/health")
    sys.stdout.flush()

    venv_python = ROOT / ".venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = sys.executable
    subprocess.run(
        [str(venv_python), "-m", "uvicorn", "main:app", "--reload", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(ROOT),
    )


if __name__ == "__main__":
    main()
