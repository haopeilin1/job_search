import subprocess
import sys
import os

cmd = [
    sys.executable, "-m", "uvicorn",
    "app.main:app",
    "--host", "0.0.0.0",
    "--port", "8000",
    "--reload"
]

# Use creationflags to create a new process group on Windows
creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0

with open("logs/backend.log", "a") as log:
    proc = subprocess.Popen(
        cmd,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    print(f"Started uvicorn with PID {proc.pid}")
