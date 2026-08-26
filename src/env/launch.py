"""Producer launcher (Shell A entry point) — spawns obs_node + bridge.

obs_node publishes /obs/* on ROS1; bridge mirrors them to ROS2.
rtnav_runner.py (Shell B) subscribes on the ROS2 side.
"""
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent

# obs_node first so /obs/* is publishing before the bridge tries to subscribe.
PROCESSES = [
    ("obs_node.py", 2.0),
    ("bridge.py", 0.0),
]


def _pump_output(proc, label):
    for line in proc.stdout:
        sys.stdout.write(f"[{label}] {line}")
        sys.stdout.flush()


def main():
    procs = []

    def shutdown(*_):
        print("\nshutting down children...")
        for p, _label in procs:
            if p.poll() is None:
                p.terminate()
        deadline = time.time() + 3.0
        for p, _label in procs:
            remaining = max(0.0, deadline - time.time())
            try:
                p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    for script, delay in PROCESSES:
        path = HERE / script
        if not path.exists():
            print(f"ERROR: {path} not found", file=sys.stderr)
            shutdown()
        label = path.stem
        p = subprocess.Popen(
            [sys.executable, "-u", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
        procs.append((p, label))
        threading.Thread(target=_pump_output, args=(p, label), daemon=True).start()
        print(f"[producer] started {label} (pid {p.pid})")
        if delay > 0:
            time.sleep(delay)

    while True:
        for p, label in procs:
            rc = p.poll()
            if rc is not None:
                print(f"[producer] {label} exited (code {rc}) — tearing down")
                shutdown()
        time.sleep(0.5)


if __name__ == "__main__":
    main()
