"""One-key e-stop on stdin.

Actions execute unprompted, so this is the only manual brake between the policy
and a moving base. Any keypress fires the callback; the thread then exits, since
after a stop there is nothing left to watch for.
"""
import select
import sys
import threading

POLL_S = 0.2


def start(on_stop, shutdown_event=None):
    """Watch stdin and call on_stop() on the first keypress.

    Returns the watcher thread, or None when there is no tty to read (piped
    output, no terminal attached) and the e-stop cannot work.
    """
    if not sys.stdin or not sys.stdin.isatty():
        print("[estop] no tty — e-stop unavailable")
        return None

    def loop():
        while shutdown_event is None or not shutdown_event.is_set():
            try:
                ready, _, _ = select.select([sys.stdin], [], [], POLL_S)
                if not ready:
                    continue
                if sys.stdin.readline() == "":
                    return          # EOF, not a keypress
            except (OSError, ValueError):
                return              # stdin closed under us during shutdown

            on_stop()
            _banner("\033[1;31m[estop] STOPPED — base zeroed. Ctrl+C to exit.\033[0m")
            if shutdown_event is not None:
                shutdown_event.set()
            return

    thread = threading.Thread(target=loop, name="rtnav-estop", daemon=True)
    thread.start()
    _banner("\033[1;33m[estop] armed — press ENTER at any time to stop.\033[0m")
    return thread


def _banner(text):
    """stderr, so the message survives a piped or interleaved stdout."""
    sys.stderr.write("\n" + text + "\n")
    sys.stderr.flush()
