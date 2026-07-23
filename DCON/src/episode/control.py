"""Graceful early stop for a running episode.

Two ways to request it; both let the loop break at the next step and fall
through to the final map snapshot + visualization (instead of losing them to a
hard kill):
  1. Ctrl-C (SIGINT) → sets a flag. A second Ctrl-C hard-aborts in case
     something is wedged.
  2. Create the sentinel file `<output_dir>/STOP` (e.g. `touch` it, handy when
     running under `docker exec` without an attached TTY).
"""

import os
import signal


class EarlyStop:
    """Context manager installing the SIGINT handler; poll `.reason` per step.

    The default Ctrl-C behavior is restored on exit so the (potentially long)
    finalization + visualization afterwards can be aborted normally.
    """

    def __init__(self, output_dir):
        self.stop_file = os.path.join(output_dir, "STOP")
        self._requested = False
        self._prev_sigint = None

    def __enter__(self):
        # A stale STOP from a previous run would end the new run immediately.
        if os.path.exists(self.stop_file):
            os.remove(self.stop_file)
        self._prev_sigint = signal.signal(signal.SIGINT, self._on_sigint)
        return self

    def __exit__(self, *exc):
        signal.signal(signal.SIGINT, self._prev_sigint)
        return False

    def _on_sigint(self, signum, frame):
        if self._requested:
            raise KeyboardInterrupt
        self._requested = True
        print("\n[main] stop requested (Ctrl-C) — finishing the current step, "
              "then saving maps + rendering the visualization. "
              "Press Ctrl-C again to abort immediately.")

    @property
    def reason(self):
        """Why the loop should stop, or None to keep going."""
        if self._requested:
            return "Ctrl-C"
        if os.path.exists(self.stop_file):
            return f"{self.stop_file} sentinel"
        return None
