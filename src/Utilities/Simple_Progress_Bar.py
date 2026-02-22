import sys
import time


class Simple_Progress_Bar:
    """
    Reusable progress bar with ETA and elapsed time.
    Now supports optional step labels.
    """

    def __init__(self, total: int, enabled: bool = True, bar_length: int = 30):
        self.total = max(1, total)
        self.enabled = enabled
        self.bar_length = bar_length

        self.start_time = time.time()
        self.current = 0

        # NEW: store last label printed
        self.last_label = ""

    def update(self, step: int = 1, label: str = None):
        if not self.enabled:
            return

        self.current += step
        pct = (self.current / self.total) * 100

        filled = int(self.bar_length * pct / 100)
        bar = "█" * filled + "░" * (self.bar_length - filled)

        elapsed = time.time() - self.start_time
        avg = elapsed / max(1, self.current)
        est_total = avg * self.total
        remaining = est_total - elapsed

        if label is not None:
            self.last_label = label

        label_str = f" | {self.last_label}" if self.last_label else ""
        clear_tail = " " * 20

        # Force overwrite of the same line
        sys.stdout.write("\r")
        sys.stdout.write(
            f"[{bar}] {pct:5.1f}%  "
            f"({self.current}/{self.total})  "
            f"Elapsed: {self._fmt(elapsed)} | "
            f"ETA: {self._fmt(remaining)} | "
            f"Total est.: {self._fmt(est_total)}"
            f"{label_str}{clear_tail}"
        )
        sys.stdout.flush()

        # Only print a newline at the very end
        if self.current >= self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def _fmt(self, seconds: float) -> str:
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"
