import sys
import time


class Simple_Progress_Bar:
    """
    Progress bar with EMA‑smoothed ETA and optional step labels.
    """

    def __init__(self, total: int, enabled: bool = True, bar_length: int = 30):
        self.total = max(1, total)
        self.enabled = enabled
        self.bar_length = bar_length

        # Progress counters
        self.current = 0

        # Timing
        self.start_time = time.time()
        self.last_update_time = self.start_time

        # EMA smoothing for step duration
        self.ema_step_time = None
        self.alpha = 0.12  # smoothing factor

        # Label handling
        self.last_label = ""

    def update(self, step: int = 1, label: str = None):
        if not self.enabled:
            return

        now = time.time()
        step_time = now - self.last_update_time
        self.last_update_time = now

        # Update progress
        self.current += step
        pct = (self.current / self.total) * 100

        # Update EMA only if the step took real time
        if step_time > 0:
            if self.ema_step_time is None:
                self.ema_step_time = step_time
            else:
                self.ema_step_time = (
                    self.alpha * step_time +
                    (1 - self.alpha) * self.ema_step_time
                )

        # Compute ETA using EMA
        elapsed = now - self.start_time
        if self.ema_step_time is None:
            est_total = 0
            remaining = 0
        else:
            est_total = self.ema_step_time * self.total
            remaining = max(0, est_total - elapsed)

        # Bar rendering
        filled = int(self.bar_length * pct / 100)
        bar = "█" * filled + "░" * (self.bar_length - filled)

        # Label
        if label is not None:
            self.last_label = label
        label_str = f" | {self.last_label}" if self.last_label else ""

        # Clear tail to overwrite previous content
        clear_tail = " " * 20

        # Render line
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

        # Finish line
        if self.current >= self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def _fmt(self, seconds: float) -> str:
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"
