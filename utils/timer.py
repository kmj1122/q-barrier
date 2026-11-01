import time


class Timer:
    """Generic timer utility with start, pause, and reset; tracks total elapsed ms."""
    def __init__(self):
        self.total_ms = 0.0
        self._running = False
        self._start_time = None

    def start(self):
        if self._running:
            self.pause()
        self._start_time = time.perf_counter()
        self._running = True

    def pause(self):
        if not self._running or self._start_time is None:
            return
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000.0
        self.total_ms += elapsed_ms
        self._running = False
        self._start_time = None

    def reset(self):
        self.total_ms = 0.0
        self._running = False
        self._start_time = None


