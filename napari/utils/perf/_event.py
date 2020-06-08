"""PerfEvent class.
"""


class PerfEvent:
    """One perf event represents a span of time.

    Parameters
    ----------
    category : str
        You can toggle categories on/off in some GUIs.
    name : str
        The name of this event like "draw".
    start_ns : int
        Start time in nanoseconds.
    end_ns : int
        End time in nanoseconds.

    Notes
    -----
    The times are from perf_counter_ns() and do not indicate time of day,
    the origin is arbitrary. But subtracting two counters is valid.
    """

    def __init__(self, category: str, name: str, start_ns: int, end_ns: int):
        self.category = category
        self.name = name
        self.start_ns = start_ns
        self.end_ns = end_ns

    @property
    def start_us(self):
        return self.start_ns / 1e3

    @property
    def start_ms(self):
        return self.start_ns / 1e6

    @property
    def duration_ns(self):
        return self.end_ns - self.start_ns

    @property
    def duration_us(self):
        return self.duration_ns / 1e3

    @property
    def duration_ms(self):
        return self.duration_ns / 1e6
