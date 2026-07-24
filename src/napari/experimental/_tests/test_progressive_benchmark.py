"""Opt-in wall-clock benchmark for progressive loading interaction.

The invariant tests (``test_streaming_invariants.py``) guarantee the
*mechanisms* behind smooth browsing cannot regress; this benchmark
measures the *seconds* a user feels while zooming and panning. It is
not a CI gate (hardware variance makes wall-clock assertions flaky);
run it manually:

    NAPARI_PROGRESSIVE_BENCH=1 python -m pytest -s \
        src/napari/experimental/_tests/test_progressive_benchmark.py

It drives a scripted zoom-in (crossing several resolution-level
switches, the reported worst case) followed by pans at depth, over the
local generative Mandelbrot zarr — deterministic and network-free,
unlike remote S3 data — and reports per move:

- ``stall``: the longest gap between event-loop heartbeats from the
  camera move until settle. Any synchronous main-thread work — slicing,
  ``_check``/``_on_set_data``, draws, GPU uploads — shows up here; this
  IS the freeze the user feels. (Polling overhead makes it a slight
  upper bound.)
- ``first content``: time until the viewport's resident data is
  non-blank (< 5% zeros) — how long the user looks at nothing new.
- ``coverage``: time until every visible chunk of the displayed level
  is loaded — how long until the view is fully sharp.
- ``blank @settle``: fraction of unfilled (zero) pixels at settle;
  nonzero means the never-blank guarantee failed.

The target for smooth browsing is: stall ~0 (never freezes), first
content ~0 (coarse data always shown), coverage bounded by IO only.
"""

import itertools
import os
import time

import numpy as np
import pytest

pytest.importorskip('qtpy', reason='requires Qt backend')
pytest.importorskip('zarr', reason='requires zarr for generative data')

pytestmark = pytest.mark.skipif(
    os.environ.get('NAPARI_PROGRESSIVE_BENCH') != '1',
    reason='opt-in benchmark: set NAPARI_PROGRESSIVE_BENCH=1',
)

#: Per-move settle timeout. A move that cannot reach coverage within
#: this window is reported as such (a stall in itself).
MOVE_TIMEOUT_S = 60.0
#: Poll cadence while waiting for a move to settle.
POLL_MS = 25
#: Zero fraction below which the viewport counts as showing content.
CONTENT_THRESHOLD = 0.05


def _viewport_zero_fraction(loader, layer) -> float:
    """Fraction of unfilled (zero) pixels in the displayed interval.

    Subsampled so the poll itself stays cheap enough not to pollute the
    heartbeat measurements.
    """
    vdata = loader._data[int(layer.data_level)]
    if vdata.interval is None:
        return 1.0
    with vdata.lock:
        hyperslice = vdata.hyperslice
        if hyperslice.size == 0:
            return 1.0
        step = tuple(max(1, s // 256) for s in hyperslice.shape)
        sample = hyperslice[tuple(slice(None, None, s) for s in step)]
        return float((sample == 0).mean())


def _coverage_complete(loader, layer) -> bool:
    """Every chunk of the displayed level's interval is loaded."""
    from napari.experimental._progressive_loading import (
        _chunk_id,
        chunk_slices,
    )

    vdata = loader._data[int(layer.data_level)]
    if vdata.interval is None:
        return False
    keys = chunk_slices(vdata, interval=vdata.interval)
    return all(
        _chunk_id(key) in vdata.loaded_chunks
        for key in itertools.product(*keys)
    )


def test_progressive_bench_zoom_pan(qtbot, make_napari_viewer, monkeypatch):
    from qtpy.QtCore import QTimer

    import napari.experimental._progressive_loading as pl
    from napari.experimental._progressive_loading_datasets import (
        mandelbrot_dataset,
    )

    # attribute the worst single main-thread stall to the loader's own
    # entry points (class-level patch, before the loader is constructed)
    sections = {'_check': 0.0, '_on_set_data': 0.0}

    def timed(name, original):
        def run(self, *args, **kwargs):
            start = time.perf_counter()
            try:
                return original(self, *args, **kwargs)
            finally:
                sections[name] = max(
                    sections[name],
                    time.perf_counter() - start,
                )

        return run

    monkeypatch.setattr(
        pl.ProgressiveLoader,
        '_check',
        timed('_check', pl.ProgressiveLoader._check),
    )
    monkeypatch.setattr(
        pl.ProgressiveLoader,
        '_on_set_data',
        timed('_on_set_data', pl.ProgressiveLoader._on_set_data),
    )

    dataset = mandelbrot_dataset(max_levels=8, cpu_relief=0.0)
    viewer = make_napari_viewer(show=True)
    layer = pl.add_progressive_loading_image(
        dataset['arrays'],
        viewer=viewer,
        contrast_limits=(0, 255),
    )
    loader = layer.metadata['progressive_loader']

    # event-loop heartbeat: gaps between 5ms ticks are main-thread stalls
    ticks: list[float] = []
    heartbeat = QTimer()
    heartbeat.setInterval(5)
    heartbeat.timeout.connect(lambda: ticks.append(time.perf_counter()))
    heartbeat.start()

    def idle():
        return (
            loader._worker is None
            and loader._resident_worker is None
            and loader._repair_worker is None
        )

    qtbot.waitUntil(idle, timeout=int(MOVE_TIMEOUT_S * 1000))
    qtbot.wait(300)

    # scripted session: zoom toward the view center through several
    # level switches, then pan by half a viewport at depth
    moves: list[tuple[str, float | tuple[int, int]]] = [
        ('zoom', float(viewer.camera.zoom) * 2**i) for i in range(1, 9)
    ]
    moves += [('pan', d) for d in ((0, 1), (1, 0), (0, -1), (1, 1))]

    rows = []
    for kind, arg in moves:
        ticks.clear()
        start = time.perf_counter()
        if kind == 'zoom':
            viewer.camera.zoom = arg
            label = f'zoom x{arg / moves[0][1] * 2:g}'
        else:
            span = max(viewer._canvas_size) / viewer.camera.zoom
            center = np.asarray(viewer.camera.center, dtype=float)
            center[1] += arg[0] * span * 0.5
            center[2] += arg[1] * span * 0.5
            viewer.camera.center = tuple(center)
            label = f'pan {arg}'

        t_content = None
        t_coverage = None
        while time.perf_counter() - start < MOVE_TIMEOUT_S:
            qtbot.wait(POLL_MS)
            now = time.perf_counter() - start
            if (
                t_content is None
                and _viewport_zero_fraction(loader, layer) < CONTENT_THRESHOLD
            ):
                t_content = now
            if idle() and _coverage_complete(loader, layer):
                t_coverage = now
                break
        qtbot.wait(100)
        gaps = np.diff(ticks) if len(ticks) > 1 else np.array([0.0])
        rows.append(
            (
                label,
                int(layer.data_level),
                float(gaps.max()) * 1e3,
                t_content,
                t_coverage,
                _viewport_zero_fraction(loader, layer),
            ),
        )

    heartbeat.stop()

    def fmt_s(value):
        return f'{value:8.2f}s' if value is not None else '  >60s  '

    print()  # noqa: T201
    print(  # noqa: T201
        f'{"move":<14} {"lvl":>3} {"stall":>9} {"first content":>13} '
        f'{"coverage":>9} {"blank @settle":>13}',
    )
    for label, level, stall_ms, t_content, t_coverage, blank in rows:
        print(  # noqa: T201
            f'{label:<14} {level:>3} {stall_ms:7.0f}ms {fmt_s(t_content):>13} '
            f'{fmt_s(t_coverage):>9} {blank:>12.1%}',
        )
    worst_stall = max(row[2] for row in rows)
    worst_blank = max(row[5] for row in rows)
    print(  # noqa: T201
        f'\nworst stall {worst_stall:.0f}ms | '
        f'worst _check {sections["_check"] * 1e3:.0f}ms | '
        f'worst _on_set_data {sections["_on_set_data"] * 1e3:.0f}ms | '
        f'worst blank@settle {worst_blank:.1%}\n'
        'target: stall ~0ms, first content ~0s, blank 0%',
    )
    # benchmark, not a gate: only sanity-check that measurements exist
    assert rows
    loader.close()
    qtbot.wait(300)
