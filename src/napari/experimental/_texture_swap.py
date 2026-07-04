"""Double-buffered texture streaming for vispy Volume and Image nodes.

Writing into a texture that the GPU is concurrently sampling is the
pathological path on macOS GL-over-Metal (and many other drivers): the
driver must either stall until in-flight frames finish or ghost-copy
the whole texture, and that cost lands inside the next ``draw()`` call
— multi-second stalls for a 32 MB volume, regardless of how small or
rare the individual ``glTexSubImage3D`` updates are. Upload pacing
alone cannot fix this; the rendered texture must never be written.

:class:`DoubleBufferedVolumeTexture` gives a vispy ``VolumeVisual`` two
textures with identical shape/format:

- the *front* texture is bound to the shader and is never modified;
- all chunk patches and full-volume rewrites are staged into the *back*
  texture (still rate-limited and GLIR-metered upstream);
- :meth:`present` swaps the sampler binding (one uniform rebind) and
  replays, from a patch log, whatever the new back texture missed while
  it was front.

Net cost: every staged byte is uploaded twice (once per texture, both
times to an unbound texture) and 2x texture GPU memory. Net win: draws
only ever sample stable textures.
"""

from __future__ import annotations

import contextlib
import logging
import time

import numpy as np

LOGGER = logging.getLogger('napari.experimental._texture_swap')


def _texture_format_will_change(texture, data, spatial_ndim: int) -> bool:
    """Whether staging ``data`` would change ``texture``'s GL format.

    A full rewrite whose channel count or dtype resolves to a different
    internalformat cannot be staged into the fixed-format double-buffer
    pair (vispy raises from ``check_data_format``); it must go through a
    reshape that allocates a matching-format texture instead.

    Uses vispy's own ``_internalformat_will_change`` when available (the
    authoritative check, comparing resolved GL formats rather than raw
    dtypes), falling back to a channel-count comparison for non-scaled
    textures.
    """
    check = getattr(texture, '_internalformat_will_change', None)
    if check is not None:
        try:
            return bool(check(data))
        except Exception:  # noqa: BLE001 # pragma: no cover - vispy moved
            pass
    tex_shape = tuple(texture.shape)
    tex_channels = (
        tex_shape[spatial_ndim] if len(tex_shape) > spatial_ndim else 1
    )
    data_channels = data.shape[spatial_ndim] if data.ndim > spatial_ndim else 1
    return data_channels != tex_channels

#: Retired textures kept for reuse instead of deleted. GL object
#: deletion synchronizes with the GPU pipeline (profiled ~25ms per
#: DELETE on busy macOS GL-over-Metal), and reallocation costs another
#: sync — so zooming back and forth across two levels would otherwise
#: pay 4 syncs per switch. Bounded GPU memory cost: at most this many
#: spare tiles. Sized for the 16 MB default tile cap, whose quantized
#: shape vocabulary spans ~5 sizes (96..251^3) — up to 4 retired
#: front/back pairs alive at once, <= ~60 MB of spares.
#:
DEFAULT_TEXTURE_POOL_SIZE = 8


class _TransformHoldMixin:
    """Coordinate node-matrix changes with deferred texture presents.

    While a full rewrite/reshape is staged but not yet presented, the
    node matrix matching the still-rendered FRONT content is held; the
    matrix napari applies for the staged tile (its origin/scale change
    in the same ``set_data`` emission) is captured by the loader and
    applied at the swap, so texture content and tile transform always
    change in the same frame — the old tile never renders misplaced.
    """

    _node = None
    _held_matrix = None
    _pending_matrix = None

    def _node_matrix_transform(self):
        transform = getattr(self._node, 'transform', None)
        return transform if hasattr(transform, 'matrix') else None

    def _begin_transform_hold(self) -> None:
        """Snapshot the node matrix matching the front's content.

        Called when a full rewrite/reshape is staged, BEFORE napari
        updates the node matrix for the new tile (vispy layers call
        ``node.set_data`` first, ``_on_matrix_change`` right after).
        Idempotent while already holding: the front content does not
        change between holds, so the first snapshot stays authoritative.
        """
        if self._held_matrix is not None:
            return
        transform = self._node_matrix_transform()
        if transform is not None:
            self._held_matrix = np.array(transform.matrix, copy=True)

    def capture_transform(self) -> None:
        """Capture napari's new matrix; keep the front-matching one.

        Called by the loader inside the ``set_data`` event emission —
        after the vispy layer applied the staged tile's matrix, before
        anything could draw. The new matrix becomes pending (applied at
        the swap) and the matrix matching the still-rendered front
        content is restored, so the old tile never renders misplaced.
        """
        if self._held_matrix is None:
            return
        transform = self._node_matrix_transform()
        if transform is None:
            return
        current = np.asarray(transform.matrix)
        if np.array_equal(current, self._held_matrix):
            return
        self._pending_matrix = np.array(current, copy=True)
        transform.matrix = self._held_matrix

    def _apply_pending_transform(self) -> None:
        """Apply the captured matrix at the swap (content now matches).

        Skipped when something other than the captured ``set_data``
        emission changed the matrix while holding — the external
        change stands.
        """
        transform = self._node_matrix_transform()
        if (
            transform is not None
            and self._pending_matrix is not None
            and self._held_matrix is not None
            and np.array_equal(np.asarray(transform.matrix), self._held_matrix)
        ):
            transform.matrix = self._pending_matrix
        self._pending_matrix = None
        self._held_matrix = None

    def _drop_transform_hold(self) -> None:
        """Forget the hold without touching the node (fallback paths)."""
        self._pending_matrix = None
        self._held_matrix = None


class _DoubleBufferedTexture(_TransformHoldMixin):
    """Shared front/back texture bookkeeping for the 2D and 3D variants.

    Subclasses set :attr:`_spatial_ndim` (2 for images, 3 for volumes) and
    provide the GL-API-specific pieces (``__init__``, ``_make_sibling``,
    ``present``, ``_catch_up``, ``_bind``, the stage/reshape family,
    ``attach_set_data`` and ``close``). Everything here — the
    retired-texture pool, the present veto, the patch-log dirty/trim
    bookkeeping, and the ``set_data`` detach — is identical across both.
    """

    #: number of leading spatial axes (a channel/dtype axis may follow)
    _spatial_ndim: int = 3

    def _sync_aux_state(self, tex):
        front = self._front
        if front.clim is not None:
            with contextlib.suppress(Exception):
                tex.set_clim(front.clim)
        if tex.interpolation != front.interpolation:
            tex.interpolation = front.interpolation
        return tex

    # -- retired-texture pool --

    def _pool_key(self, shape, dtype) -> tuple:
        # vispy texture shapes carry an explicit channel dim (scalar
        # textures end in 1); data shapes may omit it. Normalize so a
        # retired multichannel texture is never reused for a scalar tile
        # of the same spatial shape (mismatched internalformat).
        nd = self._spatial_ndim
        shape = tuple(int(s) for s in shape)
        channels = shape[nd] if len(shape) > nd else 1
        return (shape[:nd], channels, np.dtype(dtype).str)

    def _acquire(self, shape, dtype, create):
        """Reuse a retired texture of this shape/dtype, or create one."""
        key = self._pool_key(shape, dtype)
        for i in range(len(self._pool) - 1, -1, -1):
            if self._pool[i][0] == key:
                _, tex = self._pool.pop(i)
                return self._sync_aux_state(tex)
        return create()

    def _release(self, texture) -> None:
        """Retire a texture for reuse (delete only past the pool cap)."""
        dtype = getattr(texture, '_data_dtype', None)
        if self._pool_max <= 0 or dtype is None:
            with contextlib.suppress(Exception):
                texture.delete()
            return
        key = self._pool_key(texture.shape, dtype)
        self._pool.append((key, texture))
        while len(self._pool) > self._pool_max:
            _, old = self._pool.pop(0)
            with contextlib.suppress(Exception):
                old.delete()

    @property
    def shape(self) -> tuple:
        """The shape staged content must have (the pair's tile shape).

        During a pending reshape this is already the NEW shape — the
        old-shape front keeps rendering, but patches target the new
        back texture.
        """
        return self._shape

    def matches(self, node) -> bool:
        """Whether this buffer still belongs to ``node``'s texture pair."""
        if self._node is not node or node._texture not in (
            self._front,
            self._back,
        ):
            return False
        # a pending reshape legitimately renders the old-shape front;
        # otherwise an in-place resize of the bound texture (vispy
        # reuses the object) invalidates the pair
        nd = self._spatial_ndim
        return (
            self._reshape_pending
            or tuple(node._texture.shape[:nd]) == self._shape
        )

    # -- presentation bookkeeping --

    @property
    def dirty(self) -> bool:
        """Whether the front texture is behind the staged content."""
        return self._applied[id(self._front)] < len(self._log)

    def _trim_log(self) -> None:
        applied_min = min(self._applied.values())
        if applied_min:
            self._log = self._log[applied_min:]
            for key in self._applied:
                self._applied[key] -= applied_min

    # -- loader present veto --

    def hold_presents(self, timeout: float = 1.5) -> None:
        """Veto presents while staged content is known to be junk.

        Keeps the front texture on screen (e.g. during a time-step
        change where the new step's data has not arrived yet).
        Deadline-bounded so a lost release can never starve presents.
        """
        self._present_hold_until = time.monotonic() + timeout

    def release_presents(self) -> None:
        self._present_hold_until = 0.0

    # -- full-refresh interception --

    def suppress_next_full_upload(self) -> None:
        """Skip the next same-shape full rewrite through ``set_data``.

        One-shot; cleared by the next ``set_data`` whether suppressed
        or not.
        """
        self._suppress_full = True

    def detach_set_data(self) -> None:
        if self._wrapped_set_data is not None:
            self._node.set_data = self._wrapped_set_data
            self._wrapped_set_data = None


class DoubleBufferedVolumeTexture(_DoubleBufferedTexture):
    """Manage front/back 3D textures for a vispy ``VolumeVisual``.

    Parameters
    ----------
    node : vispy.visuals.VolumeVisual
        The volume node to manage. Its current texture becomes the
        front texture; a sibling back texture of the same class,
        format and interpolation is created for staging.

    """

    _spatial_ndim = 3

    def __init__(self, node, pool: list | None = None):
        self._node = node
        self._front = node._texture
        # the shape the pair was built for: a later in-place resize of
        # the front (vispy reuses the texture object) must invalidate
        # the pair, so don't read front.shape live
        self._shape = tuple(self._front.shape[:3])
        # retired-texture pool: list of (key, texture), most recent
        # last; reused by _acquire instead of delete + reallocate.
        # Accepted from a predecessor pair so rebuilds reuse textures.
        self._pool: list[tuple] = pool if pool is not None else []
        self._pool_max = DEFAULT_TEXTURE_POOL_SIZE
        self._back = self._acquire(
            self._front.shape,
            getattr(self._front, '_data_dtype', None) or np.float32,
            lambda: self._make_sibling(node, self._front),
        )
        # patch log: list of (offset, data) staged since creation; each
        # texture tracks how much of the log it has applied. 'full'
        # entries reset the log (older patches are superseded).
        self._log: list[tuple] = []
        self._applied = {id(self._front): 0, id(self._back): 0}
        self._wrapped_set_data = None
        self._suppress_full = False
        # a staged shape change: the back texture already has the new
        # shape/content while the old-shape front keeps rendering until
        # the new texture's uploads have drained (or the deadline hits)
        self._reshape_pending = False
        self._reshape_deadline = 0.0
        # transform coordination: while a full rewrite/reshape is
        # staged, the node matrix matching the still-rendered FRONT
        # content is held; the matrix napari applies for the staged
        # tile (its origin/scale change in the same set_data emission)
        # is captured by the loader and applied at the swap, so texture
        # content and tile transform always change in the same frame.
        self._held_matrix = None
        self._pending_matrix = None
        # rolling drain deadline: EVERY staged write (patch, full,
        # reshape, and the sibling replay after a reshape swap) is
        # metered, so a swap before the drain renders whatever the back
        # held before — stale/partial content for a frame or two. Every
        # present waits for the meter to empty (deadline-bounded); the
        # loader retries presents from the drain callback, so swaps
        # naturally land the moment uploads settle.
        self._stage_deadline = 0.0
        # loader veto: presents held while the staged interval has no
        # real content yet (zeros + carry-over ahead of the repair
        # worker); deadline-bounded so presents can never starve
        self._present_hold_until = 0.0

    # -- construction helpers --

    @staticmethod
    def _seed_texture(node, like, texture_format, rep):
        """Create a texture for seed data ``rep``, multichannel-aware.

        vispy's ``VolumeVisual._create_texture`` hardcodes
        ``format='luminance'`` (its placeholder is scalar), so it can
        never build a texture for an RGB(A) seed: a scalar seed fails
        the internalformat's channel check and a multichannel seed
        fails the luminance format check. Multichannel seeds construct
        the front's texture class directly, letting vispy infer the
        format from the seed's channel count.
        """
        if rep.ndim == 3 or rep.shape[-1] == 1:
            return node._create_texture(texture_format, rep)
        prefix = 'rgba'[: rep.shape[-1]]
        if isinstance(texture_format, str) and texture_format.startswith(
            prefix
        ):
            # GPUScaledTexture derives the multichannel internalformat
            # by replacing the leading 'r' of a scalar format ('r8' ->
            # 'rgb8'); an already-resolved multichannel string comes
            # out garbled ('rgbgb8'), so hand it the scalar form
            texture_format = 'r' + texture_format[len(prefix) :]
        return type(like)(
            rep,
            interpolation=like.interpolation,
            internalformat=texture_format,
            wrapping='clamp_to_edge',
        )

    @classmethod
    def _make_sibling(cls, node, front):
        """Create a back texture matching the front's class and format."""
        from vispy.visuals._scalable_textures import GPUScaledTextureMixin

        if isinstance(front, GPUScaledTextureMixin):
            # the resolved format (e.g. 'r8'), not 'auto': the pair must
            # stay format-identical for patches to mean the same thing
            texture_format = front.internalformat
        else:  # pragma: no cover - napari uses GPU-scaled textures
            texture_format = None
        dtype = getattr(front, '_data_dtype', None) or np.float32
        # Include the channel dimension when the front texture has one
        # beyond the implicit scalar 1 (e.g. RGB volumes are
        # (D, H, W, 3)); vispy validates seed channels against the
        # internalformat.
        nch = front.shape[3] if len(front.shape) > 3 else 1
        seed_shape = (1, 1, 1) if nch == 1 else (1, 1, 1, nch)
        rep = np.zeros(seed_shape, dtype=dtype)
        back = cls._seed_texture(node, front, texture_format, rep)
        # Route the new texture's GLIR commands into the canvas's shared
        # queue NOW. A fresh texture otherwise parks its allocation and
        # every staged upload in its own local queue until the first
        # bind merges them — bypassing the upload meter's pending
        # accounting (the drain gate sees zero and swaps onto a
        # partially uploaded texture: one-time garbage frames at first
        # level switches) and landing as a single unmetered burst.
        front.glir.associate(back.glir)
        # allocate at full size now (SIZE only, no pixel upload): offset
        # patches require an allocated texture
        back.resize(tuple(front.shape), internalformat=front.internalformat)
        back._data_dtype = dtype
        if front.clim is not None:
            back.set_clim(front.clim)
        if back.interpolation != front.interpolation:
            back.interpolation = front.interpolation
        return back

    def _make_texture_like(self, node, vol):
        """Create a texture formatted for ``vol`` (new shape and dtype)."""
        texture_format = getattr(node, 'texture_format', None)
        if texture_format is None:  # pragma: no cover - napari sets it
            texture_format = self._front.internalformat
        seed_shape = (1, 1, 1) + vol.shape[3:]
        rep = np.zeros(seed_shape, dtype=vol.dtype)
        tex = self._seed_texture(node, self._front, texture_format, rep)
        # shared GLIR queue from birth — see _make_sibling for why
        self._front.glir.associate(tex.glir)
        # full data shape: RGB(A) volumes keep their channel dimension
        tex.resize(tuple(vol.shape))
        tex._data_dtype = vol.dtype
        return self._sync_aux_state(tex)

    # -- staging --

    def stage(self, offset, data) -> None:
        """Stage a sub-region update; uploaded to the back texture now.

        ``data`` is retained until both textures have applied it (the
        log drains at each :meth:`present`).
        """
        self._log.append((tuple(offset), data, None))
        self._stage_deadline = max(
            self._stage_deadline, time.monotonic() + 0.75
        )
        self._catch_up(self._back)

    def stage_full(self, data, clim=None) -> None:
        """Stage a full-volume rewrite (e.g. a pass-boundary backdrop)."""
        self._begin_transform_hold()
        self._stage_deadline = time.monotonic() + 2.0
        # a full write supersedes everything staged before it
        self._log = [(None, data, clim)]
        for key in self._applied:
            self._applied[key] = 0
        self._catch_up(self._back)

    def stage_reshape(self, vol, clim=None) -> None:
        """Stage a full rewrite at a NEW tile shape.

        A fresh texture is allocated and filled off the rendered path;
        the old-shape front keeps rendering its (valid, previous-level)
        content until :meth:`present` swaps — once the new texture's
        uploads have drained, or after a deadline. This removes the
        last write-to-bound-texture path: previously a shape change
        fell through to vispy's set_data, which re-specifies and
        re-uploads the texture the GPU is sampling.
        """
        node = self._node
        self._begin_transform_hold()
        new_back = self._acquire(
            vol.shape,
            vol.dtype,
            lambda: self._make_texture_like(node, vol),
        )
        old_back = self._back
        if old_back is not self._front:
            self._release(old_back)
        self._back = new_back
        self._shape = tuple(vol.shape[:3])
        self._log = [(None, vol, clim)]
        self._applied = {id(self._front): 0, id(new_back): 0}
        self._reshape_pending = True
        self._reshape_deadline = time.monotonic() + 2.0
        self._catch_up(new_back)

    def _catch_up(self, texture) -> None:
        key = id(texture)
        start = self._applied[key]
        for offset, data, clim in self._log[start:]:
            if offset is None:
                # full rewrite, through the scaled-texture path so clim
                # normalization stays correct
                if clim is not None:
                    texture.set_clim(clim)
                texture.check_data_format(data)
                texture.scale_and_set_data(data)
            else:
                texture.set_data(data, offset=offset)
        self._applied[key] = len(self._log)

    # -- presentation --

    def present(self) -> bool:
        """Swap the freshly written back texture into the shader.

        Returns True if a swap happened. The new back texture is caught
        up immediately afterwards (still off the rendered path) and the
        drained portion of the log is released.
        """
        if not self.dirty:
            return False
        if time.monotonic() < self._present_hold_until:
            # loader veto: the staged interval has no real content yet
            return False
        if self._reshape_pending:
            return self._present_reshape()
        if not self._uploads_settled(self._stage_deadline):
            # staged writes are still queued in the GLIR meter: binding
            # the back now would render its stale or partially written
            # previous content (the rotation flicker / one-frame jump)
            return False
        front, back = self._front, self._back
        # propagate authoritative front state (napari writes clims and
        # interpolation to node._texture, i.e. the front); clims staged
        # with a full rewrite override this inside _catch_up
        if front.clim is not None and back.clim != front.clim:
            back.set_clim(front.clim)
        if back.interpolation != front.interpolation:
            back.interpolation = front.interpolation
        self._catch_up(back)
        try:
            self._bind(back)
        except RuntimeError:
            return False
        self._front, self._back = back, front

        # catch the new back up too (off the rendered path), then drop
        # the fully-applied prefix of the log to release chunk memory
        self._catch_up(front)
        self._trim_log()
        self._apply_pending_transform()
        return True

    def _present_reshape(self) -> bool:
        """Swap in the new-shape texture once its uploads have drained.

        Swapping earlier would render a partially uploaded (black)
        volume; until then the old-shape front keeps showing the
        previous level. A deadline bounds the wait in case uploads
        never fully settle (e.g. a steady chunk stream).
        """
        if not self._uploads_settled(self._reshape_deadline):
            return False
        node = self._node
        old_front, back = self._front, self._back
        try:
            self._catch_up(back)
            self._bind(back)
            z, y, x = self._shape
            node.shared_program['u_shape'] = (x, y, z)
            node._vol_shape = self._shape
            node._need_vertex_update = True
        except RuntimeError:
            return False
        self._front = back
        self._release(old_front)
        self._reshape_pending = False
        # rebuild the spare at the new shape and converge it (reusing a
        # retired same-shape texture when available)
        self._back = self._acquire(
            back.shape,
            getattr(back, '_data_dtype', np.float32),
            lambda: self._make_sibling(node, back),
        )
        self._applied = {
            id(self._front): self._applied[id(back)],
            id(self._back): 0,
        }
        self._catch_up(self._back)
        # the sibling's full replay just went through the meter: the
        # next (patch-only) present must not bind it before the replay
        # drains — a fresh pool texture would show partial content for
        # a frame (the one-time jump at first level switches)
        self._stage_deadline = time.monotonic() + 2.0
        self._trim_log()
        self._apply_pending_transform()
        with contextlib.suppress(RuntimeError):
            node.update()
        return True

    def _bind(self, texture) -> None:
        """Point the shader at ``texture`` (one sampler rebind)."""
        node = self._node
        node.shared_program['u_volumetex'] = texture
        if getattr(node, '_data_lookup_fn', None) is not None:
            # interpolation lookup samples through this binding; when it
            # is None, vispy's lazy interpolation setup reads
            # node._texture at the next draw instead
            node._data_lookup_fn['texture'] = texture
        node.shared_program['clim'] = texture.clim_normalized
        node._texture = texture

    def _front_pending_full(self) -> bool:
        """Whether the front is behind a staged full rewrite."""
        start = self._applied[id(self._front)]
        return any(
            offset is None for offset, _data, _clim in self._log[start:]
        )

    def _queued_upload_bytes(self) -> int:
        """Bytes of DATA commands queued (pre-flush) for this pair.

        The meter only accounts for uploads it has *deferred*; commands
        still sitting in the GLIR queue (staged since the last flush)
        are invisible to ``pending_upload_bytes``, so a present in that
        window would bind a texture whose rewrite has not run yet.
        """
        try:
            commands = self._front.glir._shared._commands
            ids = {self._front.id, self._back.id}
        except AttributeError:  # pragma: no cover - vispy internals moved
            return 0
        return sum(
            command[3].nbytes
            for command in commands
            if command[0] == 'DATA' and command[1] in ids
        )

    def _uploads_settled(self, deadline: float) -> bool:
        """Whether staged uploads have reached the GPU (deadline-bounded).

        The GLIR meter defers texture uploads across frames; a swap
        before the drain renders whatever the back texture held before.
        The deadline bounds the wait under a sustained upload stream.
        Full rewrites additionally wait out the pre-flush window (their
        upload may not even have been flushed yet, let alone metered);
        small patch batches skip that check — they complete within one
        frame budget, ahead of the draw that samples them.
        """
        from napari.experimental import _glir_metering

        if not _glir_metering.is_installed():
            return True
        if time.monotonic() >= deadline:
            return True
        if _glir_metering.pending_upload_bytes() > 0:
            return False
        return not (
            self._front_pending_full() and self._queued_upload_bytes() > 0
        )

    # -- full-refresh interception --

    def attach_set_data(self) -> None:
        """Route ``node.set_data`` calls through the staging texture.

        napari's slicing pipeline rewrites the whole volume through
        ``node.set_data`` at pass boundaries — a multi-second
        write-to-sampled-texture stall on slow drivers. Same-shape
        rewrites are staged into the back texture; shape changes
        (level/tile switches) are staged into a freshly allocated
        texture and swapped in once uploaded (:meth:`stage_reshape`).
        Only non-array payloads or external texture rebinds fall back
        to the original path.
        """
        if self._wrapped_set_data is not None:
            return
        node = self._node
        original = node.set_data

        def set_data_staged(vol, clim=None, copy=True):
            if (
                not isinstance(vol, np.ndarray)
                or vol.ndim not in (3, 4)
                or (vol.ndim == 4 and vol.shape[3] > 4)
                or node._texture not in (self._front, self._back)
            ):
                # unexpected payload or someone rebound the texture:
                # fall back; the loader rebuilds this buffer on its
                # next patch
                self._suppress_full = False
                self._drop_transform_hold()
                self.detach_set_data()
                return original(vol, clim=clim, copy=copy)
            # a format change — channel count (scalar -> RGB) OR dtype
            # (e.g. uint8 -> uint16 between pyramid levels) — must
            # reshape into a texture of the matching internalformat,
            # never stage_full into the format-mismatched pair (vispy's
            # check_data_format would raise at the deferred present)
            same_format = tuple(
                vol.shape[:3]
            ) == self.shape and not _texture_format_will_change(
                self._back, vol, 3
            )
            if same_format and self._suppress_full:
                # caller asserts the GPU pair already matches vol
                # (every chunk was patched): skip the redundant
                # full-tile upload entirely
                self._suppress_full = False
                node._last_data = vol
                return None
            self._suppress_full = False
            try:
                # presents happen on the loader's cadence (and on the
                # upload-drain callback), never here: the staged upload
                # is metered, so an immediate bind would render the
                # back texture's stale previous content
                if same_format:
                    self.stage_full(vol, clim=clim)
                else:
                    # a level/tile switch or format change: fill a fresh
                    # texture (correct shape AND internalformat) off the
                    # rendered path; the swap happens once its uploads
                    # drain (the old level renders meanwhile)
                    self.stage_reshape(vol, clim=clim)
            except Exception:  # noqa: BLE001 - dtype/format change
                self._drop_transform_hold()
                self.detach_set_data()
                return original(vol, clim=clim, copy=copy)
            node._last_data = vol
            return None

        node.set_data = set_data_staged
        self._wrapped_set_data = original

    def close(self) -> None:
        """Restore the node and release the spare texture.

        The node keeps rendering whatever is currently front. Any
        matrix captured for an unpresented rewrite is applied — going
        forward napari writes the texture directly, so its latest
        matrix is the right one.
        """
        self.detach_set_data()
        with contextlib.suppress(Exception):
            self._apply_pending_transform()
        self.release_presents()
        self._log = []
        if self._back is not self._front:
            with contextlib.suppress(Exception):  # pragma: no cover
                self._back.delete()
        for _key, tex in self._pool:
            with contextlib.suppress(Exception):  # pragma: no cover
                tex.delete()
        self._pool = []


class DoubleBufferedImageTexture(_DoubleBufferedTexture):
    """Manage front/back 2D textures for a vispy ``ImageVisual``.

    Same rationale as :class:`DoubleBufferedVolumeTexture`: chunk
    patches and full rewrites go to an unbound *back* texture and
    :meth:`present` swaps the sampler binding, so the texture the GPU is
    sampling is never written. One structural difference: a shape
    change (level/tile switch) binds its freshly filled texture
    *immediately* instead of waiting for uploads to drain — napari
    updates the node transform synchronously with ``set_data``, so
    keeping the old-shape front on screen would render the previous
    tile under the new tile's transform (misplaced content). 2D tiles
    are small enough that the staged-then-bound upload is the cheap
    part; what matters is that it never re-specifies a bound texture.

    Exception: while a present hold is active (e.g. a time-step change
    where the freshly sliced content is known junk), a shape change is
    kept pending instead — the old-shape front stays on screen with its
    matching held matrix (transform hold), and the swap applies content,
    quad geometry and the captured matrix together.

    Parameters
    ----------
    node : vispy.visuals.ImageVisual
        The image node to manage. Its current texture becomes the
        front texture; a sibling back texture of the same class,
        format and interpolation is created for staging.
    """

    _spatial_ndim = 2

    def __init__(self, node, pool: list | None = None):
        self._node = node
        self._front = node._texture
        self._shape = tuple(self._front.shape[:2])
        self._pool: list[tuple] = pool if pool is not None else []
        self._pool_max = DEFAULT_TEXTURE_POOL_SIZE
        self._back = self._acquire(
            tuple(self._front.shape),  # full shape: keep channels in the key
            getattr(self._front, '_data_dtype', None) or np.float32,
            lambda: self._make_sibling(node, self._front),
        )
        self._log: list[tuple] = []
        self._applied = {id(self._front): 0, id(self._back): 0}
        self._wrapped_set_data = None
        self._suppress_full = False
        # exempt the pair from upload metering: every write here targets
        # an unbound texture and becomes visible only through an atomic
        # swap — a deferred (carried) upload would present a partially
        # written texture as a black flash. The pair is stable (reshape
        # resizes in place), so the ids never change.
        self._exempt_ids = set()
        self._present_hold_until = 0.0
        # a shape change staged while presents were held: the back
        # already has the new shape/content, the old-shape front keeps
        # rendering (with its held matrix) until present() swaps
        self._reshape_pending = False
        self._pending_geometry = None
        self._held_matrix = None
        self._pending_matrix = None
        from napari.experimental import _glir_metering

        for tex in (self._front, self._back):
            glir_id = getattr(tex, 'id', None)
            if glir_id is not None:
                _glir_metering.add_unmetered_texture(glir_id)
                self._exempt_ids.add(glir_id)

    # -- construction helpers --

    @staticmethod
    def _make_sibling(node, front):
        """Create a back texture matching the front's class and format."""
        dtype = getattr(front, '_data_dtype', None) or np.float32
        front_shape = tuple(front.shape)
        nch = front_shape[2] if len(front_shape) > 2 else 1
        if nch in (3, 4):
            rep = np.zeros((1, 1, nch), dtype=dtype)
        else:
            rep = np.zeros((1, 1), dtype=dtype)
        # Let vispy infer the format from the data shape rather than
        # copying the front's internalformat: the property can return
        # a corrupted string when read concurrently with GL work.
        back = node._init_texture(rep, None)
        back.resize(
            front_shape,
            internalformat=back.internalformat,
        )
        back._data_dtype = dtype
        if front.clim is not None:
            back.set_clim(front.clim)
        if back.interpolation != front.interpolation:
            back.interpolation = front.interpolation
        return back

    # -- staging --

    def stage(self, offset, data) -> None:
        """Stage a sub-region update; uploaded to the back texture now."""
        self._log.append((tuple(offset), data))
        self._catch_up(self._back)

    def stage_full(self, data) -> None:
        """Stage a full-image rewrite (e.g. a pass-boundary backdrop)."""
        # if the present is vetoed and napari moves the tile matrix in
        # this same emission, the loader captures it; the front keeps
        # its matching matrix until the swap
        self._begin_transform_hold()
        self._log = [(None, data)]
        for key in self._applied:
            self._applied[key] = 0
        self._catch_up(self._back)

    def _catch_up(self, texture) -> None:
        key = id(texture)
        start = self._applied[key]
        for offset, data in self._log[start:]:
            if offset is None:
                texture.check_data_format(data)
                texture.scale_and_set_data(data)
            else:
                texture.set_data(data, offset=offset)
        self._applied[key] = len(self._log)

    # -- presentation --

    def present(self) -> bool:
        """Swap the freshly written back texture into the shader."""
        if not self.dirty:
            return False
        if time.monotonic() < self._present_hold_until:
            return False
        if self._reshape_pending:
            return self._present_reshape()
        front, back = self._front, self._back
        if front.clim is not None and back.clim != front.clim:
            back.set_clim(front.clim)
        if back.interpolation != front.interpolation:
            back.interpolation = front.interpolation
        self._catch_up(back)
        self._bind(back)
        self._front, self._back = back, front
        self._catch_up(front)
        self._trim_log()
        self._apply_pending_transform()
        return True

    def _present_reshape(self) -> bool:
        """Bind the new-shape back texture once the hold has released.

        The held reshape kept the old-shape front on screen (with its
        matching held matrix); this swap applies tile content, quad
        geometry and the captured matrix in the same frame.
        """
        node = self._node
        old_front, back = self._front, self._back
        self._catch_up(back)
        self._bind(back)
        self._front, self._back = back, old_front
        self._reshape_pending = False
        if self._pending_geometry is not None:
            self._apply_geometry(self._pending_geometry)
            self._pending_geometry = None
        # the old front is unbound now; the log replay resizes it to
        # the new shape (scale_and_set_data resizes on shape mismatch)
        self._catch_up(old_front)
        self._trim_log()
        self._apply_pending_transform()
        with contextlib.suppress(RuntimeError):
            node.update()
        return True

    def _bind(self, texture) -> None:
        """Point the shader at ``texture`` (one sampler rebind)."""
        node = self._node
        if node._data_lookup_fn is not None:
            node._data_lookup_fn['texture'] = texture
        # keep the clim uniform consistent with the bound texture (the
        # color transform reads it at build time, not per draw); absent
        # until the first _build_color_transform — then the build reads
        # node._texture itself
        with contextlib.suppress(Exception):
            node.shared_program.frag['color_transform'][1]['clim'] = (
                texture.clim_normalized
            )
        node._texture = texture

    # -- full-refresh interception --

    def attach_set_data(self) -> None:
        """Route ``node.set_data`` calls through the staging textures.

        Same-shape rewrites are staged into the back texture and
        swapped in; shape changes (level/tile switches) re-spec the
        unbound back texture in place, fill it, and bind it
        immediately. Non-array or multichannel payloads, and dtype or
        format changes, fall back to the original path.
        """
        if self._wrapped_set_data is not None:
            return
        node = self._node
        original = node.set_data

        def set_data_staged(image, copy=False):
            data = np.asarray(image)
            if (
                data.ndim not in (2, 3)
                or (data.ndim == 3 and data.shape[2] > 4)
                or node._texture not in (self._front, self._back)
            ):
                # unexpected payload or externally rebound texture:
                # fall back; the loader rebuilds this buffer on its
                # next patch
                self._suppress_full = False
                self.detach_set_data()
                return original(image, copy=copy)
            # a format change — channel count OR dtype — cannot be
            # absorbed by the fixed-format pair; reshape into a texture
            # of the matching internalformat instead of stage_full
            same_format = tuple(
                data.shape[:2]
            ) == self._shape and not _texture_format_will_change(
                self._back, data, 2
            )
            if same_format and self._suppress_full:
                # caller asserts the GPU pair already matches the data
                # (every chunk was patched): skip the redundant
                # full-tile upload entirely
                self._suppress_full = False
                node._data = data
                return None
            self._suppress_full = False
            try:
                if same_format:
                    self.stage_full(data)
                    self.present()
                else:
                    self._reshape_now(data)
            except Exception:  # noqa: BLE001 - dtype/format change
                self.detach_set_data()
                return original(image, copy=copy)
            node._data = data
            # we uploaded the content ourselves; vispy must not re-run
            # scale_and_set_data on the (now bound) front at next draw
            node._need_texture_upload = False
            return None

        node.set_data = set_data_staged
        self._wrapped_set_data = original

    def _reshape_now(self, data) -> None:
        """Re-spec the unbound back texture to the new shape and bind it.

        2D tile shapes follow the corner-pixels crop, which varies
        continuously with the camera — pooling by exact shape never
        hits (the macOS profile showed ~2 creates + 2 deletes per
        re-slice, and GL object deletion syncs the pipeline). Resizing
        the existing pair instead is a SIZE re-spec on textures that
        are unbound at the time, with no object churn at all.

        While a present hold is active the bind is deferred instead:
        the old-shape front (and the node state matching it) stays on
        screen and :meth:`_present_reshape` swaps everything at once.
        """
        old_front, old_back = self._front, self._back
        dtype = getattr(old_front, '_data_dtype', None)
        back_shape = tuple(old_back.shape)
        back_channels = back_shape[2] if len(back_shape) > 2 else 1
        data_channels = data.shape[2] if data.ndim == 3 else 1
        if (
            old_back is old_front
            or dtype is None
            or np.dtype(data.dtype) != np.dtype(dtype)
            or data_channels != back_channels
        ):
            # dtype/format change (or degenerate pair): a resized
            # texture would no longer match the scaled format
            raise ValueError('texture pair cannot absorb this reshape')
        internalformat = getattr(old_back, 'internalformat', None)
        new_shape = tuple(data.shape[:2])
        # back is unbound: re-spec and fill it off the rendered path
        # (full data shape so RGB(A) keeps its channel dimension)
        old_back.resize(tuple(data.shape), internalformat=internalformat)
        old_back.check_data_format(data)
        old_back.scale_and_set_data(data)
        self._shape = new_shape
        self._log = [(None, data)]
        if time.monotonic() < self._present_hold_until:
            # Hold active — keep the old front on screen (no black
            # flash) together with its matrix (transform hold) and quad
            # geometry (deferred to _present_reshape). The old front
            # resizes through the log replay once it is unbound.
            self._begin_transform_hold()
            self._applied = {id(old_front): 0, id(old_back): 1}
            self._reshape_pending = True
            self._pending_geometry = new_shape
            return
        self._bind(old_back)
        self._front, self._back = old_back, old_front
        old_front.resize(tuple(data.shape), internalformat=internalformat)
        self._applied = {id(self._front): 1, id(self._back): 0}
        self._reshape_pending = False
        self._pending_geometry = None
        # any matrix held for an earlier vetoed rewrite is stale now:
        # the new tile is bound and napari applies its matrix right
        # after this set_data — that one must stand
        self._drop_transform_hold()
        self._apply_geometry(new_shape)

    def _apply_geometry(self, shape) -> None:
        """Follow a tile shape change in the node's quad and shader state."""
        node = self._node
        node._need_vertex_update = True
        with contextlib.suppress(Exception):
            node.shared_program['image_size'] = tuple(shape[:2])[::-1]
        if node._data_lookup_fn is not None:
            with contextlib.suppress(Exception):
                # kernel-based (e.g. cubic) lookups carry the texture
                # shape as a shader parameter
                if 'shape' in node._data_lookup_fn:
                    node._data_lookup_fn['shape'] = tuple(shape[:2])[::-1]

    def close(self) -> None:
        """Restore the node and release the spare texture."""
        self.release_presents()
        from napari.experimental import _glir_metering

        for glir_id in self._exempt_ids:
            _glir_metering.discard_unmetered_texture(glir_id)
        self._exempt_ids = set()
        self.detach_set_data()
        # going forward napari writes the texture directly, so its
        # latest matrix is the right one
        with contextlib.suppress(Exception):
            self._apply_pending_transform()
        self._log = []
        if self._back is not self._front:
            with contextlib.suppress(Exception):  # pragma: no cover
                self._back.delete()
        for _key, tex in self._pool:
            with contextlib.suppress(Exception):  # pragma: no cover
                tex.delete()
        self._pool = []
