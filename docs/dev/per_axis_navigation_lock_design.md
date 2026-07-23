# Per-axis navigation locks (user-facing padlocks)

**Status:** Implemented
**Last updated:** 2026-07-23
**Scope:** napari `Dims` per-axis lock — `src/napari/components/dims.py`, `src/napari/_qt/widgets/qt_dims.py`, `src/napari/_qt/widgets/qt_dims_slider.py`, `src/napari/_qt/qt_resources/styles/02_custom.qss`

> Placement note: napari's `.gitignore` ignores `docs/` (user docs live in the separate
> `napari/docs` repo). This maintainer note is intentionally force-tracked on the
> `feature/dims-navigation-lock` branch to keep the analysis with the code. If this work is
> upstreamed, relocate it alongside [navigation_lock_v2_design.md](navigation_lock_v2_design.md).

## Context

The shipped navigation lock (`lock_navigation`, `NAVIGATION_LOCK_VERSION = 1`) is
**transient and owner-based**: one operation — e.g. Shapes drawing — takes the lock,
names the axes it wants left navigable (`exempt`), and releases it when done.

This note covers a second, **persistent and user-facing** lock: the user pins
individual axes ("lock the `time` slider") and the pin survives until they unpin it.
A padlock button on each slider row makes it directly toggleable.

The two are genuinely different mechanisms — different lifecycles, different
authorities — so they get **separate storage** and compose at one chokepoint.

## Current Decision

### Design principle

> Each lock authority tier gets its own storage; `_axis_movable` is a precedence
> ladder over them.

```python
def _axis_movable(self, axis, *, force=False):
    if force:
        return True                          # trusted internal bypass
    if self._nav_lock_owner is not None:     # an operation owns navigation...
        return axis in self._nav_lock_exempt # ...its exempt set governs
    return not self.axis_locked[axis]        # no owner: sticky user pins govern
```

`is_axis_movable(axis)` is the public form, so views ask one question instead of
re-deriving the precedence.

### The four decisions

1. **API surface: `dims` only.** `lock_axis`, `unlock_axis`, `lock_all_axes`,
   `unlock_all_axes` live on `Dims`; no `ViewerModel` shims. The state belongs to
   `Dims`, and headless code must be able to lock without Qt. Axes are addressable
   by index or by `axis_labels` name (`_normalize_axis`; raises on unknown or
   ambiguous, since labels are not unique).

2. **Padlock is a clickable button in the slider row**, not an overlay on the
   scrollbar thumb. The thumb is drawn by the Qt style, so tracking it would mean
   recomputing `QStyle.subControlRect` geometry on every value/range/resize change
   — a lot of fragility for a decoration. A row button is also *better UX*: it
   toggles rather than merely indicating.

3. **Composition during an owner lock: owner governs.** While a draw holds the
   lock, its exempt set alone decides movability and user pins are *suspended*,
   resuming on release. Rejected alternative: strict AND (a pin can veto the
   draw's request). Owner-governs matches "block everything except what the
   operation specifically asked for", keeps the predicate simple, and avoids a
   stale pin silently hobbling an active tool — which the user could not even
   clear, since the configuration is frozen (below).

4. **Per-axis configuration is frozen during an owner lock.** `lock_axis` and
   friends raise `RuntimeError` while `navigation_locked`, and the padlocks go
   inert. Changing pins mid-draw would be both surprising and ignored.

### Interactivity switch

`axis_lock_interactive: bool = True` gates **only the user click path**, never the
methods. Setting it False and driving `lock_axis` from application code yields
"locks controlled by the viewer, not the user" — padlocks become status
indicators. It lives on the model (not the widget) so app code reaches it through
`viewer.dims` rather than the private `_qt_viewer`.

### Enforcement boundary and the escape hatch

The per-axis lock inherits v1's **method-only** contract exactly. Guarded:
`set_point`, `set_current_step`, `_increment_dims_*`, and every UI path that
funnels through them (slider drag, slice-number editor, playback, wheel, arrow
keys). `_focus_up`/`_focus_down` also skip non-movable axes.

**Not** guarded, deliberately: direct field/property assignment
(`dims.point = ...`, `dims.current_step = ...`) and `force=True`. That direct path
is the **intended programmatic escape hatch** — the lock guards deliberate
*navigation*, not raw coordinate writes. Two internal lifecycle calls rely on it,
and both run exactly when the coordinate space is being rebuilt:

| Call | Site | Fires when |
|---|---|---|
| `dims.reset()` | `viewer_model.py` `_on_layers_change` | last layer removed (`ndim` → 2) |
| `dims._go_to_center_step()` | `viewer_model.py` `_add_layer_from_data` | first layer added |

Neither is reachable by navigating an existing viewer, so the user-facing promise
holds: **the padlock holds against everything a user can do in the UI.**
`test_per_axis_lock_does_not_guard_direct_assignment` pins this contract.

### UI enablement is per child

`QtDims._on_navigation_lock` delegates to `QtDimSliderWidget._update_lock_state`,
which disables *each control that moves the slice* (`slider`, `play_button`,
`curslice_label`) rather than the whole row — the padlock must stay clickable on
an axis whose navigation is frozen. `axis_label` stays enabled: it renames the
axis, it does not navigate.

### The active slider must be one you can move

`last_used` marks the active slider (its handle is drawn in the theme's
`current` colour). `_check_dims` already moved it when it stopped being
*visible*; it now also moves it when it stops being *movable*, so locking the
active axis hands focus to a movable one and locking an inactive axis does not
steal focus. When every slider is locked it falls back to the visible sliders,
so `last_used` always names a real slider — and unlocking one then makes it
active, because it becomes the only candidate. `_focus_up`/`_focus_down` skip
non-movable axes on the same reasoning.

The disabled-handle colour must not be `foreground`: that is the groove's own
background, so the thumb would vanish into the track instead of reading as
greyed out. It is `primary`, which sits between the groove and the normal
`secondary` handle in both themes. The `last_used` rule is an attribute
selector and outranks a bare `:disabled`, so the disabled rule repeats it to
keep a locked *active* slider greying out rather than staying marked current.

### Locking an axis must stop its playback

Disabling the play button is not enough. `QtDims._set_frame` clears
`Dims._play_ready` *before* calling `set_current_step`, and `_play_ready` is
restored only by a canvas draw (`_vispy/canvas.py`). A blocked write emits no
event, so no draw ever happens and playback hangs unrecoverably — with its play
button now disabled, the user cannot even stop it. In tests the stranded
animation thread survives into teardown and aborts the process.

Both halves are therefore guarded:

- `QtDims._on_navigation_lock` stops playback eagerly when the animated axis
  becomes non-movable (either lock tier).
- `QtDims._set_frame` refuses to issue a write for a non-movable axis, stopping
  instead — so `_play_ready` is never cleared for a write that cannot land.

Covered by `test_padlock_stops_active_playback` and
`test_owner_lock_stops_active_playback`.

## Alternatives Considered

- **Fold per-axis locks into `_nav_lock_exempt`.** Rejected: the exempt set is
  "everything locked *except* these" scoped to one transient owner; user pins are
  "*these* locked" and sticky. Different lifecycles, so different storage.
- **`locked_axes: tuple[int, ...]` (a set of indices)** instead of a per-axis bool
  tuple. Rejected: indices go stale on every `ndim` change. A length-`ndim` bool
  tuple normalized by `ensure_len` in `_check_dims` (like `rollable`) left-pads
  with the other per-axis tuples, so a lock tracks its axis across `ndim` changes.
- **Interactivity switch on `QtDims`** instead of the model. Rejected: app code
  would have to reach through `viewer.window._qt_viewer`, which napari discourages.

## Deferred Work

**Mandatory spatial-axis lock** — guaranteeing the spatial/slice dimensions stay
locked *regardless of user settings*, for 3D spatial datasets.

Blocked on a prerequisite napari lacks: a notion of axis **kind** (spatial vs
non-spatial). Axes are positional tuples with labels/units/world-coords; there is
no first-class flag to key the policy off. The hard part is identifying *which*
axes the guarantee covers, not the lock mechanism. Likely entangled with the
units/world-coordinate work.

It slots in as a **third tier** with its own storage:

```python
if self._axis_locked_mandatory[ax]:   # FUTURE: policy lock
    return False                       # unlock_axis() cannot reach it
return not self.axis_locked[ax]        # user pins (today)
```

The groundwork is already deliberate: **`unlock_axis` operates only on the
`axis_locked` tier**, never on another, so "regardless of user settings" falls out
for free once the tier exists.

Open sub-decision when built: precedence of the mandatory tier vs. the owner lock.
Moot in the common case — a draw will not exempt spatial axes, so they stay frozen
during a draw under owner-governs anyway.

## Next Steps

None for v1; the feature is implemented and tested. Revisit this note when the
axis-kind concept lands, or if the method-only boundary
([navigation_lock_v2_design.md](navigation_lock_v2_design.md)) is ever revisited —
a field-level chokepoint would automatically strengthen the per-axis lock too.
