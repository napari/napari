# Field-level navigation-lock chokepoint (NAVIGATION_LOCK_VERSION 2)

**Status:** Deferred
**Last updated:** 2026-07-19
**Scope:** napari `Dims` navigation lock — `src/napari/components/dims.py`; interacts with `src/napari/utils/events/evented_model.py` and `src/napari/components/viewer_model.py`

> Placement note: napari's `.gitignore` ignores `docs/` (user docs live in the separate
> `napari/docs` repo). This maintainer note is intentionally force-tracked on the
> `feature/dims-navigation-lock` branch to keep the analysis with the code. If this work is
> upstreamed, relocate it to `napari/docs` (or wherever maintainers keep dev notes).

## Context

v1 (`NAVIGATION_LOCK_VERSION = 1`, on `feature/dims-navigation-lock`) guards navigation
**through methods only**: `set_point` (per-axis `exempt`, `force` bypass), `roll` /
`transpose` (when `lock_order`), and everything funnelling through them (sliders, wheel,
arrow keys, play, the 2D/3D toggle). It deliberately does **not** guard *direct field /
property assignment* — `dims.point = ...`, `dims.current_step = ...` (property → writes
`point`), `dims.order = ...`, `dims.ndisplay = ...` — because the internal validator
`_check_dims` also uses field assignment to normalize state, and v1 draws no line between
the two.

v1 covers 100% of first-party GUI navigation. The residual gap is **raw field
assignment**, reachable by: internal resets (`reset`, `_go_to_center_step`), data-driven
`range`/`ndim` updates, saved-state restore, and any **external/programmatic** code that
assigns nav fields directly (the motivating case: a downstream `ImageScroller` that pokes
`dims.current_step`).

The v2 idea: `Dims` is an `EventedModel` with `validate_assignment=True`, so
`EventedModel.__setattr__` (evented_model.py:255) is *already* the single funnel for every
field write, and the validator's own re-writes are already flagged by `self._validating`.
No new "forwarders" are needed — the chokepoint exists; it just isn't lock-aware.

A first sketch (guard `point`/`order`/`ndisplay`; add a `_navigating` flag that vetted
methods set so their writes pass; leave `range` open; no-op on a blocked write) was
reviewed by Codex and **failed on two counts** (B1, B2 below). This note records the
corrected design, the open contract decisions it forces, and — importantly — whether v2
is worth building at all.

## Current Decision

**Deferred.** Ship v1 (method-only, with its honest documented contract) now. v2 is a
potential fast-follow, gated on (a) a decision that it is worth building, and (b) explicit
maintainer approval of the two contract changes it requires. This note is the starting
point for that follow-up, not an approval to implement.

## The corrected design

### Guiding invariant

**The lock guards deliberate *navigation*, not the *coordinate space*.** "Navigation" is a
user/programmatic change to *which slice is shown* (`point`) or *how axes are partitioned*
(`order`, `ndisplay`). The coordinate space (`range`, `ndim`) is *data-driven* — it
describes the layers' extent and is rewritten by `ViewerModel._on_layers_change`
(viewer_model.py:807-816) whenever layer data changes, including as a side effect of
drawing a shape. The lock must not freeze the coordinate space, or it breaks the model's
own bookkeeping mid-draw.

### Guarded fields and conditions

| Field | Blocked while locked when… | Rationale |
|---|---|---|
| `point` | always (owner present) | the slice position; `current_step`/wheel/keys all write it |
| `ndisplay` | always (owner present) | 2D/3D partition; matches v1's `_toggle_ndisplay` (blocks whenever `navigation_locked`, independent of `lock_order`) |
| `order` | only when `_nav_lock_order` is True | **B2 fix** — mirror `roll`/`transpose`, which allow order changes when `lock_order=False` |
| `range`, `ndim` | never | **B1 boundary** — data-driven; see below |

`current_step` (property → `point`) and the wheel/keys/sliders are covered transitively via
`point`. `nsteps` (property → `range`) and `set_range`/`set_axis_label` touch unguarded
fields and are unaffected.

### B1 — why `range`/`ndim` stay unguarded, and what that permits by design

Assigning `range` or `ndim` invokes `_check_dims`, which re-clips `point` (dims.py:197-205)
and re-normalizes `order` (dims.py:214-226) **under `_validating`** — an allowed context in
the guard. So a locked `range` narrowing *can* reposition the slice, and a locked `ndim`
change *can* alter point and order. Verified on current code: locked `range=((0,2,1),)*3`
moved `point[0]` off 9; locked `ndim=2` changed point and order.

This is **permitted by design**, and it is consistent: the outermost *guarded* external
write (`dims.point = X`) is rejected *before* the validator runs, so `_validating` only
ever fires downstream of an *allowed* trigger (a `range`/`ndim` write or a vetted method).
The lock therefore blocks deliberate navigation while letting the coordinate space track
the data.

Practical safety for the #9207 Shapes use case: during a draw, data *grows*, so `range`
*expands*, which does not clip `point` — the slice does not move. The pathological case
(data *shrinks* mid-draw, clipping `point`) is a **data race orthogonal to navigation**:
the model must reflect the new extent or become corrupt, so blocking it is wrong. If that
case matters, the correct remedy is at the Shapes layer (cancel/finish the draw when its
data extent changes underneath it), not the navigation lock. Out of scope for v2.

**But there is no source-level distinction** between `ViewerModel`'s trusted
reconciliation write and an *external* caller deliberately shrinking `dims.range` or
setting `dims.ndim` to move the slice past the lock. So "permitted by design" only holds if
the guarantee is stated honestly. Two ways to make it watertight, both needing maintainer
sign-off:

- **(a) Narrow the claim.** Do not call it a "navigation lock." Call it what it is:
  *guards direct writes to `point`/`order`/`ndisplay`*. Document that direct `range`/`ndim`
  assignment is an allowed, unsupported-for-navigation bypass during a lock. Cheapest; no
  new mechanism; but concedes the lock is not a complete navigation guarantee.
- **(b) Close it with a trusted-reconciliation context.** `_on_layers_change` wraps its
  `range`/`ndim` writes in a `_reconciling` context; external `range`/`ndim` writes that
  would alter locked navigation state are blocked. Watertight, but adds a *second* trusted
  bypass flag (`_reconciling`) with the same footgun as `_navigating`, and requires
  `ViewerModel` to mark every reconciliation site — more mechanism, more surface to forget.

Either way this is a contract decision requiring maintainer sign-off, not an implementation
detail. It is the clearest single reason v2 cannot be quietly bundled.

### The `_navigating` context (and its failure mode)

Vetted methods wrap their already-policy-checked field writes so the guard lets them
through, using an **exception-safe** context manager mirroring `_validating_ctx`
(dims.py:705-712):

```python
@contextlib.contextmanager
def _navigating_ctx(self):
    prev = self._navigating
    self._navigating = True
    try:
        yield
    finally:
        self._navigating = prev
```

**Exhaustive mutation inventory** (every write to a guarded field in `dims.py`). Only
methods that already apply a *lock policy* (owner/`force`/`lock_order`/per-axis `exempt`)
may set `_navigating`; everything else must not, or the flag becomes a blanket bypass.

| Site | Field(s) | Handling |
|---|---|---|
| `_check_dims` normalization (202, 226) | point, order | allowed via `_validating` |
| `set_point` (439) | point | policy-checked (exempt/force) → wrap in `_navigating_ctx` after filtering |
| `roll` (571) | order | policy-checked (`lock_order`/force) → keep check, wrap the write |
| `transpose` (508) | order | policy-checked (`lock_order`/force) → keep check, wrap the write |
| `reset` (485-487) | point, order, range | **NOT policy-checked** — see below (pass-1 B2) |
| `_go_to_center_step` (574) | point (via `current_step` setter) | **NOT policy-checked** — see below (pass-1 B2) |
| `current_step` setter (304) | point | never wrapped — external `dims.current_step = X` must block; internal callers wrap at the method level |
| `ndisplay` | — | no in-model method writes it; `ViewerModel._toggle_ndisplay` keeps its `navigation_locked` check for the user message, with the field guard as backstop |

**B2 — `reset` / `_go_to_center_step` must not be blanket-wrapped.** Both are *public*
(or first-party-called) and carry **no** owner/`force`/lock-policy check. Wrapping them in
`_navigating` would let `viewer.dims.reset()` or a center-step silently reposition a locked
`Dims` — an unqualified navigation bypass. But they also cannot simply hit the field guard
and be *partially* blocked: `reset` writes `range`+`point`+`order`+margins+`rollable`
together, so blocking only `point`/`order` yields an inconsistent half-reset. So they need
**all-or-nothing, lock-aware** handling: a method-entry check that no-ops (or requires
`force`) the *whole* method while locked, mirroring `roll`/`transpose`. Their real internal
callers do not run mid-lock anyway — `reset` fires from `_on_layers_change` only when the
last layer is removed (which releases the draw lock via the backstop, viewer_model.py), and
`_go_to_center_step` fires only on the *first* layer add (impossible while a second, drawing
layer holds the lock). That the "fix" is *more* method-level machinery is itself evidence
against v2 (see verdict).

**The footgun (N3):** any *future* method that writes a guarded field must either apply a
lock policy + wrap, or accept being blocked — a bug that only manifests mid-draw. This is
trusted global-bypass state. Mitigations: the inventory above, a test per row, and a
comment on the guard pointing here. An alternative that avoids the global flag is in
Alternatives.

### Semantics of a blocked write (settled)

- **No-op** (not raise), matching `set_point`. Rationale: won't crash callers that assign
  in a `try`-tolerant way; the lock is short-lived (a draw).
- **Coarse (N2):** raw `dims.point = tuple` is all-or-nothing — it is rejected even if only
  an `exempt` axis changed. Per-axis `exempt` is a property of `set_point`, not of raw
  assignment. Documented; not exercised by the `exempt=()` Shapes consumer.

### `EventedModel.update()` (N3)

`update()` (evented_model.py:415-424) iterates keys and calls `setattr` one field at a
time under an event blocker, **with no transaction or rollback**. While locked,
`update({"range": ..., "point": ...})` can apply `range`, let validation clip `point`, then
reject the explicit `point` write — an observable *partial* update whose result depends on
key order. Per pass-1 B3 this is a **public API semantic change and must be resolved before
implementation, not deferred.** `_check_dims` preserves structural invariants, so this is
not verified corruption, but it is a behavior change needing a maintainer-approved contract:
per-field partial application (documented, tested in both key orders), reject-before-mutate
if any guarded field is present, or a lock-aware atomic `Dims` update. Event-emission
behavior must be specified alongside.

## Is v2 worth building at all?

The honest cost/benefit, stated plainly so it can be rejected:

- **Benefit:** closes raw-assignment navigation for external/programmatic callers. First-
  party napari already fully covered by v1.
- **Cost:** additive two-tier enforcement (method guards *and* field chokepoint *and* a
  trusted bypass flag), a hot-path change on `Dims.__setattr__` needing a benchmark, two
  API-contract decisions requiring maintainer approval (B1, B2), and the `_navigating`
  footgun.

A defensible maintainer position is **"don't build it": if you assign `dims.point`
directly, use `set_point` instead — bypassing the API is the caller's bug.** Under that
view the downstream `ImageScroller` fix is one line (call `set_current_step`), and v2 is
not worth the surface area.

**Current recommendation (after refinement pass 1): do not build v2 as drafted.** It cannot
honestly promise complete navigation protection without resolving three maintainer-level
contract decisions (B1 range/ndim, B2 reset/center, B3 `update()`), and every path to
watertight *adds* mechanism (a second `_reconciling` flag, all-or-nothing lock handling on
`reset`/`_go_to_center_step`, a defined `update()` transaction) on top of the `_navigating`
flag and the hot-path `__setattr__` change — all for a benefit v1 already delivers for
first-party navigation. The honest default is: **document v1's method-only contract, steer
callers (including `ImageScroller`) to `set_point`/`set_current_step`, and keep this note as
the rationale should raw-assignment protection ever become a hard requirement.** Only revisit
if maintainers require raw-assignment compatibility strongly enough to accept and specify
B1–B3.

## Alternatives Considered

- **Method-only (v1) — chosen for now.** Covers all first-party navigation; documented
  gap on raw assignment. Zero new mechanism.
- **Explicit private writer instead of a flag.** Methods call `self._nav_write('point', v)`
  (which does `super().__setattr__` directly) and `__setattr__` blocks *all* guarded writes
  when locked with no `_navigating` exception. Trades the global flag for an explicit
  bypass at each call site. Same "forget it → self-block" failure mode, but the trust is
  visible at the write rather than in ambient state. Still needs the `_validating`
  exception for validator normalization.
- **Privatize the fields (`_point`, …) + read-only properties + explicit setters.** The
  literal "forwarders" idea. Rejected: fights pydantic field/validation/serialization
  machinery; large invasive change for no extra safety over the `__setattr__` hook.
- **Raise instead of no-op.** Rejected (see settled semantics).

## Deferred Work / Open decisions (need maintainer approval)

Labelled by topic to avoid collision with the review-round IDs used above.

1. **Worth building?** — the gating verdict: v2 vs. "use the API". Refinement leans *no*
   (see recommendation above).
2. **`range`/`ndim` boundary** — confirm layer-driven `range`/`ndim` changes may reposition
   a slice during a lock (coordinate space ≠ navigation), and choose the honest framing:
   narrow the guarantee's *name*, or add a `_reconciling` trusted context that blocks
   external `range`/`ndim`.
3. **`reset` / `_go_to_center_step` while locked** — choose **no-op vs. a new `force`/owner
   path**, all-or-nothing (not partial), including event-emission behavior. *(This is the
   genuinely open item; do not conflate it with the already-settled guard condition that
   `order` is blocked only when `_nav_lock_order`, which is part of the accepted design, not
   an open question.)*
4. **`update()` semantics** while locked — per-field partial, reject-before-mutate, or
   atomic; specify event behavior; test both key orders.
5. **Flag vs. explicit-writer** for the trusted-bypass mechanism (`_navigating`).
6. **Benchmark** the unlocked `Dims.__setattr__` fast path (representative `set_point`, raw
   `point` assign, layer-change flows). The override is on `Dims`, not shared
   `EventedModel`, so blast radius is `Dims`-scoped, not repo-wide.

## Refinement outcome (2 Codex passes, gpt-5.6-terra, high)

Two passes reached convergence. Pass 1 confirmed the sketch's B1 (range/ndim bypass) and
found a new B2 (blanket-wrapping `reset`/`_go_to_center_step` = unqualified bypass) plus B3
(`update()` partial-application is a public-API change). Pass 2 marked B1/B2/B3 **Resolved**
in this note and confirmed the recommendation, adding only B4 (this decision-list relabel).

**Verdict: do not build v2 as drafted; escalate the contract to maintainers.** Designer and
reviewer agree. v1 already documents direct assignment as an intentional bypass
(`dims.py:116-126`) and fully covers first-party navigation; v2 adds trusted ambient state
and changes public `update()` behavior, and none of B1/B2/B3 can be settled without
maintainer contract decisions. The remaining "open disagreements" are human-decision flags,
not designer/reviewer conflicts.

## Next Steps

1. **Land v1 as-is** (`feature/dims-navigation-lock`) — the shippable, honestly-scoped work.
2. **Fix downstream callers** (e.g. `ImageScroller`) to navigate via
   `set_point`/`set_current_step` rather than raw `dims.current_step =`.
3. Revisit v2 **only** if maintainers require raw-assignment/plugin compatibility strongly
   enough to accept and specify decisions 2–4 above; if so, open a focused v2 PR off
   `feature/dims-navigation-lock` starting from this note.
