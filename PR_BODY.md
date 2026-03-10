# References and relevant issues

This PR addresses a common pain point where users (and plugin developers) accidentally delete or convert important layers. No existing issue found, but related discussions have come up around layer protection for plugin-managed layers.

# Description

Adds a `Layer.locked` property to protect layers from accidental deletion, type conversion, splitting, and merging through the UI.

## `Layer.locked`

- **Purpose**: Prevent accidental deletion and destructive operations via the UI
- **UI integration**: Lock/unlock via a padlock icon on each layer row, or through the right-click context menu ("Lock Layer(s)" / "Unlock Layer(s)")
- **Protected operations**: Delete (keyboard/menu), Convert to Labels/Image, Split Stack/RGB, Merge to Stack/RGB
- **Notification**: When a user attempts a blocked operation, an info notification lists the locked layer names
- **Mixed selection**: If some selected layers are locked and some are not, only the unlocked layers are deleted; the locked layers remain with a notification

```python
layer = viewer.add_image(data, name="important")
layer.locked = True  # now protected from UI deletion/conversion
```

## `Layer.lock_permanent`

Additionally, a `lock_permanent` property is provided for programmatic use cases (e.g., plugins) where a layer must remain locked and cannot be unlocked by the user through the UI.

- Setting `lock_permanent = True` automatically sets `locked = True`
- The user cannot unlock the layer via the UI (lock icon click is ignored, "Unlock Layer(s)" menu item is hidden)
- The lock can only be released programmatically:

```python
# Plugin protecting its output layer:
layer.lock_permanent = True

# Later, programmatically releasing:
layer.lock_permanent = False
layer.locked = False
```

> **Note on `lock_permanent`**: This is an optional addition to the core lock feature. If the maintainers feel there are any issues with this addition or prefer a different approach, it can be removed from this PR without affecting the core `locked` functionality.

## UI

Each layer row shows a padlock icon to the left of the layer type icon:
- 🔓 `lock_open` when unlocked (clickable to lock)
- 🔒 `lock` when locked (clickable to unlock, unless `lock_permanent`)

Right-click context menu shows "Lock Layer(s)" or "Unlock Layer(s)" depending on the current state.

## Changes

**13 files changed, +400 lines**

| File | Change |
|------|--------|
| `layers/base/base.py` | `locked` and `lock_permanent` properties with event emission |
| `components/layerlist.py` | Deletion guard in `remove_selected()`, locked event subscription |
| `layers/_layer_actions.py` | Guards on `_convert`, `_split_stack`, `_merge_stack` |
| `_app_model/context/_context_keys.py` | `ContextNamespace.refresh()` for non-selection event updates |
| `_app_model/context/_layerlist_context.py` | Context keys: `any/all_selected_layers_locked`, `any_selected_layers_permanently_locked` |
| `_app_model/constants/_menus.py` | `LOCK` menu group |
| `_app_model/actions/_layerlist_context_actions.py` | Lock/Unlock context menu actions |
| `_qt/containers/qt_layer_model.py` | `LockRole` for Qt model data |
| `_qt/containers/_layer_delegate.py` | Lock icon rendering and click handling |

## Design decisions

- **`locked` / `lock_permanent` excluded from `_get_base_state()`**: Layer subclass constructors (`Image.__init__`, `Labels.__init__`, etc.) do not accept these as keyword arguments. Including them would cause `TypeError` during layer type conversion. Lock state should not carry over to a converted layer.
- **Notification over exception**: Blocked operations show `show_info()` notifications rather than raising exceptions — locking is a protective convenience feature, not a security boundary.
- **`ContextNamespace.refresh()`**: Allows re-evaluating context keys when a layer property changes without a selection change event. This is a general-purpose utility that could benefit other property-driven context keys in the future.

## Tests

14 new test cases across 4 test files:
- `layers/base/_tests/test_base.py` — locked/lock_permanent property behavior (8 tests)
- `components/_tests/test_layers_list.py` — deletion guard, permanent lock (3 tests)
- `_qt/containers/_tests/test_qt_layer_list.py` — LockRole, delegate painting, permanent lock rendering (6 tests)
- `_qt/_qapp_model/_tests/test_layerlist_context_actions.py` — action dispatch

<!-- Final Checklist
- [x] My PR is the minimum possible work for the desired functionality
- [x] I have commented my code, particularly in hard-to-understand areas
- [x] I have made corresponding changes to docstrings and documentation
- [x] I have added tests that prove my fix is effective or that my feature works
- [x] If an API has been modified, I have added a `.. versionadded::` directive to the appropriate docstring
-->
