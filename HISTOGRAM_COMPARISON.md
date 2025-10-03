# Histogram Implementation Comparison

## Overview
Comparing three implementations:
1. **Current napari implementation** (this PR) - Quick integration into contrast limits popup
2. **ndv implementation** - Full-featured, production-ready histogram widget
3. **napari PR #675** (2019, closed) - Early napari histogram attempt

---

## Architecture Comparison

### Current Implementation (Simple & Focused)
```
QtHistogramWidget (Qt wrapper)
  └── HistogramVisual (VisPy mesh + lines)
      ├── Mesh (histogram bars)
      ├── Line (clim indicators)
      └── Line (gamma curve)
```
- **Scope**: Embedded in contrast limits popup
- **Features**: Basic histogram, clim lines, gamma curve, log scale, interactive gamma
- **Integration**: Directly modifies layer.gamma on drag

### ndv Implementation (Full-featured)
```
VispyHistogramWidget (Qt + Model separation)
  ├── HistogramModel (data/computation)
  │   └── Histogram calculations
  ├── VispyHistogramCanvas (VisPy rendering)
  │   ├── PlotWidget (axes, camera, pan/zoom)
  │   ├── Mesh (histogram)
  │   ├── LinePlot (LUT line - clims + gamma)
  │   ├── Markers (gamma handle)
  │   └── Line (highlight indicator)
  └── Mouse interaction handlers
```
- **Scope**: Standalone widget, model-view separation
- **Features**: Full plotting infrastructure, draggable clims, draggable gamma handle, visual feedback, cursor changes
- **Integration**: Bidirectional binding to model with ClimPolicy

### napari PR #675 (Archived, 2019)
```
QtHistogramWidget
  ├── NapariPlotWidget (VisPy PlotWidget wrapper)
  │   ├── HistogramVisual
  │   └── AxisWidgets
  └── QtPlotWidget (Qt wrapper with interaction)
```
- **Scope**: General plotting infrastructure for napari
- **Features**: Full plot widget, axes, multiple plot types
- **Status**: Closed - deemed too broad, API concerns

---

## Feature Comparison

| Feature | Current | ndv | PR #675 |
|---------|---------|-----|---------|
| **Core** |
| Histogram mesh rendering | ✅ | ✅ | ✅ |
| Log scale | ✅ | ✅ | ✅ |
| Clim indicator lines | ✅ | ✅ | ✅ |
| Gamma curve visualization | ✅ | ✅ | ✅ |
| **Interactivity** |
| Drag gamma | ✅ (simple) | ✅ (handle) | ✅ |
| Drag clims | ❌ | ✅ | ✅ |
| Visual cursor feedback | ❌ | ✅ | ✅ |
| Highlight on hover | ❌ | ✅ | ❌ |
| **UI/UX** |
| Embedded in popup | ✅ | ❌ | ❌ |
| Standalone widget | ❌ | ✅ | ✅ |
| Axes/ticks | ❌ | ✅ | ✅ |
| Pan/zoom | ❌ | ✅ (locked Y) | ✅ |
| **Architecture** |
| Model-view separation | ❌ | ✅ | ✅ |
| Reusable plot widget | ❌ | ✅ | ✅ |
| **Performance** |
| Data sampling (>1M) | ✅ | ✅ | ❌ visible |
| Efficient updates | ✅ | ✅ | ✅ |

---

## Key Differences in Design Philosophy

### 1. **Clim Lines vs. LUT Line**

**Current** (separate lines):
```python
# Two separate vertical lines
clim_lines = Line(connect='segments')  # 2 disconnected lines
gamma_line = Line(connect='strip')     # Curve between them
```

**ndv** (unified LUT line):
```python
# Single connected line: left_clim + gamma_curve + right_clim
lut_line = LinePlot(connect='strip')  # All connected
# Points: [clim_min_bottom, clim_min_top, ...gamma_points..., clim_max_top, clim_max_bottom]
```

**Advantage of ndv**: Single visual element, cleaner rendering order, easier to style

---

### 2. **Gamma Handle**

**Current**:
- Drag anywhere on gamma curve
- No visual feedback for drag target
- Simple mouse position → gamma calculation

**ndv**:
```python
gamma_handle = scene.Markers(pos=gamma_handle_pos, size=6)
```
- Explicit draggable handle (marker)
- Shows exactly where to grab
- Cursor changes on hover
- Better UX for discovery

**Advantage of ndv**: More discoverable, clearer interaction model

---

### 3. **Transform Scaling**

**Current**:
- Camera rect directly maps data coordinates
- No transform scaling
- Fixed viewport

**ndv**:
```python
handle_transform = scene.transforms.STTransform()
handle_transform.scale = (max_count / 0.98, 1)
lut_line.transform = handle_transform
```
- Normalizes histogram to 0-1 range
- Scales handles/line to fill view
- More flexible for different data ranges

**Advantage of ndv**: Better handling of extreme data ranges, more robust

---

### 4. **Mouse Interaction**

**Current**:
```python
# Simple: map mouse → calc gamma → set layer.gamma
if self._dragging_gamma:
    pos = tr.imap(event.pos)[:2]
    # Calculate gamma from position
    self.layer.gamma = gamma
```

**ndv**:
```python
# Sophisticated: check what's being grabbed
_grabbed = Grabbable.NONE | CLIM_LOW | CLIM_HIGH | GAMMA
def _find_nearby_node(pos, tolerance=5):
    # Check if near any interactive element
    if near_gamma_handle: return Grabbable.GAMMA
    if near_clim: return Grabbable.CLIM_LOW/HIGH
# Then update model based on what's grabbed
```

**Advantage of ndv**: Can drag multiple different elements, better tolerance checking

---

### 5. **Alignment with Slider**

**Current**:
```python
# Calculate slider groove position
groove_rect = style.subControlRect(QStyle.ComplexControl.CC_Slider, ...)
handle_rect = style.subControlRect(QStyle.SubControl.SC_SliderHandle, ...)
# Apply margins to match
layout.setContentsMargins(10 + left_offset, 0, 10 + right_offset, 0)
```

**ndv**: N/A - standalone widget, doesn't align with slider

**Advantage of current**: Seamless integration with existing UI

---

### 6. **Model Separation**

**Current**:
- Direct layer access: `self.layer.gamma`, `self.layer.contrast_limits`
- Tightly coupled to napari layer

**ndv**:
```python
class Histogram:
    """Pure data model for histogram computation"""
    def set_data(values, bin_edges): ...
    
class VispyHistogramCanvas:
    """View that observes model"""
    model.events.clims.connect(...)
    model.events.gamma.connect(...)
```

**Advantage of ndv**: Reusable, testable, clearer separation of concerns

---

## What We Could Improve

### Short-term (quick wins):
1. **Unified LUT line** - Combine clim_lines + gamma_line into single LinePlot
2. **Visual gamma handle** - Add Markers visual at gamma curve midpoint
3. **Cursor feedback** - Change cursor on gamma hover (Qt setCursor)
4. **Draggable clims** - Add interaction for dragging the clim lines themselves

### Medium-term (quality improvements):
5. **Transform scaling** - Use STTransform to normalize display
6. **Better grabbable detection** - Check tolerance zones for each interactive element
7. **Highlight on hover** - Show which value cursor is over
8. **Percentile markers** - Show common percentiles (1%, 99%, etc.)

### Long-term (architectural):
9. **Model separation** - Extract histogram computation from widget
10. **Reusable plot widget** - Build general PlotWidget base class
11. **Plugin architecture** - Allow custom histogram overlays/annotations

---

## Implementation Recommendations

### For Current PR (Minimal Viable Product):
Keep it simple! The current implementation is great for:
- Quick integration
- Learning/experimentation
- 80% of use cases

**Don't add** (scope creep):
- ❌ Axes/ticks (adds complexity)
- ❌ Pan/zoom (not needed in popup)
- ❌ Model separation (premature abstraction)

**Do consider** (low-hanging fruit):
- ✅ Unified LUT line (cleaner rendering)
- ✅ Visual gamma handle (better UX)
- ✅ Cursor feedback (minimal Qt code)
- ✅ Draggable clims (reuse gamma drag logic)

### Future: Standalone Histogram Widget
If/when napari wants a full histogram widget:
- Fork/adapt ndv's implementation
- Build on existing PlotWidget infrastructure
- Make it a dockable widget, not popup
- Add plugin hooks for customization

---

## Code Patterns Worth Adopting

### 1. Grabbable Enum (from ndv)
```python
class Grabbable(Enum):
    NONE = auto()
    CLIM_LOW = auto()
    CLIM_HIGH = auto()
    GAMMA = auto()

def _find_nearby_node(pos, tolerance=5) -> Grabbable:
    # Returns what user is trying to grab
    ...
```

### 2. Unified LUT Line (from ndv)
```python
# Instead of separate lines, build one connected path
def _update_lut_line(clims, gamma, max_count, npoints=256):
    # Bottom-left clim
    points = [[clims[0], 0]]
    # Top-left clim
    points.append([clims[0], max_count])
    # Gamma curve points
    x = np.linspace(clims[0], clims[1], npoints)
    y = ((x - clims[0]) / (clims[1] - clims[0])) ** gamma * max_count
    points.extend(zip(x, y))
    # Top-right clim  
    points.append([clims[1], max_count])
    # Bottom-right clim
    points.append([clims[1], 0])
    return np.array(points)
```

### 3. Transform-based Scaling (from ndv)
```python
# Normalize histogram height to 0-1, then scale to fill view
handle_transform = scene.transforms.STTransform()
handle_transform.scale = (max_count / 0.98, 1)
gamma_line.transform = handle_transform
```

---

## Conclusion

### Current Implementation Strengths:
- ✅ Simple, focused, easy to maintain
- ✅ Perfect for embedded use case
- ✅ Good enough for 80% of users
- ✅ Easy to understand and modify

### What ndv Does Better:
- ✅ Professional polish (handles, cursors, hover)
- ✅ Better architecture (model separation)
- ✅ More features (drag clims, axes, zoom)
- ✅ Reusable as standalone widget

### What PR #675 Attempted:
- ❓ Too ambitious (general plotting infrastructure)
- ❓ Unclear API boundaries
- ❓ Overlaps with external plotting libraries
- ✅ Good ideas, but too broad for core napari

### Recommendation:
1. **Ship current implementation** as-is for MVP
2. **Add quick wins** (gamma handle, unified line) if time permits
3. **Create issue** for future full-featured histogram widget
4. **Reference ndv** when implementing that future widget

The current implementation hits the sweet spot for a first iteration! 🎯
