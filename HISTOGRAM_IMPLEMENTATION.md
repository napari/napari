# Histogram for Contrast Limits - Implementation Summary

## Overview

This implementation adds a **VisPy-based histogram visualization** to napari's contrast limits popup, inspired by the `ndv` library's approach. The focus is on **performance over visual quality**, making it suitable for interactive use with large datasets.

## Architecture

The implementation consists of three main components:

### 1. **HistogramVisual** (`napari/_vispy/visuals/histogram.py`)
A pure VisPy visual component that renders the histogram using:
- `vispy.scene.visuals.Mesh` for histogram bars (converted from bin counts)
- `vispy.scene.visuals.Line` for contrast limit indicators (yellow vertical lines)

**Key Features:**
- **Performance optimized**: Samples large arrays (>1M elements) to stay responsive
- Removes NaN/Inf values automatically
- Simple mesh-based rendering (adapted from vispy's histogram visual)
- Vertical orientation only (for now)

**API:**
```python
histogram = HistogramVisual(parent=scene, color='#888888')
histogram.set_data(data=array, bins=128, data_range=(min, max))
histogram.set_clims(clims=(vmin, vmax))
```

### 2. **QtHistogramWidget** (`napari/_qt/layer_controls/widgets/qt_histogram_widget.py`)
A Qt widget that wraps the VisPy histogram in a small embedded canvas:
- Creates a `vispy.scene.SceneCanvas` (400x150 pixels)
- Contains a ViewBox with PanZoom camera
- Manages the HistogramVisual instance
- Handles data updates and camera positioning

**Features:**
- Compact size suitable for popups
- Auto-scales camera to data range
- Updates histogram from layer's current displayed slice

### 3. **QContrastLimitsPopup Integration** (`napari/_qt/layer_controls/widgets/qt_contrast_limits.py`)
Modified the existing contrast limits popup to include the histogram:
- Changed layout from horizontal to vertical
- Histogram widget stacked above the range slider
- Connected slider value changes to update histogram indicators
- Reset button updates histogram after auto-contrast

## Design Decisions

### Why This Approach?

1. **VisPy for Performance**: Following napari's existing architecture and ndv's proven approach
   - GPU-accelerated rendering via VisPy
   - Mesh-based bars are faster than individual rectangles
   - Can handle updates in real-time

2. **Embedded in Popup**: Keeps the main UI clean while providing detail on-demand
   - Right-click on contrast slider → see histogram
   - No permanent UI space required
   - Natural workflow: adjust slider while seeing distribution

3. **Simplified vs. ndv**: Focused on the essentials
   - No gamma curve visualization (can be added later)
   - No draggable handles on histogram (use slider instead)
   - Vertical orientation only
   - Basic styling (dark background, gray bars, yellow limit lines)

### Key ndv Concepts Borrowed

- **Mesh-based histogram**: Converting bin counts to mesh vertices/faces
- **Separate visuals**: Histogram mesh + indicator lines as separate visuals
- **Performance sampling**: Don't histogram all 10M+ pixels
- **Data range awareness**: Use contrast_limits_range to set histogram bounds

### Differences from ndv

| Feature | ndv | napari (this impl) |
|---------|-----|-------------------|
| **Integration** | Standalone canvas widget | Embedded in popup |
| **Interactivity** | Draggable clim handles | Slider-based only |
| **Gamma curve** | Full visualization | Not implemented |
| **Orientation** | Vertical & horizontal | Vertical only |
| **Visual polish** | High quality | Minimal (gray bars, simple) |
| **Use case** | Primary viewer | Secondary tool |

## Usage

### For Users

1. Open napari with an image layer
2. Find the "contrast limits" slider in layer controls (right panel)
3. **RIGHT-CLICK** the slider
4. Popup appears with:
   - Histogram showing data distribution
   - Yellow lines indicating current contrast limits
   - Enhanced slider for fine control
   - Reset/range buttons

### For Developers

```python
# The histogram automatically updates when:
# - Layer contrast_limits change
# - Layer contrast_limits_range changes  
# - User clicks "reset"

# Manual updates can be triggered:
popup.histogram_widget.update_histogram()  # Recompute from data
popup.histogram_widget.update_clim_lines()  # Just update indicator lines
```

## Performance Characteristics

- **Large data sampling**: Arrays > 1M elements are downsampled
- **Histogram bins**: Fixed at 128 bins (good balance of detail vs. speed)
- **Update frequency**: Fast enough for real-time slider dragging
- **Memory**: Minimal - only histogram bins stored, not full data

## Future Enhancements

Potential improvements (not implemented yet):

1. **Logarithmic mode**: For data with high dynamic range
2. **Gamma curve visualization**: Show the full LUT curve like ndv
3. **Draggable indicators**: Click-drag yellow lines to adjust limits
4. **Horizontal orientation**: For different layout needs
5. **Colormap preview**: Show how colormap maps to data
6. **Percentile markers**: Visual indicators for common percentiles
7. **Histogram smoothing**: For cleaner visualization
8. **Always-visible option**: Pin histogram to main UI instead of popup

## Testing

Run the example:
```bash
python examples/histogram_contrast_limits.py
```

This creates a sample image with interesting features (bright spots, dark regions) and provides instructions for testing the histogram feature.

## Files Changed/Created

### New Files:
1. `napari/_vispy/visuals/histogram.py` - Core VisPy histogram visual
2. `napari/_qt/layer_controls/widgets/qt_histogram_widget.py` - Qt wrapper widget
3. `examples/histogram_contrast_limits.py` - Demo example

### Modified Files:
1. `napari/_qt/layer_controls/widgets/qt_contrast_limits.py`:
   - Import QtHistogramWidget
   - Modified QContrastLimitsPopup to use vertical layout
   - Added histogram widget above slider
   - Connected update signals

## Comparison to ndv

The ndv implementation is more feature-complete and production-ready. This napari implementation is a "quick and dirty" version that:

✅ **Keeps from ndv:**
- VisPy mesh-based rendering approach
- Performance-first mindset  
- Basic histogram + clim indicators
- Sampling for large data

❌ **Simplifies from ndv:**
- No gamma curve visualization
- No interactive dragging of clims on histogram
- No logarithmic mode
- Simpler visual styling
- Only vertical orientation
- Embedded in popup instead of standalone

This makes sense for napari's current use case where the histogram is a **supplementary tool** rather than a **primary interface** as in ndv.

## Notes

- The implementation uses `# type: ignore` comments in a few places to suppress type checker warnings that are false positives (e.g., vispy's camera property that changes from string to camera object)
- Error handling is defensive - if data access fails, the histogram simply doesn't update rather than crashing
- The histogram computes from the currently displayed slice, not the full 3D volume
