"""
Example demonstrating the histogram feature for contrast limits.

This example shows how to use the histogram visualization when
adjusting contrast limits on an image layer.
"""

import numpy as np

import napari

# Create a sample image with interesting structure
np.random.seed(42)
data = np.random.normal(100, 20, (512, 512)).astype(np.float32)

# Add some bright spots
for _ in range(10):
    x, y = np.random.randint(0, 512, 2)
    data[max(0, x-20):min(512, x+20), max(0, y-20):min(512, y+20)] += 100

# Add some dark regions
for _ in range(5):
    x, y = np.random.randint(0, 512, 2)
    data[max(0, x-30):min(512, x+30), max(0, y-30):min(512, y+30)] -= 50

# Create viewer and add image
viewer = napari.Viewer()
layer = viewer.add_image(data, name='example')

# Instructions
print("=" * 60)
print("Histogram Contrast Limits Example")
print("=" * 60)
print()
print("To see the histogram:")
print("1. Look at the 'contrast limits' slider in the layer controls")
print("2. RIGHT-CLICK on the contrast limits slider")
print("3. A popup will appear showing:")
print("   - A histogram of the current data")
print("   - Yellow lines indicating current contrast limits")
print("   - Enhanced slider controls")
print()
print("Try:")
print("  - Adjusting the slider handles to see limits update")
print("  - Clicking 'reset' to auto-contrast")
print("  - Dragging the range edges to change the available range")
print("=" * 60)

napari.run()
