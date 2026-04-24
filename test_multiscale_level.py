# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "napari[all] @ git+https://github.com/cmalinmayor/napari.git@multiscale-level-lock",
#     "numpy",
# ]
# ///
"""Test script for the locked_data_level feature with the Qt widget.

Creates a 3D multiscale image with 4 resolution levels and opens it in napari.
Each level has a distinct value so you can verify which level is being rendered.
"""

import numpy as np
import napari

# Build a 4-level multiscale pyramid (3D)
shapes = [(128, 256, 256), (64, 128, 128), (32, 64, 64), (16, 32, 32)]
multiscale_data = []
for i, shape in enumerate(shapes):
    # Fill each level with a distinct intensity so it's obvious which is loaded
    arr = np.random.randint(0, 80, size=shape, dtype=np.uint8)
    # Add a bright marker whose size scales with resolution so you can tell levels apart
    s = shape[0] // 4
    arr[s : s * 2, s : s * 2, s : s * 2] = 50 * (i + 1)
    multiscale_data.append(arr)

viewer = napari.Viewer(ndisplay=3)
layer = viewer.add_image(multiscale_data, name="test_multiscale", multiscale=True)

print("Multiscale layer added.")
print(f"  Levels: {[a.shape for a in multiscale_data]}")
print()
print("Use the 'data level' dropdown in the layer controls to lock a level.")
print("  'Auto' = default napari behavior (lowest res in 3D)")
print("  '0: ...' = full resolution")
print("  '3: ...' = coarsest resolution")
print()
print("You can also set it programmatically:")
print("  viewer.layers['test_multiscale'].locked_data_level = 0")
print("  viewer.layers['test_multiscale'].locked_data_level = None  # restore auto")

napari.run()
