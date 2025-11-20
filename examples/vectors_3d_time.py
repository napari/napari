"""
Display 3D+T vectors
====================

This example demonstrates visualization of particle displacement vectors
over time in 3D.
Simulates a compression experiment where the top boundary moves down while
the bottom remains fixed, causing particle rearrangement and radial flow.

.. tags:: visualization-basic, visualization-advanced
"""

import numpy as np

import napari

np.random.seed(42)

# Parameters
n_particles = 800
n_timepoints = 50
volume_shape = np.array([100, 400, 100])  # Z, Y, X

# Initialize particles throughout the volume
positions = np.random.rand(n_particles, 3) * volume_shape
positions[:, 1] = positions[:, 1] * 0.7 + 50  # Fill middle portion

# Generate positions and displacements over time
all_positions = []
all_vectors = []
all_magnitudes = []

for t in range(n_timepoints):
    # Axial compression: top (low Y) moves down, bottom (high Y) stays fixed
    # Normalized height (1 at top/Y=0, 0 at bottom/Y=max)
    normalized_height = 1 - (positions[:, 1] - 50) / (volume_shape[1] * 0.7)
    normalized_height = np.clip(normalized_height, 0, 1)

    # Vertical displacement: top moves down more (positive Y direction)
    y_displacement = normalized_height * 2.0 * (1 + t * 0.1)

    # Radial expansion increases with compression (material pushed outward)
    center_xz = volume_shape[[0, 2]] / 2
    xz_offset = positions[:, [0, 2]] - center_xz
    xz_dist = np.maximum(np.linalg.norm(xz_offset, axis=1, keepdims=True), 1)

    # Expansion is stronger in the middle, weaker at top/bottom boundaries
    radial_factor = np.sin(normalized_height * np.pi)  # Peak in middle
    xz_displacement = (
        xz_offset / xz_dist * radial_factor[:, np.newaxis] * (0.5 + t * 0.1)
    )

    # Combine: [dz, dy, dx]
    displacement = np.column_stack([
        xz_displacement[:, 0],
        y_displacement,
        xz_displacement[:, 1]
    ])

    # Calculate magnitudes
    mag = np.linalg.norm(displacement, axis=1)
    all_magnitudes.append(mag)

    # Build vector data in napari's expected format: (N vectors, 2, D)
    # For 3D+T data: N vectors, 2 points (start + displacement), D=4 dimensions (t,z,y,x)
    # Shape for this timepoint: (n_particles, 2, 4)
    vectors_t = np.stack([
        np.column_stack([np.full(n_particles, t), positions]),  # Start: (t, z, y, x)
        np.column_stack([np.zeros(n_particles), displacement])  # Displacement: (0, dz, dy, dx)
    ], axis=1)

    # Store current state
    all_positions.append(positions.copy())
    all_vectors.append(vectors_t)

    # Update positions by taking just part of the displacement for each timepoint
    # otherwise, something like a tracks layer would be more useful!
    positions += displacement * 0.1

# Combine all timepoints into final arrays
# Points: shape (n_particles * n_timepoints, 4) with (t, z, y, x)
points = np.vstack([np.column_stack([np.full(n_particles, t), pos])
                    for t, pos in enumerate(all_positions)])

# Vectors: shape (n_particles * n_timepoints, 2, 4)
# First column [N vectors, 0, :] = starting positions (t, z, y, x)
# Second column [N vectors, 1, :] = displacement vectors (0, dz, dy, dx)
vectors = np.vstack(all_vectors)

# Magnitudes for coloring: shape (n_particles * n_timepoints,)
magnitudes = np.concatenate(all_magnitudes)

# Create viewer and add layers
viewer = napari.Viewer()
viewer.add_points(
    points,
    name='Particle Centers',
    size=1,
    opacity=0.5
)
viewer.add_vectors(
    vectors,
    name='Displacement Vectors',
    properties={'magnitude': magnitudes},
    edge_color='magnitude',
    edge_colormap='viridis',
    edge_contrast_limits=(0, np.max(magnitudes)),
    vector_style='arrow',
    opacity=1.0,
)
viewer.dims.ndisplay = 3
viewer.camera.angles = (-140, 77, -57)
viewer.layers['Displacement Vectors'].bounding_box.visible = True

if __name__ == '__main__':
    napari.run()
