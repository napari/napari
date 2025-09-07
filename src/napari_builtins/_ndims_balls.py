import numpy as np

from napari.benchmarks.utils import labeled_particles


def labeled_particles2d():
    seed = np.random.default_rng().integers(np.iinfo(np.int64).max)
    labels, density, points = labeled_particles(
        (1024, 1024), seed=seed, return_density=True
    )

    return [
        (
            density.copy(),
            {'name': 'density', 'metadata': {'seed': seed}},
            'image',
        ),
        (
            labels.copy(),
            {'name': 'labels', 'metadata': {'seed': seed}},
            'labels',
        ),
        (
            points.copy(),
            {'name': 'points', 'metadata': {'seed': seed}},
            'points',
        ),
    ]


def labeled_particles3d():
    seed = np.random.default_rng().integers(np.iinfo(np.int64).max)
    labels, density, points = labeled_particles(
        (256, 512, 512), seed=seed, return_density=True
    )

    return [
        (
            density.copy(),
            {'name': 'density', 'metadata': {'seed': seed}},
            'image',
        ),
        (
            labels.copy(),
            {'name': 'labels', 'metadata': {'seed': seed}},
            'labels',
        ),
        (
            points.copy(),
            {'name': 'points', 'metadata': {'seed': seed}},
            'points',
        ),
    ]
