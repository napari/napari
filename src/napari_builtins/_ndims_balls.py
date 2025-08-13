from napari.benchmarks.utils import labeled_particles


def labeled_particles2d():
    seed = 275961054812084171
    labels, density, points = labeled_particles(
        (1024, 1024), seed=seed, return_density=True
    )

    return [
        (density, {'name': 'density', 'metadata': {'seed': seed}}, 'image'),
        (labels, {'name': 'labels', 'metadata': {'seed': seed}}, 'labels'),
        (points, {'name': 'points', 'metadata': {'seed': seed}}, 'points'),
    ]


def labeled_particles3d():
    seed = 275961054812084171
    labels, density, points = labeled_particles(
        (256, 512, 512), seed=seed, return_density=True
    )

    return [
        (density, {'name': 'density', 'metadata': {'seed': seed}}, 'image'),
        (labels, {'name': 'labels', 'metadata': {'seed': seed}}, 'labels'),
        (points, {'name': 'points', 'metadata': {'seed': seed}}, 'points'),
    ]
