from napari.benchmarks.utils import labeled_particles


def labeled_particles2d():
    labels, density, points = labeled_particles(
        (1024, 1024), return_density=True
    )

    return [
        (density, {"name": "density"}, "image"),
        (labels, {"name": "labels"}, "labels"),
        (points, {"name": "points"}, "points"),
    ]


def labeled_particles3d():
    labels, density, points = labeled_particles(
        (256, 512, 512), return_density=True
    )

    return [
        (density, {"name": "density"}, "image"),
        (labels, {"name": "labels"}, "labels"),
        (points, {"name": "points"}, "points"),
    ]
