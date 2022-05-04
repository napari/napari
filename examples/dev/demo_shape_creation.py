import argparse
from timeit import default_timer

import numpy as np

import napari


def create_sample_coords(n_polys=3000, n_vertices=32):
    """random circular polygons with given number of vertices"""
    center = np.random.randint(0, 1000, (n_polys, 2))
    radius = (
        1000
        / np.sqrt(n_polys)
        * np.random.uniform(0.9, 1.1, (n_polys, n_vertices))
    )

    phi = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
    rays = np.stack([np.sin(phi), np.cos(phi)], 1)

    radius = radius.reshape((-1, n_vertices, 1))
    rays = rays.reshape((1, -1, 2))
    center = center.reshape((-1, 1, 2))
    coords = center + radius * rays
    return coords


def time_me(label, func):
    # print(f'{name} start')
    t = default_timer()
    res = func()
    t = default_timer() - t
    print(f"{label}: {t:.4f} s")
    return res


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-n",
        "--n_polys",
        type=int,
        default=5000,
        help='number of polygons to show',
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        default="path",
        choices=['path', 'path_concat', 'polygon', 'rectangle', 'ellipse'],
    )
    parser.add_argument(
        "-c",
        "--concat",
        action="store_true",
        help='concatenate all coordinates to a single mesh',
    )
    parser.add_argument(
        "-v", "--view", action="store_true", help='show napari viewer'
    )
    parser.add_argument(
        "--properties", action="store_true", help='add dummy shape properties'
    )

    args = parser.parse_args()

    coords = create_sample_coords(args.n_polys)

    if args.type == 'rectangle':
        coords = coords[:, [4, 20]]
    elif args.type == 'ellipse':
        coords = coords[:, [0, 8, 16,22]]
    elif args.type == 'path_concat':
        args.type = 'path'
        coords = coords.reshape((1, -1, 2))


    print(f'number of polygons: {len(coords)}')
    print(f'layer type: {args.type}')
    print(f'properties: {args.properties}')

    properties = {
        'class': (['A', 'B', 'C', 'D'] * (len(coords) // 4 + 1))[
            : len(coords)
        ],
    }
    color_cycle = ['blue', 'magenta', 'green']

    kwargs = dict(
        shape_type=args.type,
        properties=properties if args.properties else None,
        face_color='class' if args.properties else [1,1,1,1],
        face_color_cycle=color_cycle,
        edge_color='class' if args.properties else [1,1,1,1],
        edge_color_cycle=color_cycle,
    )

    layer = time_me(
        "time to create layer",
        lambda: napari.layers.Shapes(coords, **kwargs),
    )

    if args.view:
        # add the image
        viewer = napari.Viewer()
        viewer.add_layer(layer)
        napari.run()
