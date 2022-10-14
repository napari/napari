from napari.layers.shapes._shapes_utils import (
    generate_2D_edge_meshes,
)  # , old_generate_2D_edge_meshes
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

fig, axes = plt.subplots(2, 3)
# fig.set_figwidth(15)
# fig.set_figheight(10)
colors = iter(['red', 'green', 'blue', 'yellow'])
itaxes = iter(axes.flatten())
sup = axes.flatten()[4]
for closed in [False, True]:
    for beveled in [False, True]:
        ax = next(itaxes)
        c = next(colors)
        centers, offsets, triangles = generate_2D_edge_meshes(
            [[0, 3], [1, 0], [2, 3], [5, 0], [2.5, 5]],
            closed=closed,
            limit=3,
            bevel=beveled,
        )
        points = centers + 0.3 * offsets
        for t in triangles:
            trp = points[t]
            ax.add_patch(Polygon(trp, ec='#000000', fc=c, alpha=0.2))
            sup.add_patch(Polygon(trp, ec='#000000', fc=c, alpha=0.1))
        ax.scatter(*(points).T)
        ax.scatter(*(centers).T)
        ax.set_aspect('equal')
        ax.set_title(f' {closed=}, {beveled=}')
        ax.set_xlim(-1, 6)
        ax.set_ylim(-1, 6)
        sup.set_xlim(-1, 6)
        sup.set_ylim(-1, 6)
plt.show()
