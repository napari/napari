from skimage import data
import numpy as np
import napari

np.random.seed(0)
n = 10_000
data = 100 * np.random.rand(n, 3)
size = np.round(10 * np.random.rand(n)).astype(int)
features = {
    'class': np.random.choice(['a', 'b', 'c'], n),
    'confidence': np.random.rand(n),
}

viewer = napari.view_points(
        data,
        features=features,
        face_color='class',
        face_color_cycle=['r', 'g', 'b'],
        edge_color='confidence',
        edge_colormap='gray',
        size=size,
)

if __name__ == '__main__':
    napari.run()
