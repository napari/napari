import numpy as np
import napari

image_data = np.random.rand(1000, 256, 256)
translate = [0, 10, -20]
scale = [1, 2, 3]

viewer = napari.view_image(image_data, translate=translate, scale=scale)

if __name__ == '__main__':
    napari.run()
