from napari.experimental._progressive_loading import (
    add_progressive_loading_image,
)
from napari.experimental._progressive_loading_datasets import (
    luethi_zenodo_7144919,
)

if __name__ == "__main__":
    import napari

    viewer = napari.Viewer(ndisplay=3)

    multiscale_img = luethi_zenodo_7144919()["arrays"]

    print(multiscale_img[0])

    add_progressive_loading_image(
        multiscale_img,
        viewer=viewer,
        contrast_limits=[0, 255],
        colormap='twilight_shifted',
        ndisplay=3,
    )

    viewer.axes.visible = True
