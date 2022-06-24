from skimage import data, feature, filters
import napari

cells = data.cells3d()
nuclei = cells[:, 1]
smooth = filters.gaussian(nuclei, sigma=10)
pts = feature.peak_local_max(smooth)
viewer = napari.Viewer(ndisplay=3)
viewer.add_image(nuclei)
viewer.add_points(pts)

napari.run()