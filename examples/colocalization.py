import numpy as np
from skimage import data
from skimage.color import rgb2hed
from scipy.stats import linregress


import napari

ihc_rgb = data.immunohistochemistry()
ihc_hed = rgb2hed(ihc_rgb)

with napari.gui_qt():
    viewer = napari.Viewer()
    viewer.add_image(ihc_hed[..., ::2], channel_axis=-1)
    # add a docked figure
    fig, dw = viewer.window.add_docked_figure()
    # get a handle to the plotWidget
    ax = fig[0, 0]

    # calculate some data
    x, y = ihc_hed[..., 0].ravel(), ihc_hed[..., 2].ravel()
    data = np.array((x, y)).T
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    linex = np.linspace(x.min(), x.max(), 200)
    liney = slope * linex + intercept
    
    # plot the scatter plot and line
    ax.plot((linex, liney), color='g', marker_size=0, width=2)
    ax.scatter(data, size=2, edge_width=0, face_color='m')
    viewer.window._qt_window.setGeometry(400, 100, 800, 900)