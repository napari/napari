
from typing import Dict, List
import numpy as np

from skimage.measure import regionprops_table
from skimage.util import map_array
import dask.array as da
import napari
import pathlib
from tqdm import tqdm
from omero.gateway import BlitzGateway
from itertools import product


# config
IDR_imageID = 1512575
props = ['area', 'label']
props_intensity = ['mean_intensity']

import numpy as np


def IDR_fetch_image(image_id: int, progressbar: bool = True) -> np.ndarray:
    """
    Download the image with id image_id from the IDR

    Will fetch all image planes corresponding to separate
    timepoints/channels/z-slices and return a numpy
    array with dimension order (t,z,y,x,c)

    Displaying download progress can be disabled by passing
    False to progressbar.
    """

    conn = BlitzGateway(
        host="ws://idr.openmicroscopy.org/omero-ws",
        username="public",
        passwd="public",
        secure=True,
    )
    conn.connect()
    conn.c.enableKeepAlive(60)

    idr_img = conn.getObject("Image", image_id)
    idr_pixels = idr_img.getPrimaryPixels()

    _ = idr_img
    nt, nz, ny, nx, nc = (
        _.getSizeT(),
        _.getSizeZ(),
        _.getSizeY(),
        _.getSizeX(),
        _.getSizeC(),
    )

    plane_indices = list(product(range(nz), range(nc), range(nt)))
    idr_plane_iterator = idr_pixels.getPlanes(plane_indices)

    if progressbar:
        idr_plane_iterator = tqdm(idr_plane_iterator, total=len(plane_indices))

    _tmp = np.asarray(list(idr_plane_iterator))
    _tmp = _tmp.reshape((nz, nc, nt, ny, nx))
    return np.einsum("jmikl", _tmp)



def get_video_and_segmentations(image_id: int):
    # video
    img_file = f"{image_id}.zarr"
    if not pathlib.Path(img_file).exists():
        print("Downloading sample image sequence from IDR.")
        raw_video = da.from_array(np.squeeze(IDR_fetch_image(image_id)))
        da.to_zarr(raw_video, img_file)
    else:
        print(f"Opening previously downloaded sample image sequence: {img_file}.")
        raw_video = da.from_zarr(img_file)
    # segmentation
    _p = pathlib.Path(img_file)
    label_path = _p.parent / (_p.stem + "_labels.zarr")
    if label_path.exists():
        print(f"Reading existing segmentation from {str(label_path)}.")
        labels = np.asarray(da.from_zarr(str(label_path)))
    else:
        print("No existing segmentation found. Segmenting timelapse using cellpose:")
        import mxnet
        from cellpose import models 
        model = models.Cellpose(device=mxnet.cpu(), model_type='nuclei')
        def _segment_cellpose(img):
            _labels, _, _, _ = model.eval(
            [np.asarray(img)], rescale=None, channels=[0,0]
            )
            return _labels
        labels = np.asarray([_segment_cellpose(_frame) for _frame in tqdm(raw_video)])
        print(f"Saving segmentation as {str(label_path)}")
        da.from_array(labels).to_zarr(str(label_path))

    labels = np.squeeze(labels)
    return raw_video, labels


def integrated_intensity(mask, intensity):
    return np.sum(intensity[mask])


def measure_timelapse_features(vid, seg, properties, extra_properties):
    def _measure_frame(frame, frameseg):
        _frame_props = regionprops_table(
                        label_image=np.asarray(frameseg),
                        intensity_image=np.asarray(frame),
                        properties=properties,
                        extra_properties=extra_properties)
        return _frame_props
    return map( _measure_frame, tqdm(vid), seg)

def heatmap(labels, measurements: List[Dict[str, np.array]], property: str) -> np.ndarray:
    def _map_frame(label_frame, meas_dict):
       return map_array(label_frame, meas_dict["label"], meas_dict[property])
    return np.asarray(list(map(_map_frame, labels, measurements)))

print("Obtaining raw image sequence and segmentation:")
vid, seg = get_video_and_segmentations(IDR_imageID)
print("Calculating region properties:")
measurements = list(measure_timelapse_features(vid,seg, props+props_intensity, extra_properties=(integrated_intensity,)))
print("Generating measurement colormap overlays")
area_map = heatmap(seg, measurements, 'area')
mean_int_map = heatmap(seg, measurements, 'mean_intensity')
integrated_intensity_map = heatmap(seg, measurements, 'integrated_intensity')

with napari.gui_qt():
    v=napari.Viewer()
    scale = (5,1,1)
    v.add_image(vid, scale=scale)
    v.add_labels(seg, scale=scale)
    v.add_image(area_map, colormap='red', blending='additive', scale=scale)
    v.add_image(mean_int_map, colormap='green', blending='additive', scale=scale)
    v.add_image(integrated_intensity_map, colormap='blue', blending='additive', scale=scale)
