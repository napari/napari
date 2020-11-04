from typing import Dict, List, Tuple
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
#IDR_imageID = 1512575
#IDR_imageID = 1490296 
#IDR_imageID = 1486120 
IDR_imageID =  1486532 # a few very fast moving abberant cells nice cell division
#IDR_imageID =  1483589 # KIF11
#IDR_imageID = 2862565 # KIF11 validation screen

props = ['area', 'label']
props_intensity = ['mean_intensity']


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


def get_video(image_id: int):
    """ Return IDR timelapse video with ID image_id
    as dask array. Download if necessary and cache 
    on disk as Zarr. Returns the cached version from
    disk if present"""
    img_file = f"{image_id}.zarr"
    if not pathlib.Path(img_file).exists():
        print("Downloading sample image sequence from IDR.")
        raw_video = da.from_array(np.squeeze(IDR_fetch_image(image_id)))
        da.to_zarr(raw_video, img_file)
    else:
        print(
            f"Opening previously downloaded sample image sequence: {img_file}."
        )
        raw_video = da.from_zarr(img_file)
    return raw_video

def get_video_segmentation(image_id: int):
    """ Return nucleus segmentation as label image
    from IDR timelapse video with ID image_id
    as dask array. 
    If necessary, the segmentation is calculated 
    slice by slice using cellpose and cached as a zarr file 
    on disk.
    If the cached zarr file is found, it is read.
    
    Returns both the raw timelapse video and 
    the segmentation.
    """
    raw_video = get_video(image_id)
    label_file = f"{image_id}_labels.zarr"

    if pathlib.Path(label_file).exists():
        print(f"Reading existing segmentation from {label_file}.")
        labels = np.asarray(da.from_zarr(label_file))
    else:
        print(
            "No existing segmentation found. Segmenting timelapse using cellpose:"
        )
        import mxnet
        from cellpose import models

        model = models.Cellpose(device=mxnet.gpu(), model_type='nuclei')

        def _segment_cellpose(img):
            _labels, _, _, _ = model.eval(
                [np.asarray(img)], rescale=None, channels=[0, 0]
            )
            return _labels

        labels = np.asarray(
            [_segment_cellpose(_frame) for _frame in tqdm(raw_video)]
        )
        print(f"Saving segmentation as {label_file}")
        labels=np.squeeze(labels)     
        da.from_array(labels).to_zarr(label_file)

    return raw_video, labels

def integrated_intensity(mask, intensity):
    return np.sum(intensity[mask])

def measure_timelapse_features(vid, seg, properties, extra_properties):
    """Given a timelapse sequence vid and a corresponding
    label image seg, compute regionprops properties and extra_properties
    for all frames."""
    def _measure_frame(frame, frameseg):
        """measures properties in a single frame"""
        _frame_props = regionprops_table(
            label_image=np.asarray(frameseg),
            intensity_image=np.asarray(frame),
            properties=properties,
            extra_properties=extra_properties,
        )
        return _frame_props

    return list(map(_measure_frame, tqdm(vid), seg))


def heatmap(
    labels, measurements: List[Dict[str, np.array]], property: str
) -> np.ndarray:
    def _map_frame(label_frame, meas_dict):
        return map_array(label_frame, meas_dict["label"], meas_dict[property])

    return np.asarray(list(map(_map_frame, labels, measurements)))


###
###


print("Obtaining raw image sequence and segmentation:")
vid, seg = get_video_segmentation(IDR_imageID)
print("Calculating region properties:")
measurements = measure_timelapse_features(
        vid,
        seg,
        props + props_intensity,
        extra_properties=(integrated_intensity,),
)

print("Generating measurement colormap overlays")
area_map = heatmap(seg, measurements, 'area')
mean_int_map = heatmap(seg, measurements, 'mean_intensity')
integrated_intensity_map = heatmap(seg, measurements, 'integrated_intensity')

with napari.gui_qt():
    v = napari.Viewer()
    scale = (5, 1, 1)
    v.add_image(vid, scale=scale)
    v.add_labels(seg, scale=scale)
    v.add_image(area_map, colormap='red', blending='additive', scale=scale)
    v.add_image(
        mean_int_map, colormap='green', blending='additive', scale=scale
    )
    v.add_image(
        integrated_intensity_map,
        colormap='blue',
        blending='additive',
        scale=scale,
    )
