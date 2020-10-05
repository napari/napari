
from sys import path
from typing import Dict, List
from numba.cuda.cudadrv.driver import profile_stop
import numpy as np
import mxnet
from cellpose import models
import matplotlib.pyplot as plt
from numpy.lib.function_base import meshgrid
from scipy.ndimage import measurements
from skimage.measure import regionprops_table
from skimage.util import map_array
import dask.array as da
import napari
import pathlib
import tqdm

# config
input_file = "/home/volker/Mitocheck_MP4/demo.zarr"
props = ['area', 'label']
props_intensity = ['mean_intensity']


def get_video_and_segmentations(input_file):
    raw_video = da.from_zarr(input_file)
    _p = pathlib.Path(input_file)
    label_path = _p.parent / (_p.stem + "_labels.zarr")
    print(str(label_path))
    if label_path.exists():
        print(f"Reading existing segmentation from {str(label_path)}.")
        labels = np.asarray(da.from_zarr(str(label_path)))
    else:
            # create segmentation using cellpose
        print("Segmenting timelapse using cellpose:")
        model = models.Cellpose(device=mxnet.gpu(), model_type='nuclei')
        def _segment_cellpose(img):
            _labels, _, _, _ = model.eval(
            [np.asarray(img)], rescale=None, channels=[0,0]
            )
            return _labels
        labels = np.asarray([_segment_cellpose(_frame) for _frame in tqdm.tqdm(raw_video)])
        print(f"Saving segmentation as {str(label_path)}")
        da.from_array(labels).to_zarr(str(label_path))

    print(labels.shape)
    labels = np.squeeze(labels)
    return raw_video, labels


   

vid, seg = get_video_and_segmentations(input_file)


def measure_timelapse_features(vid, seg, properties):
    def _measure_frame(frame, frameseg):
        _frame_props = regionprops_table(
                        label_image=np.asarray(frameseg),
                        intensity_image=np.asarray(frame),
                        properties=properties)
        return _frame_props
    return map( _measure_frame, vid,seg)

def heatmap(labels, measurements: List[Dict[str, np.array]], property: str) -> np.ndarray:
    def _map_frame(label_frame, meas_dict):
       return map_array(label_frame, meas_dict["label"], meas_dict[property])
    return np.asarray(list(map(_map_frame, labels, measurements)))

measurements = list(measure_timelapse_features(vid,seg, props+props_intensity))
area_map = heatmap(seg, measurements, 'area')
mean_int_map = heatmap(seg, measurements, 'mean_intensity')

with napari.gui_qt():
    v=napari.Viewer()
    scale = (5,1,1)
    v.add_image(vid, scale=scale)
    v.add_labels(seg, scale=scale)
    v.add_image(area_map, colormap='magenta', blending='additive', scale=scale)
    v.add_image(mean_int_map, colormap='green', blending='additive', scale=scale)
