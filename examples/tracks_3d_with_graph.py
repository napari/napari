import napari
import numpy as np

def _circle(r, theta):
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x, y


def tracks_3d_merge_split():
    """ create tracks with splitting and merging """

    timestamps = np.arange(300)


    def _trajectory(t, r):
        theta = t*0.1
        x, y = _circle(r, theta)
        z = np.zeros(x.shape)
        return np.stack([t, z, y, x], axis=1)

    trackA = _trajectory(timestamps[:100], 30.)
    trackB = _trajectory(timestamps[100:200], 10.)
    trackC = _trajectory(timestamps[100:200], 50.)
    trackD = _trajectory(timestamps[200:], 30.)

    data = [trackA, trackB, trackC, trackD]
    tracks = np.concatenate(data, axis=0)
    tracks[:,1:] += 50. # centre the track at (50, 50, 50)

    graph = {1:0, 2:[0], 3:[1,2]}

    properties = {'track_id': np.concatenate([[i]*100 for i in range(4)]),
                  'time': tracks[:,0]}

    return tracks, properties, graph



tracks, properties, graph = tracks_3d_merge_split()

with napari.gui_qt():
    viewer = napari.Viewer()
    # viewer.add_points(tracks, size=1)
    viewer.add_tracks(tracks, properties=properties, graph=graph, name='tracks')
