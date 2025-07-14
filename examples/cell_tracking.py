"""
Updated Cell tracking example
==============================

This example demonstrates how the tracking layer is used
for displaying 2D + time dataset of cells with colored properties for track-id.

Thanks to Dr. Alessia Ruggieri and Philipp Klein, Centre for Integrative Infectious
Disease Research (CIID), University Hospital Heidelberg, Germany for the data.
You can find the data on: https://doi.org/10.5281/zenodo.15597019

"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pooch
import tifffile

import napari

###############################################################################
# Download data
# ------------------
download = pooch.DOIDownloader(progressbar=True)
doi = "10.5281/zenodo.15597019/01.zip"
tmp_dir = Path(pooch.os_cache('napari-cell-tracking-example'))
#os.makedirs(tmp_dir, exist_ok=True)
tmp_dir.mkdir(parents=True, exist_ok=True)

url = "https://zenodo.org/records/15832699/files/enhanced_gnn_tracks.csv?download=1"
data_dir = pooch.retrieve(
    url=url,
    known_hash="sha256:a410154f3d0cdd2807f0aa66a87527c8125816979cd035fc9b61378551fe7b52",  
    fname="enhanced_gnn_tracks",
    path=pooch.os_cache("my_tracking_project")
)

label_url = 'https://zenodo.org/records/15852284/files/masks_pred.npz?download=1'
label_path = pooch.retrieve(url=label_url, 
                            fname="masks_pred.npz",
                            known_hash=None,
                            path=pooch.os_cache("cell_labels")
                            )

data_path = tmp_dir/"01.zip"

if not data_path.exists():
    print(f"downloading archive {data_path.name}")
    #download(f"doi:{doi}", output_file=data_path, path=data_files)
    download(f"doi:{doi}", output_file=data_path, pooch=None)
else:
    print(f"using cached {data_path}")

with zipfile.ZipFile(data_path, 'r') as zip_ref:
    zip_ref.extractall(tmp_dir)

print("Contents of tmp_dir after extraction:")
for p in tmp_dir.rglob("*"):
    print(f"Extracted {p} files to {tmp_dir}")

#imgs = glob.glob(os.path.join(tmp_dir, "*.tif"))
imgs = list((tmp_dir/"01").glob ("*.tif"))

if not imgs:
    raise FileNotFoundError("no tif files found")

images = np.stack([
        tifffile.imread(fn) for fn in sorted(imgs)
        ])
viewer = napari.Viewer()
viewer.add_image(images, name='Cell Tracking', colormap='gray',
                 metadata={'Unit': 'Hours'})

def load_tracks_data(data_dir, filename=""):
    """
    Load tracks data from a specific CSV file.
    Returns tracks data in napari format: [track_id, t, y, x]
    """
    track_file = data_dir / filename

    if not track_file.exists():
        print(f"Tracks file not found: {track_file}")
        return None

    try:
        df = pd.read_csv(track_file)
        print(f"Loaded tracks from: {track_file}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Shape: {df.shape}")

        # Reorder CSV columns to napari format: [track_id, t, y, x]
        if len(df.columns) >= 4:
            # Reorder to [track_id, t, y, x] format
            tracks_data = df.iloc[:, [0, 1, 4, 3]].values
            print(f"Successfully loaded {len(tracks_data)} track points")
            print("Data reordered to [track_id, frame, y, x] format")
            return tracks_data
        print(f"CSV file has fewer than 4 columns: {len(df.columns)}")
        return None

    except Exception as e:
        print(f"Error loading tracks file: {e}")
        return None

url = "https://zenodo.org/records/15832699/files/enhanced_gnn_tracks.csv?download=1"
data_dir = pooch.retrieve(
    url=url,
    known_hash="sha256:a410154f3d0cdd2807f0aa66a87527c8125816979cd035fc9b61378551fe7b52",
    fname="enhanced_gnn_tracks",
    path=pooch.os_cache("my_tracking_project")
)

tracks_data = load_tracks_data(Path(data_dir).parent, filename=Path(data_dir).name)

# Add tracks layer with colors
if tracks_data is not None:
    print(f"Adding {len(np.unique(tracks_data[:, 0]))} tracks to napari...")

    # Create track properties for coloring
    track_ids = np.unique(tracks_data[:, 0])


    # Detect splitting tracks (tracks that share the same parent or split from one another)
    # Logic: identify tracks that start at the same time and in close position (less than 10 pixels apart)
    splitting_tracks = set()

    for track_id in track_ids:
        track_points = tracks_data[tracks_data[:, 0] == track_id]
        start_time = track_points[:, 1].min()
        start_pos = track_points[track_points[:, 1] == start_time][0, 2:4]  # [y, x]

        # Check if other tracks start at same time and nearby position
        for other_track_id in track_ids:
            if other_track_id != track_id:
                other_track_points = tracks_data[tracks_data[:, 0] == other_track_id]
                other_start_time = other_track_points[:, 1].min()

                if other_start_time == start_time:
                    other_start_pos = other_track_points[other_track_points[:, 1] == other_start_time][0, 2:4]
                    # If they start within 10 pixels of each other, consider them splitting
                    distance = np.sqrt(np.sum((start_pos - other_start_pos)**2))
                    if distance < 10:
                        splitting_tracks.add(track_id)
                        splitting_tracks.add(other_track_id)

    # Create the arrays using the populated splitting_tracks
    track_id_per_point = tracks_data[:, 0].astype(int)
    is_splitting_per_point = np.array([int(tid) in splitting_tracks for tid in track_id_per_point], dtype=float)

    # Create features dictionary
    features = pd.DataFrame({
    'track_id': track_id_per_point,
    'is_splitting': is_splitting_per_point
    })

    # Add tracks layer
    tracks_layer = viewer.add_tracks(
        tracks_data,
        features=features,
        name='Cell Tracks',
        tail_length=10,
        tail_width=2,
        head_length=0,
        colormap='viridis',
        color_by='is_splitting',
        blending='additive'
    )
# Add cell lebels 
# Add cell lebels
label_url = 'https://zenodo.org/records/15852284/files/masks_pred.npz?download=1'
label_path = pooch.retrieve(url=label_url,
                            fname="masks_pred.npz",
                            known_hash=None,
                            path=pooch.os_cache("cell_labels")
                            )

label_data = np.load(label_path)
labels = label_data["masks"]

viewer.add_labels(labels, name="Predicted Masks")
napari.run()
