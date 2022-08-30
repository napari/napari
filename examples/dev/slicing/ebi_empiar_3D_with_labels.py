import napari
import pooch
from tifffile import imread

"""
This data comes from the MitoNet Benchmarks.

Six benchmark volumes of instance segmentation of mitochondria from diverse volume EM datasets
Narayan K , Conrad RW
DOI: https://dx.doi.org/10.6019/EMPIAR-10982

Data is stored at EMPIAR and can be explored here: https://www.ebi.ac.uk/empiar/EMPIAR-10982/

With respect to the napari async slicing work, this dataset is small enough that it performs well in synchronous mode.
"""

salivary_gland_em_path = pooch.retrieve(
    url='https://ftp.ebi.ac.uk/empiar/world_availability/10982/data/mito_benchmarks/salivary_gland/salivary_gland_em.tif',
    known_hash='222f50dd8fd801a84f118ce71bc735f5c54f1a3ca4d98315b27721ae499bff94',
)

salivary_gland_mito_path = pooch.retrieve(
    url='https://ftp.ebi.ac.uk/empiar/world_availability/10982/data/mito_benchmarks/salivary_gland/salivary_gland_mito.tif',
    known_hash='95247d952a1dd0f7b37da1be95980b598b590e4777065c7cd877ab67cb63c5eb',
)

salivary_gland_em = imread(salivary_gland_em_path)
salivary_gland_mito = imread(salivary_gland_mito_path)

viewer = napari.view_image(salivary_gland_em)
viewer.add_labels(salivary_gland_mito)

napari.run()
