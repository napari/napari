from setuptools import find_packages, setup

setup(
    name="napari-language-pack-es-CO",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "napari.languagepack": ["es_CO = napari_language_pack_es_CO"]
    },
)
