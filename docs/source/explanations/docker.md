# Napari in Docker (WIP)

## Build
A dockerfile is added to napari root to allow build of a docker image using official napari release. 
Note that napari in docker is still in alpha stage and not working universally, feedback and contribution also welcomed.

To build the image, run from napari root
```
docker build -t napari .
```
which would build a docker image with tag napari, releases are also available in dockerhub under napari/napari.

## Usage
