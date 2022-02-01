# Napari in Docker

## Build

Builds are avilable through [dockerhub](https://hub.docker.com/repository/docker/napari/napari)

A dockerfile is added to napari root to allow build of a docker image using official napari release. 
Note that napari in docker is still in alpha stage and not working universally, feedback and contribution also welcomed.

To build the image, run from napari root
```
docker build -t napari/napari:<version> .
```
which would build a docker image tagged with napari version

## Usage

Enable XServer on the host machine, these can be useful if you are looking for options:
* Windows: [vcxsrc](https://sourceforge.net/projects/vcxsrv/)
* MacOS: [xquartz](https://www.xquartz.org/) (may not work due to graphical driver issue with opengl)

To run a container with external mapping of display, an example being:
```
docker run -d  -e DISPLAY=host.docker.internal:0 napari/napari:0.3.6 python3 /tmp/examples/add_image.py
```
