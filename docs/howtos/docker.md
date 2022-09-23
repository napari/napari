# Napari in Docker

## Build

Builds are available in the [GitHub Container Registry](https://github.com/orgs/napari/packages).

A dockerfile is added to napari root to allow build of a docker image using official napari release.
It contains two targets built on top of Ubuntu 20.04:

* `napari`: The result of `pip install napari[all] scikit-image` for Python 3.8, including all the system libraries required by PyQt.
* `napari-xpra`: Same as above, plus a preconfigured Xpra server.

Note that napari in Docker is still in alpha stage and not working universally. Feedback and contributions are welcomed!

To build the image, run one of these commands from napari root:

```bash
# build napari image
docker build --target napari -t ghcr.io/napari/napari:<version> .
# build napari + xpra image
docker build --target napari-xpra -t ghcr.io/napari/napari-xpra:<version> .
```

which would build a Docker image tagged with napari version.

## Usage

### Base `napari` image

First, make sure there's a running X server on the host machine.
These can be useful if you are looking for options:
* Windows: [vcxsrc](https://sourceforge.net/projects/vcxsrv/)
* MacOS: [xquartz](https://www.xquartz.org/) (may not work due to graphical driver issue with opengl)

To run a container with external mapping of display, an example being:

```
docker run -it --rm -e DISPLAY=host.docker.internal:0 ghcr.io/napari/napari
```

### `napari-xpra` image

With this image you don't need X running on the host. A browser is sufficient!

```
docker run -it --rm -p 9876:9876 ghcr.io/napari/napari-xpra
```

Once that's running, you can open a tab on your browser of choice and go to ``http://localhost:9876``.
You'll be presented with a virtual desktop already running running napari.
The desktop features a basic menu at the top with some extra items, like a `Xterm` terminal.

This image features a series of environment variables you can use to customize its behaviour:

* `XPRA_PORT=9876`: Port where Xpra will publish the display feed (if you change this, make sure to use the new port in your  `docker run`)
* `XPRA_START="python3 -m napari"`: Xpra will run this command once it has started
* `XPRA_EXIT_WITH_CLIENT="yes"`: By default, Xpra will exit if you close the browser tab
* `XPRA_XVFB_SCREEN="1920x1080x24+32"`: The resolution and bit depth of the virtual display created by Xvfb

## For development

The Docker images are also useful for developers who need to debug issues on Linux.
The images include the latest napari version published on PyPI by default, but you can also install your own local version of napari if needed.
For this, you need to mount a volume as part of the `docker run` command and make sure you land in a `bash` session:

```bash
# base napari image, we replace entry point (python3 -m napari) with a bash session
docker run -it --rm -e DISPLAY=host.docker.internal:0 -v ~/devel/napari:/opt/napari --entrypoint /bin/bash ghcr.io/napari/napari
# napari-xpra image, we replace the command Xpra will run on start (python3 -m napari) to a bash session running on xterm
docker run -it --rm -p 9876:9876 -v ~/devel/napari:/opt/napari -e XPRA_START=xterm ghcr.io/napari/napari-xpra
```

> Change `~/devel/napari` to your local copy of the napari repository, which will be visible in the image as `/opt/napari`.

In both cases you'll have a running shell session where you can run these commands:

```bash
# Install local napari
$ python3 -m pip install /opt/napari[all]
# Run napari
$ python3 -m napari
```

## Troubleshooting

Making the Docker image run seamlessly with your host graphical stack can be tricky.
You might find issues like these:

* napari seems to be running, but no window appears at all
* napari does run and the UI works as intended, but the viewer portion is completely black
* napari does not start due to missing libraries or mismatched ABIs (e.g. drivers in host are incompatible with guest)

In these cases, the best option is to stop relying on the host graphics and let the guest handle
everything. To do this, we need several pieces in place:

* A headless X server running on the guest; e.g. `Xvfb` (X virtual frame buffer)
* A way of "seeing" that X server from the host. Some options include:
    * VNC (might have issues with OpenGL)
    * NX (NoMachine;, [should work](https://github.com/napari/napari/issues/886#issuecomment-873178682))
    * Microsoft Remote Desktop Protocol (e.g. XRDP; [should work](https://github.com/napari/napari/issues/886#issuecomment-875959941))
    * Chrome Remote Desktop ([should work](https://github.com/napari/napari/issues/886#issue-551159225))
    * Xpra (part of the Docker image detailed above)

Since napari relies on OpenGL for its (hardware) accelerated parts, we need to ensure that the piece we choose are OpenGL compatible.
Xvfb itself is compatible, but the display "exporter" needs to know how to deal with that part of the graphics too!

You must also note that Docker won't probably have access to the host GPU, so everything will be rendered in the CPU.
Expect some performance overhead!
