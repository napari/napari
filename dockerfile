# official python 3.11 image
# if you upgrade it, ensure you update the constraints file used
FROM --platform=linux/amd64 python:3.11-slim-bookworm AS napari
# if you change the distro version, remember to update
# the APT definitions for Xpra below so it reflects the
# new codename (e.g. bookworm)

# below env var required to install non-interactively
ENV TZ=America/Los_Angeles
ARG DEBIAN_FRONTEND=noninteractive
ARG NAPARI_COMMIT=main

# install python resources + graphical libraries used by qt and vispy
RUN apt-get update && \
    apt-get install -qqy  \
        build-essential \
        git \
        libglib2.0-0 \
        mesa-utils \
        libglx-mesa0 \
        # tlambert03/setup-qt-libs
        libegl1 \
        libdbus-1-3 \
        libxkbcommon-x11-0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-xinerama0 \
        libxcb-xinput0 \
        libxcb-xfixes0 \
        x11-utils \
        libxcb-cursor0 \
        libopengl0 \
        # other/remaining
        libfontconfig1 \
        libxrender1 \
        libxi6 \
        libxcb-shape0 \
        && apt-get clean

# install napari from repo
# Grab the constraints file to use for the install
# make sure it matches the base image python version!
COPY resources/constraints/constraints_py3.11.txt /tmp/constraints_py3.11.txt

# see https://github.com/pypa/pip/issues/6548#issuecomment-498615461 for syntax
RUN pip install --upgrade pip && \
    pip install "napari[all] @ git+https://github.com/napari/napari.git@${NAPARI_COMMIT}" \
    -c /tmp/constraints_py3.11.txt

# copy examples
COPY examples /tmp/examples

ENTRYPOINT ["python3", "-m", "napari"]

#########################################################
# Extend napari with a preconfigured Xpra server target #
#########################################################

FROM napari AS napari-xpra

ARG DEBIAN_FRONTEND=noninteractive

# Install Xpra and dependencies
# Remember to update the xpra.sources link for any change in distro version
RUN apt-get update && apt-get install -y wget gnupg2 apt-transport-https \
    software-properties-common ca-certificates && \
    wget -O "/usr/share/keyrings/xpra.asc" https://xpra.org/xpra.asc && \
    wget -O "/etc/apt/sources.list.d/xpra.sources" https://raw.githubusercontent.com/Xpra-org/xpra/master/packaging/repos/bookworm/xpra.sources


RUN apt-get update && \
    apt-get install -yqq \
        xpra \
        xvfb \
        menu-xdg \
        xdg-utils \
        xterm \
        sshfs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV DISPLAY=:100
ENV XPRA_PORT=9876
ENV XPRA_START="python3 -m napari"
ENV XPRA_EXIT_WITH_CLIENT="yes"
ENV XPRA_XVFB_SCREEN="1920x1080x24+32"
EXPOSE 9876

CMD echo "Launching napari on Xpra. Connect via http://localhost:$XPRA_PORT or $(hostname -i):$XPRA_PORT"; \
    xpra start \
    --bind-tcp=0.0.0.0:$XPRA_PORT \
    --html=on \
    --start="$XPRA_START" \
    --exit-with-client="$XPRA_EXIT_WITH_CLIENT" \
    --daemon=no \
    --xvfb="/usr/bin/Xvfb +extension Composite -screen 0 $XPRA_XVFB_SCREEN -nolisten tcp -noreset" \
    --pulseaudio=no \
    --notifications=no \
    --bell=no \
    $DISPLAY

ENTRYPOINT []
