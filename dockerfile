FROM --platform=linux/amd64 ubuntu:22.04 AS napari
# if you change the Ubuntu version, remember to update
# the APT definitions for Xpra below so it reflects the
# new codename (e.g. 20.04 was focal, 22.04 had jammy)

# below env var required to install libglib2.0-0 non-interactively
ENV TZ=America/Los_Angeles
ARG DEBIAN_FRONTEND=noninteractive
ARG NAPARI_COMMIT=main

# install python resources + graphical libraries used by qt and vispy
RUN apt-get update && \
    apt-get install -qqy  \
        build-essential \
        python3.9 \
        python3-pip \
        git \
        mesa-utils \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libfontconfig1 \
        libxrender1 \
        libdbus-1-3 \
        libxkbcommon-x11-0 \
        libxi6 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-xinerama0 \
        libxcb-xinput0 \
        libxcb-xfixes0 \
        libxcb-shape0 \
        && apt-get clean

# install napari from repo
# see https://github.com/pypa/pip/issues/6548#issuecomment-498615461 for syntax
RUN pip3 install "napari[all] @ git+https://github.com/napari/napari.git@${NAPARI_COMMIT}"

# copy examples
COPY examples /tmp/examples

ENTRYPOINT ["python3", "-m", "napari"]

#########################################################
# Extend napari with a preconfigured Xpra server target #
#########################################################

FROM napari AS napari-xpra

# Install Xpra and dependencies
RUN apt-get update && apt-get install -y wget gnupg2 apt-transport-https \
    software-properties-common ca-certificates && \
    wget -O "/usr/share/keyrings/xpra.asc" https://xpra.org/xpra.asc && \
    wget -O "/etc/apt/sources.list.d/xpra.sources" https://xpra.org/repos/jammy/xpra.sources


RUN apt-get update && \
    apt-get install -yqq \
        xpra \
        xvfb \
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
