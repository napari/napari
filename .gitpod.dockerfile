FROM gitpod/workspace-full-vnc

# Install dependencies
RUN sudo apt-get update \
    && sudo apt-get install -y build-essential python3.8 python3-pip \
        mesa-utils libgl1-mesa-glx libglib2.0-0 \
        libfontconfig1 libxrender1 libdbus-1-3 libxkbcommon-x11-0 libxi6 \
        libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0  \
        libxcb-xinerama0 libxcb-xinput0 libxcb-xfixes0 libxcb-shape0 \
        x11vnc xvfb \
    && sudo pip3 install napari[all] scikit-image \
    && sudo apt-get clean && sudo rm -rf /var/cache/apt/* && sudo rm -rf /var/lib/apt/lists/* && sudo rm -rf /tmp/*