FROM gitpod/workspace-full-vnc

# Install dependencies
RUN sudo apt-get update \
    && sudo apt-get install -y build-essential python3.8 python3-pip \
    && sudo pip3 install napari[all] scikit-image \
    && sudo apt-get clean && rm -rf /var/cache/apt/* && rm -rf /var/lib/apt/lists/* && rm -rf /tmp/*