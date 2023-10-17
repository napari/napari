# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.233.0/containers/python-3-miniconda/.devcontainer/base.Dockerfile

FROM mcr.microsoft.com/vscode/devcontainers/miniconda:0-3


RUN sudo apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && sudo apt-get -y install --no-install-recommends \
    libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
    libxcb-render-util0 libxcb-xinerama0 libxkbcommon-x11-0
