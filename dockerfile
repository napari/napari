FROM ubuntu:latest

# install python resources
RUN apt-get update && apt-get install -qqy build-essential python3.8 python3-pip

# below env var required to install libglib2.0-0 non-interactively
ENV TZ America/Los_Angeles
ENV DEBIAN_FRONTEND noninteractive

# install graphical libraries used by qt and vispy
RUN apt-get install -qqy mesa-utils libgl1-mesa-glx  libglib2.0-0
RUN apt-get install -qqy libfontconfig1 libxrender1 libdbus-1-3 libxkbcommon-x11-0 libxi6
RUN apt-get install -qqy libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0
RUN apt-get install -qqy libxcb-xinerama0 libxcb-xinput0 libxcb-xfixes0 libxcb-shape0

# install napari release version
RUN pip3 install napari[all]

# install scikit image for examples
RUN pip3 install scikit-image
COPY examples /tmp/examples

ENTRYPOINT ["python3", "-m", "napari"]