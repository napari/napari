FROM ubuntu:latest

# install python resources
RUN apt-get update && apt-get install -qqy build-essential python3.8 python3-pip

# below env var required to install libglib2.0-0 non-interactively
ENV TZ America/Los_Angeles
ENV DEBIAN_FRONTEND noninteractive

# install graphical libraries used by qt and vispy
RUN apt-get install -qqy libxi6 libglib2.0-0 fontconfig libgl1-mesa-glx libfontconfig1 libxrender1 libdbus-1-3

# install napari release version
RUN pip3 install napari[all]

# library missing when using provided pyqt version, reversed to an earlier version for now
RUN pip3 install PyQt5==5.11.3

ENTRYPOINT ["python3", "napari"]
