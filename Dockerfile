FROM debian:latest

# Install required system packages
RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get install -y python3 
RUN apt-get install -y python3-pip
RUN apt-get install -y libsm6
RUN apt-get install -y libxrender-dev
RUN apt-get install -y libxext6

# Install required Python3 packages
RUN pip3 install opencv-contrib-python
RUN pip3 install python-chess
RUN pip3 install tensorflow

RUN apt-get clean

# Set working directory
WORKDIR /
