# zugzwang

> A Python3-based Chess game recording system using OpenCV deployed in a Raspberry Pi. 

> Convolutional Neural Network model training with TensorFlow and Keras

> Inference via TensorFlow Lite

## Table of Contents

- [Installation](#installation)

---

## Installation

- Raspberry Pi
    - pip3 install opencv-contrib-python==3.4.6.27
    - pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl
    - pip3 install cairosvg

### Docker

- docker pull tensorflow/tensorflow:latest-jupyter
- docker run -it --rm -v /c/zugzwang:/tf/zugzwang -p 8888:8888 tensorflow/tensorflow:latest-jupyter

### Jupyter Notebook for Training

- http://192.168.99.100:8888/tree
- To access the notebook, open this link in a browser:
http://192.168.99.100:8888/?token=replace_this_with_actual_token_value

Copyright © 2020 Rennel Ongcoy (Eli). All rights reserved.