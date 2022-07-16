FROM tensorflow/tensorflow:latest-gpu

COPY docker/requirements.txt requirements.txt

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential\
         libsm6 \
         libxext6 \
         libxrender-dev \
         ffmpeg  \
         graphviz \
         tree \
    python3-pip &&\
    pip install --upgrade pip

RUN pip install -r requirements.txt

WORKDIR /app/work/
COPY src/ src/

CMD ["bash"]
