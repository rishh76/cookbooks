FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ENV LANG C.UTF-8
ENV PYTHONUNBUFFERED 1
RUN rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update -y&& \
    DEBIAN_FRONTEND=noninteractive apt install -y git curl  \
    python3-dev  \
    python3-pip

# Set the working directory in the container
WORKDIR /app

COPY src/. /app

# Make setup.sh executable
RUN chmod +x setup.sh

# Run setup.sh (assuming it installs dependencies)
RUN ./setup.sh

# Capture arguments for run.py
CMD ["python", "run.py"]
