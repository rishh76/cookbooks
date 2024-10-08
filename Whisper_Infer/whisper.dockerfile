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

# Install any needed packages specified in requirements.txt
COPY requirements.txt .

RUN pip install -r requirements.txt

# Copy the current directory contents into the container at /app
COPY src/. /app

# Set environment variables
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

# Run the Python script with arguments passed to the Docker container
ENTRYPOINT ["python3", "monster.py"]
