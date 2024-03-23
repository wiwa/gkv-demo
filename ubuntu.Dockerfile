# https://github.com/openucx/ucx/blob/master/buildlib/dockers/ubuntu-release.Dockerfile
# docker run --rm -it --runtime=nvidia --gpus all -p 8080:8080 ucx-base:latest 'python3 server.py'
ARG CUDA_VERSION=11.8.0
ARG UBUNTU_VERSION=20.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ARG KV_PORT=8080

ARG NV_DRIVER_VERSION=535.161.07
RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata && \
  apt-get install -y \
  apt-file \
  automake \
  default-jdk \
  dh-make \
  g++ \
  git \
  openjdk-8-jdk \
  libcap2 \
  libnuma-dev \
  libtool \
  # Provide CUDA dependencies by libnvidia-compute*
  libnvidia-compute-${NV_DRIVER_VERSION} \
  make \
  maven \
  pkg-config \
  udev \
  wget \
  environment-modules \
  # Remove cuda-compat* from nvidia/cuda:x86_64 images, provide CUDA dependencies by libnvidia-compute* instead
  && apt-get remove -y openjdk-11-* cuda-compat* || apt-get autoremove -y \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# https://stackoverflow.com/questions/58269375/how-to-install-packages-with-miniconda-in-dockerfile
# Conda
# not sure why wget got wiped
RUN apt-get update && apt-get install -y wget
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
RUN /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
  rm /tmp/miniconda.sh && \
  echo "export PATH=/opt/conda/bin:$PATH" > /etc/profile.d/conda.sh
ENV PATH /opt/conda/bin:$PATH

RUN conda create --solver=libmamba -n cenv -c conda-forge -c rapidsai -c nvidia \
  cudatoolkit=${CUDA_VERSION} ucx>=1.14.0 ucx-py python=3.10 cupy pandas cuda-version=${CUDA_VERSION}

RUN apt-get install -y vim net-tools 

EXPOSE ${KV_PORT}
# SHELL ["conda", "run", "--no-capture-output", "-n", "cenv", "/bin/bash", "-c"]

# https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/
# MOFED
# ARG MOFED_VERSION=5.0-1.0.0.0
# ARG MOFED_VERSION=23.10-2.1.3.1
# ARG UBUNTU_VERSION=20.04
# ARG MOFED_OS=ubuntu${UBUNTU_VERSION}
# ARG ARCH=x86_64
# ENV MOFED_DIR MLNX_OFED_LINUX-${MOFED_VERSION}-${MOFED_OS}-${ARCH}
# ENV MOFED_SITE_PLACE MLNX_OFED-${MOFED_VERSION}
# ENV MOFED_IMAGE ${MOFED_DIR}.tgz
# RUN wget --no-verbose http://content.mellanox.com/ofed/${MOFED_SITE_PLACE}/${MOFED_IMAGE} && \
#   tar -xzf ${MOFED_IMAGE}
# RUN ${MOFED_DIR}/mlnxofedinstall --all -q \
#   --user-space-only \
#   --without-fw-update \
#   --skip-distro-check \
#   --without-ucx \
#   --without-hcoll \
#   --without-openmpi \
#   --without-sharp && \
#   rm -rf ${MOFED_DIR} && rm -rf *.tgz

ENV CPATH /usr/local/cuda/include:${CPATH}
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH /usr/local/cuda/lib64:${LIBRARY_PATH}

COPY client.py .
COPY server.py .
COPY limap.cu .
RUN conda init
ENV UCX_TLS=tcp,cuda_copy
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "cenv", "/bin/bash", "-c"]
