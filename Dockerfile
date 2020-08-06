FROM nvidia/cudagl:9.0-devel-ubuntu16.04

ENV CUDA_ARCH_BIN "30 35 50 52 60"
ENV CUDA_ARCH_PTX "60"

# Install dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        libatlas-base-dev \
        libatlas-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-dev \
        libprotobuf-dev \
        pkg-config \
        protobuf-compiler \
        python-yaml \
        python-six \
        wget && \
    rm -rf /var/lib/apt/lists/*

ENV OPENCV_VERSION 3.4.0
RUN git clone --depth 1 -b ${OPENCV_VERSION} https://github.com/opencv/opencv.git /opencv

RUN mkdir /opencv/build && cd /opencv/build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
          -DWITH_CUDA=ON -DWITH_CUFFT=OFF -DCUDA_ARCH_BIN="${CUDA_ARCH_BIN}" -DCUDA_ARCH_PTX="${CUDA_ARCH_PTX}" \
          -DWITH_JPEG=ON -DBUILD_JPEG=ON -DWITH_PNG=ON -DBUILD_PNG=ON \
          -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DWITH_FFMPEG=OFF -DWITH_GTK=OFF \
          -DWITH_OPENCL=OFF -DWITH_QT=OFF -DWITH_V4L=OFF -DWITH_JASPER=OFF \
          -DWITH_1394=OFF -DWITH_TIFF=OFF -DWITH_OPENEXR=OFF -DWITH_IPP=OFF -DWITH_WEBP=OFF \
          -DBUILD_opencv_superres=OFF -DBUILD_opencv_java=OFF -DBUILD_opencv_python2=OFF \
          -DBUILD_opencv_videostab=OFF -DBUILD_opencv_apps=OFF -DBUILD_opencv_flann=OFF \
          -DBUILD_opencv_ml=OFF -DBUILD_opencv_photo=OFF -DBUILD_opencv_shape=OFF \
          -DBUILD_opencv_cudabgsegm=OFF -DBUILD_opencv_cudaoptflow=OFF -DBUILD_opencv_cudalegacy=OFF \
          -DCUDA_NVCC_FLAGS="-O3" -DCUDA_FAST_MATH=ON .. && \
    make -j"$(nproc)" install && \
    ldconfig && \
    rm -rf /opencv


# cuda-samples-9-1 is required for the helper_math.h
RUN apt-get update && apt-get install -y --no-install-recommends \
    unzip \
    cuda-samples-9-1  && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install libglm-dev

RUN git clone -b glew-2.1.0 https://github.com/nigels-com/glew.git
RUN wget https://github.com/nigels-com/glew/releases/download/glew-2.1.0/glew-2.1.0.tgz
RUN tar -xzf glew-2.1.0.tgz

RUN cd glew-2.1.0; make; make install
RUN rm -r glew-2.1.0*

# A modified version of Caffe is used to properly handle multithreading and CUDA streams.
RUN git clone --depth 1 https://github.com/BVLC/caffe.git /caffe && \
    cd /caffe && \
    cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
          -DCUDA_ARCH_NAME=Manual -DCUDA_ARCH_BIN="${CUDA_ARCH_BIN}" -DCUDA_ARCH_PTX="${CUDA_ARCH_PTX}" \
          -DUSE_CUDNN=ON -DUSE_OPENCV=ON -DUSE_LEVELDB=OFF -DUSE_LMDB=OFF \
          -DBUILD_python=OFF -DBUILD_python_layer=OFF -DBUILD_matlab=OFF \
          -DCMAKE_INSTALL_PREFIX=/usr/local \
          -DCUDA_NVCC_FLAGS="--default-stream per-thread -O3" && \
    make -j"$(nproc)" install && \
    make clean

RUN apt-get update && apt-get install -y --no-install-recommends libleveldb-dev libsnappy-dev  liblmdb-dev
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends xserver-xorg-core 	xserver-xorg-core-dbg
RUN apt-get update && apt-get install -y --no-install-recommends xserver-xorg-dev
RUN apt-get update && apt-get install -y --no-install-recommends xorg-dev

# install glfw
RUN wget https://github.com/glfw/glfw/releases/download/3.2.1/glfw-3.2.1.zip; unzip glfw-3.2.1.zip
RUN cd glfw-3.2.1; mkdir glfw_build; cd glfw_build; cmake .. -DBUILD_SHARED_LIBS=ON; make install -j"$(nproc)"
RUN rm -r glfw-3.2.1*

# **********************************************************
ENV CUDNN_VERSION 7.1.1.5
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
            libcudnn7=$CUDNN_VERSION-1+cuda9.0 \
            libcudnn7-dev=$CUDNN_VERSION-1+cuda9.0 && \
    rm -rf /var/lib/apt/lists/*
# **********************************************************

# update cmake to find EGL
RUN wget https://github.com/Kitware/CMake/releases/download/v3.18.1/cmake-3.18.1.tar.gz
RUN tar -xzf cmake-3.18.1.tar.gz; cd cmake-3.18.1; mkdir build; cd build;
RUN cmake ../; make; make install; cd /; rm -r cmake-3.18.1*

