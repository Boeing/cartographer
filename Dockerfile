FROM sres.web.boeing.com:5000/ubuntu:bionic

RUN set -ex \
    && apt-get update -y --no-install-recommends \
    && apt-get install -y --no-install-recommends \
        git \
        curl \
        ninja-build \
        build-essential \
        cmake \
        googletest \
        google-mock \
        libceres-dev \
        libboost-iostreams-dev \
        liblua5.2-dev \
        libcairo2-dev \
    && rm -rf /var/lib/apt/lists/*

RUN cd /root \
    && git clone https://github.com/protocolbuffers/protobuf.git -b 'v3.4.1' --single-branch --depth 1 \
    && cd /root/protobuf \
    && mkdir build \
    && cd build \
    && cmake ../cmake -GNinja -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_BUILD_TYPE=Release -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=install \
    && ninja \
    && ninja install

RUN curl -L https://git.web.boeing.com/brta-robotics/ros/modular_cartographer/-/raw/master/cartographer_ros/dependencies/abseil-cpp-7b46e1d31a6b08b1c6da2a13e7b151a20446fa07.tar.gz --insecure -o /root/abseil-cpp.tar.gz

# Copy project source
COPY cartographer /root/cartographer/cartographer
COPY cmake /root/cartographer/cmake
COPY CMakeLists.txt /root/cartographer/CMakeLists.txt
COPY cartographer-config.cmake.in /root/cartographer/cartographer-config.cmake.in
COPY configuration_files /root/cartographer/configuration_files

RUN cd /root/cartographer \
    && mkdir build \
    && cd build \
    && cmake .. -DABSEIL_TAR_PATH=/root/abseil-cpp.tar.gz -DCMAKE_PREFIX_PATH=/root/protobuf/build/install -DCMAKE_INSTALL_PREFIX=install -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTS:BOOL=On \
    && make -j4
