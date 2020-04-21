# Boeing Cartographer

Based on [goolge-cartographer](https://github.com/googlecartographer/cartographer)

Modified for fast robust 2D SLAM for factory environments.

**Changes from upstream**
- Add 2D Submap features
  - Match reflective poles for robust constraint matching
- Add `GlobalICPScanMatcher2D`
  - Fast sampling based global constraint finder
  - Significantly faster than `FastCorrelativeScanMatcher2D` (1s verse 30s)
- Add `ICPScanMatcher2D`
  - Fast dense point matcher
  - Allows for significant deviation from local minima
  - Inclusion of 2d features (in addition to points)
- Optimise `PoseExtrapolator` for wheeled odometry rather than IMU
  - Achieve perfect maps in sim
  - Resolve issues with rotations / poor local mapping
- Remove 3d mapping
- Add heuristics for performant constraint finding
  - Desired number of constraints per submap / trajectory
  - Maximum work queue size
- RangeDataCollator strategy

## How to Build

**Build protobuf**
```bash
cd /home/boeing/git
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git checkout v3.4.1
mkdir build
cd build
cmake ../cmake -GNinja -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_BUILD_TYPE=Release -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=install
ninja
ninja install
```

**Build cartographer**
- You need to provide the path to an Abseil tar `ABSEIL_TAR_PATH`
- You need to provide the correct version of protobuf on `CMAKE_PREFIX_PATH`
```bash
cd cartographer
mkdir build
cd build
cmake .. -DABSEIL_TAR_PATH=/home/boeing/ros/robotics_ws/src/modular_cartographer/cartographer_ros/dependencies/abseil-cpp-7b46e1d31a6b08b1c6da2a13e7b151a20446fa07.tar.gz -DCMAKE_PREFIX_PATH=/home/boeing/git/protobuf/build/install -DCMAKE_INSTALL_PREFIX=install -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTS:BOOL=Off
make -j8
make install
```

**For development of cartographer**
- Modify `cartographer_ros` cmake to point to the install path for cartographer
CMakeLists.txt
```
# Remove the `externalproject_add`

# Cartographer
set(CARTOGRAPHER_PROJECT_NAME cartographer)
set(CARTOGRAPHER_INSTALL_DIR /home/boeing/ros/cartographer/build/install)
set(CARTOGRAPHER_INCLUDE_DIRS ${CARTOGRAPHER_INSTALL_DIR}/include)
set(CARTOGRAPHER_LIBRARIES ${CARTOGRAPHER_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}cartographer${CMAKE_STATIC_LIBRARY_SUFFIX})

```