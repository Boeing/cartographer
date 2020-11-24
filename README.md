# Boeing Cartographer

Modular Cartographer is a Boeing fork of Google's [Cartographer](https://github.com/googlecartographer/cartographer) to suit factory environments.
Cartographer is a real-time simultaneous localization and mapping (SLAM) system for mobile robots.

This repository holds the changes made by Boeing for fast robust 2D SLAM for factory environments.

## Why not upstream?

Using upstream cartographer for our use cases, we identified three main issues:

- Noisy odometry estimation
- Poor performing constraint finding
- Complex background task management

**Noisy Odometry**

We observed that the way which cartographer estimates odometry is noisy. This results in blurry maps. The code in question is found in `PoseExtrapolator`. The job of the `PoseExtrapolator` is to estimate the pose of the robot for front end mapping. Estimated robot poses are required to first project the points of a rotating laser (due to the motion of the vehicle) and then seed the Ceres scan matcher (to match the laser scan to the current submap). Once the scan matcher has run, the front end mapper is able to pass this data to the back end to generate a node in the pose graph. Inaccuracy in the estimated pose of the robot will first cause blurry projected points for a spinning lidar, and secondly potentially throw off the Ceres scan matcher with a bad prior.

The original implementation sent both IMU and Odometry data to the `PoseExtrapolator` class. In addition, each time a Ceres scan match was performed this updated pose was sent to the `PoseExtrapolator` as a “node” (used as a reference point). Extrapolation was implemented as a combination of both IMU and Odometry to estimate the velocity of the robot at any point in time, which essentially integrated the position of the robot since the last inserted “node”. The IMU data was smoothed with an exponential decaying moving average filter, the Odometry data was not smoothed. The Odometry data is missing velocity, so velocity is estimated with instantaneous calculations. Overall this approach produced noisy odometry. This is most evident in a simulator where perfect data yields blurry maps.

The solution was quite straightforward. We replaced this implementation with a rolling buffer (sufficiently sized) of Odometry messages. Velocity information was added to the Odometry struct. A single “node” is kept as a reference point in the same way as before. When the robot pose is required at a particular time we traverse the rolling buffer and interpolate the correct reference pose at the “node” time, then do the same for the queried time. The extrapolated pose is the “node” pose plus the transform between the two interpolated poses. This results in perfectly interpolated odometry since the last successfully scan matched “node”. Yielding perfect maps given perfect data.

**Poor Performing Constraint Finding**

Constraint finding is the process of taking the current robot scan and attempting to match this to a previously generated submap. Cartographer was developed with a novel branch-and-bound constraint finding algorithm. This approach uses a stack of decreasing resolution copies of a submap. The idea of the algorithm is to evaluate the alignment of all possible robot poses (across a submap) by matching the scan to progressively higher resolution layers. If the alignment is poorer than the best current alignment, all higher resolution layers can be disregarded for that pose (the bounding stage of a branch-and-bound).

This algorithm has some limitations:
1. The branch-and-bound technique can't sidestep rotational complexity (constrained by the map resolution). All possible rotations of a proposal must be considered (expensive).
2. The branch-and-bound algorithm must test all possible cells on the map if there is self-similarity (unnecessary computation to test impossible or improbable locations).
3. Poor behavior is very hard to visualize or debug (due to the tree search, one cannot clearly label why certain areas were not explored).

In practice, a 2cm resolution map would take ~30seconds to resolve a constraint. If there was self-similarity (forcing exploration up resolution layers) this would blow out to minutes of computation time.

Our solution was to develop a stochastic constraint finding algorithm - Monte Carlo Global ICP (described below). This algorithm relies on meta information about the environment (free space, features, ray-casting) to quickly eliminate randomly sampled proposals. Randomness helps to improve robustness across time without exhaustively searching (which is slow). Good proposals are sent to ICP for refinement.

**Complex Background Task Management**

The third issue with upstream cartographer is the complexity of its task management. There are two major components in the implementation, a front-end mapper, and a back-end pose graph. The front-end mapper is responsible for building submaps, the back-end is responsible for building the pose graph, finding constraints, and optimizing the position of the trajectory nodes and submaps.

Upstream cartographer implements a task queue and thread pool in the back-end component. We observed delays and deadlocks with this approach.

Our solution was to:
- Remove the task queue and thread pool
- Make all fast API operations synchronous (add node, etc.)
- Copy the pose graph data for background optimization (to prevent holding the data mutex)
- Implement a single background thread for constraint searching and optimization.

This simple single background thread allows for more sophisticated heuristics and logic to be added to the process of constraint finding and optimization. Decisions like whether to search globally or locally and how far to search for constraints, can be easily made and modified.

## Monte Carlo Global ICP

We required a fast and robust algorithm to evaluate if a scan matches a submap. This algorithm is required to find constraints to previously generated submaps with or without a prior pose.

Process
1. Randomly sample proposals across the free space of the submap
2. Select the good proposals (with fast short-circuit logic) based on a number of heuristics (average point-to-map distance, ray-cast checks, average feature hit distance, point inliers, etc.)
3. Cluster the good proposals with DBScan and computed the weighted center.
4. Run ICP (with features) on the cluster centers.
5. Select the best ICP convergence
6. Test the pose based on a number of heuristics (hit, ray-cast, etc.) to determine score and suitability of constraint

Randomness is used to minimize the number of proposals required to find a suitable prior for ICP. Clustering helps to reduce the number of ICP attempts. ICP is excellent at converging to the best match (if started near the solution).

## Changes from upstream

- Add 2D Submap features
  - Match reflective poles for robust constraint matching
- Add `GlobalICPScanMatcher2D`
  - Fast sampling based global constraint finder
  - Significantly faster than `FastCorrelativeScanMatcher2D` (1s verse 30s)
- Add `ICPScanMatcher2D`
  - Fast dense point matcher
  - Allows for significant deviation from local minima
  - Inclusion of 2d features (in addition to points)
  - Match evaluation based on raytracing and hit (of all points, not just subsampled points)
  - Match evaluation based on feature match
- Optimise `PoseExtrapolator` for wheeled odometry rather than IMU
  - Achieve perfect maps in sim
  - Resolve issues with rotations / poor local mapping
- Remove 3d mapping
- Add heuristics for performant constraint finding
  - Desired number of constraints per submap / trajectory
  - Maximum work queue size
- RangeDataCollator strategy
  - Use a 'one-of-each' strategy rather than time based
- Simplify background task managmenet
  - Remove the complex task queue and thread pool, replace with a single background thread

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
CMakeLists.txt.

Above the line
```bash
if (NOT DEFINED CARTOGRAPHER_INSTALL_DIR)
```
add
```
set(CARTOGRAPHER_INSTALL_DIR /home/boeing/ros/cartographer/build/install)
```

## Authors
Google Inc.

The Boeing Company

Phillip Haeusler (Boeing Fork Author)

William Ko (Boeing Fork Maintainer - william.ko@boeing.com)

Alexandre Desbiez

## License
Original Copyright 2020 Google Inc.
Changes Copyright 2020 The Boeing Company

Licensed under the Apache License, Version 2.0 (the "License") with the following modification;
you may not use this file except in compliance with the Apache License and the following modification to it:

(Appended as Section 10)

By accepting this software, recipient agrees that the representations, warranties, obligations, and liabilities of The Boeing Company set forth in this software, if any, are exclusive and in substitution for all other all other representations, warranties, obligations, and liabilities of The Boeing Company.
Recipient hereby waives, releases, and renounces all other rights, remedies, and claims (including tortious claims for loss of or damage to property) of recipient against The Boeing Company with respect to this software.
The Boeing Company hereby disclaims all implied warranties, including but not limited to, all implied warranties of merchantability, fitness, course of dealing, and usage of trade.
The Boeing Company shall have no liability, whether arising in contract (including warranty), tort (whether or not arising from the negligence of The Boeing Company), or otherwise, for any loss of use, revenue, or profit, or for any other unspecified direct, indirect, incidental, or consequential damages for anything delivered or otherwise provided by The Boeing Company under this software.

You may obtain a copy of the original, unmodified License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Contributing
Any contribution that you make to this repository will
be under the Modified Apache 2 License, as dictated by that
[license](http://www.apache.org/licenses/LICENSE-2.0):

```
5. Submission of Contributions. Unless You explicitly state otherwise,
   any Contribution intentionally submitted for inclusion in the Work
   by You to the Licensor shall be under the terms and conditions of
   this License, without any additional terms or conditions.
   Notwithstanding the above, nothing herein shall supersede or modify
   the terms of any separate license agreement you may have executed
   with Licensor regarding such Contributions.
```

To contribute, issue a PR and @brta-mszarski (martin.a.szarski@boeing.com).

