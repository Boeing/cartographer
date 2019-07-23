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

