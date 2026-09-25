# Placement decision for a single package.
# 
# The algorithm evaluates 6 axis-aligned orientations of the package;
# each orientation permutes which of length/width/height goes along
# which world axis. This message communicates the chosen orientation
# in two complementary ways:
#   - pose.orientation: rotation quaternion (from default to placed)
#   - dim_x/y/z: extents of the placed package along world X/Y/Z
# Downstream code can use either representation.

string package_id

# pose.position: geometric CENTER of the placed package (world frame)
# pose.orientation: rotation of package frame from default to placed
geometry_msgs/Pose pose

# Package extents along world axes after orientation applied.
# Used for orchestrator to compute wrist target (e.g., top of package)
float32 dim_x
float32 dim_y
float32 dim_z

# Score of this placement (higher is better)
float32 score
