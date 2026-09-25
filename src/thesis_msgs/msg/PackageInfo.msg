# Information about a package that needs to be placed in the container.
# Published by random_package_generator_node (or real sensors in deployment).
# Subscribed by placement_planner_node.

string id           # Unique package identifier (e.g., "pkg_001")
float32 length      # X dimension (meters)
float32 width       # Y dimension (meters)
float32 height      # Z dimension (meters)
float32 mass        # Weight (kilograms)
