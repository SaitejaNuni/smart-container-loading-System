"""
Random Package Generator Node.
Serves packages on-demand via /get_next_package service.
For each service request, generates one random PackageInfo with values
within configured min/max ranges. Optionally also calls the MuJoCo bridge
to physically spawn the package in simulation.
Length, width, height are randomized independently (a package can have
different values for each of its three dimensions).
Parameters:
  min_size (float, default 0.15): min dimension in meters
  max_size (float, default 0.50): max dimension in meters
  min_mass (float, default 0.5): min mass in kg
  max_mass (float, default 10.0): max mass in kg (capped for UR10e payload)
  spawn_in_mujoco (bool, default True): also call /spawn_package on bridge
  max_packages (int, default 0): if >0, generator returns has_next=False
                                  once this many have been generated.
                                  0 means unlimited.
"""
import random
import rclpy
from rclpy.node import Node
from thesis_msgs.msg import PackageInfo
from thesis_msgs.srv import GetNextPackage
from std_srvs.srv import Trigger
class RandomPackageGenerator(Node):
    """Generates random packages on-demand."""
    def __init__(self):
        super().__init__('random_package_generator')
        # ---- Parameters ----
        self.declare_parameter('min_size', 0.15)
        self.declare_parameter('max_size', 0.50)
        self.declare_parameter('min_mass', 0.5)
        self.declare_parameter('max_mass', 10.0)
        self.declare_parameter('spawn_in_mujoco', False)
        self.declare_parameter('max_packages', 0)
        self.min_size = self.get_parameter('min_size').value
        self.max_size = self.get_parameter('max_size').value
        self.min_mass = self.get_parameter('min_mass').value
        self.max_mass = self.get_parameter('max_mass').value
        self.spawn_in_mujoco = self.get_parameter('spawn_in_mujoco').value
        self.max_packages = self.get_parameter('max_packages').value
        # ---- State ----
        self.package_counter = 0
        # ---- MuJoCo bridge client (optional) ----
        self.spawn_client = None
        if self.spawn_in_mujoco:
            self.spawn_client = self.create_client(Trigger, '/spawn_package')
            self.get_logger().info(
                'Will call /spawn_package on bridge when packages are generated'
            )
        # ---- Service server ----
        self.srv = self.create_service(
            GetNextPackage, '/get_next_package', self.on_get_next_package
        )
        limit_str = (f'{self.max_packages} max' if self.max_packages > 0
                     else 'unlimited')
        self.get_logger().info(
            f'Random package generator ready. '
            f'Size range: {self.min_size*1000:.0f}-{self.max_size*1000:.0f}mm, '
            f'mass range: {self.min_mass:.1f}-{self.max_mass:.1f}kg, '
            f'limit: {limit_str}'
        )
    def on_get_next_package(self, request, response):
        """Generate one random package on demand.
        
        Dimensions are sorted so length >= width >= height. This matches the
        pool generator convention (smallest dim vertical when spawned in
        MuJoCo with identity quaternion) and keeps orientation reasoning
        consistent across the pipeline.
        
        If max_packages is set and reached, returns has_next=False.
        """
        # Check limit
        if self.max_packages > 0 and self.package_counter >= self.max_packages:
            self.get_logger().info(
                f'Package limit reached ({self.max_packages}). '
                f'Returning has_next=False.'
            )
            response.has_next = False
            return response
        self.package_counter += 1
        # Three independent random dims, then sorted largest-to-smallest
        dims = sorted(
            [random.uniform(self.min_size, self.max_size) for _ in range(3)],
            reverse=True
        )
        pkg = PackageInfo()
        pkg.id = f'pkg_{self.package_counter:03d}'
        pkg.length = dims[0]   # largest
        pkg.width = dims[1]    # middle
        pkg.height = dims[2]   # smallest
        pkg.mass = random.uniform(self.min_mass, self.max_mass)
        response.has_next = True
        response.package = pkg
        self.get_logger().info(
            f'Generated {pkg.id}: '
            f'{pkg.length*1000:.0f} x {pkg.width*1000:.0f} x {pkg.height*1000:.0f} mm, '
            f'{pkg.mass:.2f} kg'
        )
        # Also spawn in MuJoCo (async, don't block on response)
        if self.spawn_client is not None:
            if self.spawn_client.service_is_ready():
                self.spawn_client.call_async(Trigger.Request())
            else:
                self.get_logger().warn(
                    '/spawn_package service not ready, skipping MuJoCo spawn'
                )
        return response
def main(args=None):
    rclpy.init(args=args)
    node = RandomPackageGenerator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
if __name__ == '__main__':
    main()
