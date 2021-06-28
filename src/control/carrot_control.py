import threading

import numpy
import rospy
from dynamic_reconfigure.server import Server
from hippocampus_msgs.msg import PathFollowerTarget
from geometry_msgs.msg import PoseStamped
from hippocampus_common.node import Node
from nav_msgs.msg import Odometry
from path_planning.path_planning import Path
from std_msgs.msg import Float64

from control.cfg import CarrotControlConfig


class PathFollowerNode(Node):
    def __init__(self, name):
        super(PathFollowerNode, self).__init__(name)
        self.data_lock = threading.RLock()
        self.look_ahead_distance = 0.0
        self.carrot_dyn_server = Server(CarrotControlConfig,
                                        self._on_reconfigure)
        self.path = Path()
        self.path.update_path_from_param_server()

        self.yaw_pub = rospy.Publisher("yaw_angle", Float64, queue_size=1)
        self.target_pub = rospy.Publisher("~target",
                                          PathFollowerTarget,
                                          queue_size=30)

        self.use_ground_truth = self.get_param("~use_ground_truth",
                                               default=False)
        if self.use_ground_truth:
            self.ground_truth_sub = rospy.Subscriber("ground_truth/state",
                                                     Odometry,
                                                     self.on_local_pose,
                                                     queue_size=1)
        else:
            self.local_pose_sub = rospy.Subscriber("mavros/local_position/pose",
                                                   PoseStamped,
                                                   self.on_local_pose,
                                                   queue_size=1)

    def _on_reconfigure(self, config, level):
        with self.data_lock:
            self.look_ahead_distance = config["look_ahead_dist"]
        return config

    def on_local_pose(self, msg):
        if self.use_ground_truth:
            position = msg.pose.pose.position
        else:
            position = msg.pose.position
        p = numpy.array([position.x, position.y, position.z])
        with self.data_lock:
            if self.path.update_target(
                    position=p,
                    look_ahead_distance=self.look_ahead_distance,
                    loop=True,
                    ignore_z=True):
                target = self.path.get_target_point()
                diff = target - p
                # ignore z coordinate, because z position is handled by the
                # depth controller.
                angle = self.angle(diff[:2])
                self.yaw_pub.publish(Float64(angle))
                self.publish_debug(current=p, target=target)
            else:
                rospy.logwarn("[%s] Could not update target position.",
                              rospy.get_name())

    def angle(self, vec):
        return numpy.arctan2(vec[1], vec[0])

    def publish_debug(self, current, target):
        msg = PathFollowerTarget()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        (msg.current_position.x, msg.current_position.y,
         msg.current_position.z) = current
        (msg.target_position.x, msg.target_position.y,
         msg.target_position.z) = target

        self.target_pub.publish(msg)
