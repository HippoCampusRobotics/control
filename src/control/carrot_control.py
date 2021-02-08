import rospy
import numpy
from hippocampus_common.node import Node
from path_planning.path_planning import Path
from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from dynamic_reconfigure.server import Server
from control.cfg import CarrotControlConfig
import threading


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
        self.target_pub = rospy.Publisher("~target_position",
                                          PointStamped,
                                          queue_size=30)
        self.current_pub = rospy.Publisher("~current_position",
                                           PointStamped,
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
        c_msg = PointStamped()
        c_msg.point.x, c_msg.point.y, c_msg.point.z = current
        c_msg.header.stamp = rospy.Time.now()
        t_msg = PointStamped()
        t_msg.point.x, t_msg.point.y, t_msg.point.z = target
        t_msg.header.stamp = c_msg.header.stamp

        self.target_pub.publish(t_msg)
        self.current_pub.publish(c_msg)
