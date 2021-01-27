#!/usr/bin/env python
import rospy
import threading
from hippocampus_common.node import Node
from control import pid
from control.cfg import PidControlConfig
from dynamic_reconfigure.server import Server
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64


class DepthControlNode(Node):
    def __init__(self, name):
        super(DepthControlNode, self).__init__(name=name)
        self.data_lock = threading.RLock()
        self.controller = pid.Controller()

        self.target = 0.0
        self.t_last = rospy.get_time()

        self.pitch_angle_pub = rospy.Publisher("pitch_angle",
                                               Float64,
                                               queue_size=1)
        self.dyn_server = Server(PidControlConfig, self.on_reconfigure)
        self.local_pose_sub = rospy.Subscriber("mavros/local_position/pose",
                                               PoseStamped,
                                               self.on_local_pose,
                                               queue_size=1)

    def on_reconfigure(self, config, level):
        with self.data_lock:
            self.controller.p_gain = config["p"]
            self.controller.i_gain = config["i"]
            self.controller.d_gain = config["d"]
            config["p"] = self.controller.p_gain
            config["i"] = self.controller.i_gain
            config["d"] = self.controller.d_gain
        return config

    def on_local_pose(self, msg):
        z = msg.pose.position.z
        with self.data_lock:
            error = z - self.target
            now = msg.header.stamp.to_sec()
            # avoid too large dt if no poses were published and unreasonable
            # small dt.
            dt = max(min(now - self.t_last, 0.1), 0.01)
            self.t_last = now
            u = self.controller.update(error, dt)
            self.pitch_angle_pub.publish(Float64(u))


def main():
    node = DepthControlNode("depth_controller")
    node.run()


if __name__ == "__main__":
    main()
