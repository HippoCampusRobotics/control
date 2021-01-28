#!/usr/bin/env python
import threading

import rospy
from control import pid
from control.cfg import PidControlConfig
from dynamic_reconfigure.server import Server
from geometry_msgs.msg import PoseStamped
from hippocampus_common.node import Node
from std_msgs.msg import Float64


class DepthControlNode(pid.PidNode):
    def __init__(self, name):
        super(DepthControlNode, self).__init__(name=name)

        self.pitch_angle_pub = rospy.Publisher("pitch_angle",
                                               Float64,
                                               queue_size=1)
        self.local_pose_sub = rospy.Subscriber("mavros/local_position/pose",
                                               PoseStamped,
                                               self.on_local_pose,
                                               queue_size=1)

    def on_local_pose(self, msg):
        z = msg.pose.position.z
        now = msg.header.stamp.to_sec()
        with self.data_lock:
            error = z - self.setpoint
            u = self.update_controller(error=error, now=now)

        self.pitch_angle_pub.publish(Float64(u))


def main():
    node = DepthControlNode("depth_controller")
    node.run()


if __name__ == "__main__":
    main()
