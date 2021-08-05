import threading

import numpy
import rospy
import tf.transformations
from hippocampus_common.node import Node
from dynamic_reconfigure.server import Server
from control.cfg import GeometricControlConfig
from geometry_msgs.msg import TwistStamped, PoseStamped
from mavros_msgs.msg import AttitudeTarget


class AttitudeController(object):
    def __init__(self):
        self.p_gains = numpy.array([1.0, 1.0, 1.0], dtype=float)
        self.d_gains = numpy.array([0.1, 0.1, 0.1], dtype=float)

    def update(self, q_actual, q_desired, omega_actual, omega_desired):
        R = tf.transformations.quaternion_matrix(q_actual)
        R_desired = tf.transformations.quaternion_matrix(q_desired)
        error_R = (numpy.matmul(R_desired.transpose() * R) -
                   numpy.matmul(R.transpose(), R_desired)) * 0.5
        error_R_vec = numpy.array([error_R[2, 1], error_R[0, 2], error_R[1, 0]],
                                  dtype=float)
        error_omega = omega_actual - omega_desired

        T = error_R_vec * self.p_gains - error_omega * self.d_gains
        return numpy.clip(T, -1.0, 1.0)


class AttitudeControllerNode(Node):
    def __init__(self, name, anonymous=False, disable_signals=False):
        super().__init__(name=name,
                         anonymous=anonymous,
                         disable_signals=disable_signals)
        self.data_lock = threading.RLock()
        self.controller = AttitudeController()

        self.geom_reconfigure_server = Server(GeometricControlConfig,
                                              self._on_geom_reconfigure)

        self.omega = numpy.array([0.0, 0.0, 0.0], dtype=float)
        self.omega_desired = numpy.array([0.0, 0.0, 0.0], dtype=float)
        self.attitude = numpy.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        self.attitude_desired = numpy.array([0.0, 0.0, 0.0, 1.0], dtype=float)

        self.velocity_sub = rospy.Subscriber(
            "mavros/local_position/velocity_body", TwistStamped,
            self.on_velocity_body)
        self.attitude_sub = rospy.Subscriber("mavros/local_position/pose",
                                             PoseStamped, self.on_mavros_pose)

        self.setpoint_sub = rospy.Subscriber("~setpoint", AttitudeTarget,
                                             self.on_attitude_setpoint)

    def on_mavros_pose(self, msg: PoseStamped):
        q = msg.pose.orientation
        with self.data_lock:
            self.attitude[:] = [q.x, q.y, q.z, q.w]
        # TODO apply control

    def on_velocity_body(self, msg: TwistStamped):
        omega = msg.twist.angular
        with self.data_lock:
            self.omega[:] = [omega.x, omega.y, omega.z]

    def on_attitude_setpoint(self, msg: AttitudeTarget):
        with self.data_lock:
            q = msg.orientation
            r = msg.body_rate
            self.attitude_desired[:] = [q.x, q.y, q.z, q.w]
            self.omega_desired[:] = [r.x, r.y, r.z]

    def _on_geom_reconfigure(self, config, level):
        with self.data_lock:
            self.controller.p_gains[:] = [
                config["P_roll"], config["P_pitch"], config["P_yaw"]
            ]
            self.controller.d_gains[:] = [
                config["D_roll"], config["D_pitch"], config["D_yaw"]
            ]
        return config
