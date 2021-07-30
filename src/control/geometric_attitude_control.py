import threading

import numpy
import rospy
import tf.transformations
from hippocampus_common.node import Node


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

        self.vehicle_type = self.get_param("vehicle_type")
        if self.vehicle_type is None:
            rospy.logfatal("No vehicle_type param specified. Exiting")
            exit(1)
