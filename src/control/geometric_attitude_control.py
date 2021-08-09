import threading

import numpy
import rospy
import tf.transformations
from geometry_msgs.msg import PoseStamped, TwistStamped
from hippocampus_common.node import Node
from hippocampus_msgs.msg import ActuatorControls
from mavros_msgs.msg import AttitudeTarget
from mavros_msgs.srv import ParamGet

P_GAIN_NAMES = ("UUV_ROLL_P", "UUV_PITCH_P", "UUV_YAW_P")
D_GAIN_NAMES = ("UUV_ROLL_D", "UUV_PITCH_D", "UUV_YAW_D")


class AttitudeController(object):
    def __init__(self):
        self.p_gains = numpy.array([1.0, 1.0, 1.0], dtype=float)
        self.d_gains = numpy.array([0.1, 0.1, 0.1], dtype=float)

    def update(self, q_actual, q_desired, omega_actual, omega_desired):
        R = tf.transformations.quaternion_matrix(q_actual)
        R_desired = tf.transformations.quaternion_matrix(q_desired)
        error_R = (numpy.matmul(R_desired.transpose(), R) -
                   numpy.matmul(R.transpose(), R_desired)) * 0.5
        error_R_vec = numpy.array([error_R[2, 1], error_R[0, 2], error_R[1, 0]],
                                  dtype=float)
        error_omega = omega_actual - omega_desired
        # dont know why i have to change the sign for roll
        error_omega[0] = -error_omega[0]

        T = error_R_vec * self.p_gains + error_omega * self.d_gains
        return numpy.clip(T, -1.0, 1.0)


class AttitudeControllerNode(Node):
    def __init__(self, name, anonymous=False, disable_signals=False):
        super().__init__(name=name,
                         anonymous=anonymous,
                         disable_signals=disable_signals)
        self.data_lock = threading.RLock()
        self.controller = AttitudeController()

        self.get_px4_param = rospy.ServiceProxy("mavros/param/get", ParamGet)

        rospy.Timer(rospy.Duration(2), self.update_px4_params)

        self.omega = numpy.array([0.0, 0.0, 0.0], dtype=float)
        self.omega_desired = numpy.array([0.0, 0.0, 0.0], dtype=float)
        self.attitude = numpy.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        self.attitude_desired = numpy.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        self.thrust = 0.3

        # topic name corresponds to PX4's µORB topic name
        self.control_pub = rospy.Publisher("actuator_controls_0",
                                           ActuatorControls,
                                           queue_size=1)

        self.velocity_sub = rospy.Subscriber(
            "mavros/local_position/velocity_body", TwistStamped,
            self.on_velocity_body)
        self.attitude_sub = rospy.Subscriber("mavros/local_position/pose",
                                             PoseStamped, self.on_mavros_pose)

        self.setpoint_sub = rospy.Subscriber("~setpoint", AttitudeTarget,
                                             self.on_attitude_setpoint)

    def update_px4_params(self, _):
        for i, param in enumerate(P_GAIN_NAMES):
            try:
                ret = self.get_px4_param(param_id=param)
            except rospy.ServiceException:
                rospy.logerr("Service call failed to get '{}'".format(param))
            else:
                if ret.success:
                    with self.data_lock:
                        self.controller.p_gains[i] = ret.value.real
                else:
                    rospy.logerr("Could not get param '{}'".format(param))

        for i, param in enumerate(D_GAIN_NAMES):
            try:
                ret = self.get_px4_param(param_id=param)
            except rospy.ServiceException:
                rospy.logerr("Service call failed to get '{}'".format(param))
            else:
                if ret.success:
                    with self.data_lock:
                        self.controller.d_gains[i] = ret.value.real

    def on_mavros_pose(self, msg: PoseStamped):
        controls = ActuatorControls()
        q = msg.pose.orientation
        with self.data_lock:
            self.attitude[:] = [q.x, q.y, q.z, q.w]
            u = self.controller.update(self.attitude, self.attitude_desired,
                                       self.omega, self.omega_desired)
        controls.header.stamp = rospy.Time.now()
        controls.control[0:3] = u
        controls.control[3] = self.thrust
        self.control_pub.publish(controls)

    def on_velocity_body(self, msg: TwistStamped):
        omega = msg.twist.angular
        with self.data_lock:
            self.omega[:] = [omega.x, omega.y, omega.z]

    def on_attitude_setpoint(self, msg: AttitudeTarget):
        with self.data_lock:
            q = msg.orientation
            r = msg.body_rate
            self.thrust = msg.thrust
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
