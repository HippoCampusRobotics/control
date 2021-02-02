import threading

import rospy
from dynamic_reconfigure.server import Server
from hippocampus_common.node import Node
from std_msgs.msg import Float64

from control.cfg import PidControlConfig


class PidNode(Node):
    """A template class for PID controller nodes.
    """
    def __init__(self, name):
        """

        Args:
            name (str): The node's name.
        """
        super(PidNode, self).__init__(name=name)

        #: lock to provide thread safety for non local variables
        self.data_lock = threading.RLock()

        #: The controller's setpoint. In most cases you want to update the value
        #: in a message callback function.
        self.setpoint = 0.0

        self._controller = Controller()
        self._t_last = rospy.get_time()

        self._dyn_reconf_pid = Server(PidControlConfig,
                                      self._on_pid_reconfigure)

        self.setpoint_sub = rospy.Subscriber("~setpoint", Float64,
                                             self._on_setpoint)

    def _on_setpoint(self, msg):
        with self.data_lock:
            self.setpoint = msg.data

    def _on_pid_reconfigure(self, config, level):
        """Callback for the dynamic reconfigure service to set PID control
        specific parameters.

        Args:
            config (dict): Holds parameters and values of the dynamic
                reconfigure config file.
            level (int): Level of the changed parameters

        Returns:
            dict: The actual parameters that are currently applied.
        """
        with self.data_lock:
            self._controller.p_gain = config["p"]
            self._controller.i_gain = config["i"]
            self._controller.d_gain = config["d"]
            self._controller.saturation = [
                config["saturation_lower"], config["saturation_upper"]
            ]
            self._controller.integral_limits = [
                config["integral_limit_lower"], config["integral_limit_upper"]
            ]
            config["p"] = self._controller.p_gain
            config["i"] = self._controller.i_gain
            config["d"] = self._controller.d_gain
            lower, upper = self._controller.saturation
            config["saturation_lower"] = lower
            config["saturation_upper"] = upper
            lower, upper = self._controller.integral_limits
            config["integral_limit_lower"] = lower
            config["integral_limit_upper"] = upper
        return config

    def _update_dt(self, now):
        """Updates the time difference between controller updates.

        Args:
            now (float): Current UTC stamp in seconds.

        Returns:
            float: Time difference between current and last controller update.
        """
        dt = now - self._t_last
        dt_max = 0.1
        dt_min = 0.01
        if dt > dt_max:
            rospy.logwarn(
                "[%s] Timespan since last update too large (%fs)."
                "Limited to %fs", rospy.get_name(), dt, dt_max)
            dt = dt_max
        elif dt < dt_min:
            rospy.logwarn(
                "[%s] Timespan since last update too small (%fs)."
                "Limited to %fs", rospy.get_name(), dt, dt_min)
            dt = dt_min

        self._t_last = now
        return dt

    def update_controller(self, error, now, derror=None):
        """Computes the updated PID controller's output.

        Args:
            error (float): Control error.
            now (float): Current UTC timestamp in seconds.
            derror (float, optional): If set the value is used as derivate of
                the control error. Otherwise a difference quotient is computed.
                Defaults to None.

        Returns:
            float: Control output.
        """
        dt = self._update_dt(now)
        u = self._controller.update(error=error, dt=dt, derror=derror)
        return u


class Controller():
    """A quite normal PID controller.

    PID gains can either be set on the creation of a controller instance or
    via the respective properties any time later on. Compute the control output
    by invoking ``update``.

    Examples:
        ::
            controller = Controller(p_gain=3.4, i_gain=1.1, d_gain=0.1)
            controller.p_gain = 1.0
            print("Controller's p_gain: {}".format(controller.p_gain))

            control_error = 1.0
            control_output = controller.update(error=control_error, dt=0.02)
            print("Control output: {}".format(control_output))
    """
    def __init__(self, p_gain=1.0, i_gain=0.0, d_gain=0.0):
        self.p_gain = p_gain
        self.i_gain = i_gain
        self.d_gain = d_gain
        self.saturation = [-100, 100]
        self.integral_limits = [-1, 1]
        self._integral = 0.0
        self._derivative = 0.0
        self._last_error = 0.0

    def update(self, error, dt, derror=None):
        """Compute the control output.

        Args:
            error (float): Control error.
            dt (float): Timespan used to integrate the control error and
                optionally compute the derivative of the control error.
            derror (float, None, optional): If not None use this argument as
                derivative of the control error instead of computing it.
                Defaults to None.

        Returns:
            float: Control output.
        """
        self._update_integral(error, dt)
        self._update_derivative(error, dt, derror)
        u = (error * self.p_gain + self._integral * self.i_gain +
             self._derivative * self.d_gain)
        u = max(self.saturation[0], min(self.saturation[1], u))
        return u

    def _update_integral(self, error, dt):
        delta_integral = dt * error
        self._integral = max(
            self.integral_limits[0],
            min(self.integral_limits[1], self._integral + delta_integral))

    def _update_derivative(self, error, dt, derror=None):
        """Computes the derivative of the control error.

        Args:
            error (float): The control error. If 'derror' is passed this value
                has no effect.
            dt (float): Timespan to compute the difference quotient.
            derror (float): If not None this value is used as derivative of the
                control error. Otherwise the derivate of computed via difference
                quotient.
        """
        if derror is None:
            self._derivative = (error - self._last_error) / dt
        else:
            self._derivative = derror

    @property
    def saturation(self):
        """Get or set the saturation.

        The saturation limits the control output. Useful in cases where the
        control output is not allowed to exceed certain limits.
        """
        return self._saturation

    @saturation.setter
    def saturation(self, boundaries):
        lower = float(boundaries[0])
        upper = float(boundaries[1])
        if lower > upper:
            self._saturation = [upper, lower]
        else:
            self._saturation = [lower, upper]

    @property
    def integral_limits(self):
        """Get or set the limits of the controller's integral part.

        """
        return self._integral_limits

    @integral_limits.setter
    def integral_limits(self, boundaries):
        lower = float(boundaries[0])
        upper = float(boundaries[1])
        if lower > upper:
            self._integral_limits = [upper, lower]
        else:
            self._integral_limits = [lower, upper]

    @property
    def p_gain(self):
        """Get or set the proportional gain of the controller.

        """
        return self._p_gain

    @p_gain.setter
    def p_gain(self, value):
        self._p_gain = float(value)

    @property
    def i_gain(self):
        """Get or set the integral gain of the controller.

        """
        return self._i_gain

    @i_gain.setter
    def i_gain(self, value):
        self._i_gain = float(value)

    @property
    def d_gain(self):
        """Get or set the derivative gain of the controller.

        """
        return self._d_gain

    @d_gain.setter
    def d_gain(self, value):
        self._d_gain = float(value)
