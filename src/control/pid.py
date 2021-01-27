class Controller():
    # TODO: Implement an antiwindup for the integral. The integral should be
    # frozen if the output control output is saturated and control error and
    # integral have the same sign.
    #
    def __init__(self, p_gain=1.0, i_gain=0.0, d_gain=0.0):
        self.p_gain = p_gain
        self.i_gain = i_gain
        self.d_gain = d_gain
        self.saturation = [-100, 100]
        self.integral_limits = [-100, 100]
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
        delta_integral = dt * error * self.i_gain
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
