import numpy as np


class Screw:
    """
    A transformation in screw form.

    See "Dual-Quaternions: From Classical Mechanics to Computer Graphics and Beyond" by Ben Kenwright.
    """

    # CONSTRUCTOR

    def __init__(self, angle: float, pitch: float, direction: np.ndarray, moment: np.ndarray):
        """
        Construct a screw transformation.

        :param angle:       The screw angle.
        :param pitch:       The screw pitch.
        :param direction:   The screw direction.
        :param moment:      The screw moment.
        """
        self.angle = angle          # type: float
        self.pitch = pitch          # type: float
        self.direction = direction  # type: np.ndarray
        self.moment = moment        # type: np.ndarray

    # SPECIAL METHODS

    def __repr__(self) -> str:
        """
        Get the formal string representation of the screw transformation.

        :return:    The formal string representation of the screw transformation.
        """
        return "Screw({}, {}, {}, {})".format(self.angle, self.pitch, repr(self.direction), repr(self.moment))

    def __str__(self) -> str:
        """
        Get the informal string representation of the screw transformation.

        :return:    The informal string representation of the screw transformation.
        """
        return "Screw({}, {}, {}, {})".format(self.angle, self.pitch, self.direction, self.moment)
