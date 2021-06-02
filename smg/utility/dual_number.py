import math

from typing import Union


class DualNumber:
    """
    A dual number of the form ^n^ = r + eps * d, where eps^2 = 0.

    See "Dual Quaternions for Rigid Transformation Blending" by Kavan et al.
    """

    # CONSTRUCTOR

    def __init__(self, r: Union[float, int] = 0.0, d: Union[float, int] = 0.0):
        """
        Construct a dual number with the specified components.

        :param r:   The real component of the dual number.
        :param d:   The dual component of the dual number.
        """
        self.r = float(r)  # type: float
        self.d = float(d)  # type: float

    # SPECIAL METHODS

    def __add__(self, rhs: "DualNumber") -> "DualNumber":
        """
        Add another dual number to this one.

        :param rhs: The other dual number.
        :return:    The result of the operation.
        """
        if type(rhs) is not DualNumber:
            return NotImplemented

        copy = self.copy()  # type: DualNumber
        copy += rhs
        return copy

    def __iadd__(self, rhs: "DualNumber") -> "DualNumber":
        """
        Add another dual number to this one (in-place).

        :param rhs: The other dual number.
        :return:    This dual number (after the operation).
        """
        self.r += rhs.r
        self.d += rhs.d
        return self

    def __imul__(self, rhs: "DualNumber") -> "DualNumber":
        """
        Multiply this dual number by another one (in-place).

        :param rhs: The other dual number.
        :return:    This dual number (after the operation).
        """
        self.d = self.r * rhs.d + self.d * rhs.r
        self.r *= rhs.r
        return self

    def __isub__(self, rhs: "DualNumber") -> "DualNumber":
        """
        Subtract another dual number from this one (in-place).

        :param rhs: The other dual number.
        :return:    This dual number (after the operation).
        """
        self.r -= rhs.r
        self.d -= rhs.d
        return self

    def __mul__(self, rhs: "DualNumber") -> "DualNumber":
        """
        Multiply this dual number by another one.

        :param rhs: The other dual number.
        :return:    The result of the operation.
        """
        if type(rhs) is not DualNumber:
            return NotImplemented

        copy = self.copy()  # type: DualNumber
        copy *= rhs
        return copy

    def __neg__(self) -> "DualNumber":
        """
        Calculate the negation of the dual number.

        :return:    The negation of the dual number.
        """
        return DualNumber(-self.r, -self.d)

    def __repr__(self) -> str:
        """
        Get the formal string representation of the dual number.

        :return:    The formal string representation of the dual number.
        """
        return "DualNumber({}, {})".format(self.r, self.d)

    def __str__(self) -> str:
        """
        Get the informal string representation of the dual number.

        :return:    The informal string representation of the dual number.
        """
        return "({}, {})".format(self.r, self.d)

    def __sub__(self, rhs: "DualNumber") -> "DualNumber":
        """
        Subtract another dual number from this one.

        :param rhs: The other dual number.
        :return:    The result of the operation.
        """
        if type(rhs) is not DualNumber:
            return NotImplemented

        copy = self.copy()    # type: DualNumber
        copy -= rhs
        return copy

    # PUBLIC STATIC METHODS

    @staticmethod
    def close(lhs: "DualNumber", rhs: "DualNumber", tolerance: float = 1e-4) -> bool:
        """
        Check whether two dual numbers are approximately equal, up to a tolerance.

        :param lhs:         The first dual number.
        :param rhs:         The second dual number.
        :param tolerance:   The tolerance value.
        :return:            True, if the two dual numbers are approximately equal, or False otherwise.
        """
        return abs(lhs.r - rhs.r) <= tolerance and abs(lhs.d - rhs.d) <= tolerance

    # PUBLIC METHODS

    def conjugate(self) -> "DualNumber":
        """
        Calculate the conjugate of the dual number.

        :return:    The conjugate of the dual number.
        """
        return DualNumber(self.r, -self.d)

    def copy(self) -> "DualNumber":
        """
        Make a copy of the dual number.

        :return:    A copy of the dual number.
        """
        return DualNumber(self.r, self.d)

    def inverse(self) -> "DualNumber":
        """
        Calculate the inverse of the dual number.

        :return:    The inverse of the dual number.
        """
        assert not self.is_pure()
        return DualNumber(1 / self.r, -self.d / self.r ** 2)

    def is_pure(self) -> bool:
        """
        Determine whether or not the dual number is pure (has a zero real component).

        :return:    True, if the dual number is pure, or False otherwise.
        """
        return self.r == 0

    def sqrt(self) -> "DualNumber":
        """
        Calculate the square root of the dual number.

        :return:    The square root of the dual number.
        """
        assert self.r >= 0
        root_r = math.sqrt(self.r)  # type: float
        return DualNumber(root_r, self.d / (2 * root_r))
