import math
import numpy as np

from scipy.spatial.transform import Rotation
from typing import List, Union

from .dual_number import DualNumber
from .screw import Screw


class DualQuaternion:
    """
    A dual quaternion, an extension of a normal quaternion that can represent a full rigid-body transform.

    See "Dual Quaternions for Rigid Transformation Blending" by Kavan et al.
    """

    # CONSTRUCTOR

    def __init__(self, w: Union[DualNumber, float] = 0.0, x: Union[DualNumber, float] = 0.0,
                 y: Union[DualNumber, float] = 0.0, z: Union[DualNumber, float] = 0.0):
        """
        Construct a dual quaternion with the specified components.

        .. note:
            The x^, y^, z^ and w^ components of the dual quaternion q^ are such that q^ = w^ + x^.i + y^.j + z^.k.

        :param w:   The w^ component.
        :param x:   The x^ component.
        :param y:   The y^ component.
        :param z:   The z^ component.
        """
        self.w = w if type(w) is DualNumber else DualNumber(w)  # type: DualNumber
        self.x = x if type(x) is DualNumber else DualNumber(x)  # type: DualNumber
        self.y = y if type(y) is DualNumber else DualNumber(y)  # type: DualNumber
        self.z = z if type(z) is DualNumber else DualNumber(z)  # type: DualNumber

        # Put the dual quaternion into canonical form.
        if self.w.r < 0.0:
            self.set_from(-self)

    # SPECIAL METHODS

    def __iadd__(self, rhs: "DualQuaternion") -> "DualQuaternion":
        """
        Add another dual quaternion to this one.

        :param rhs: The other dual quaternion.
        :return:    The result of the operation.
        """
        self.w += rhs.w
        self.x += rhs.x
        self.y += rhs.y
        self.z += rhs.z
        return self

    def __mul__(self, rhs: "DualQuaternion") -> "DualQuaternion":
        """
        Multiply this dual quaternion by another one.

        :param rhs: The other dual quaternion.
        :return:    The result of the operation.
        """
        # Note that it's possible to optimise this if necessary. See:
        #
        # https://github.com/sgolodetz/hesperus2/blob/master/source/engine/core/hesp/math/quaternions/Quaternion.cpp
        return DualQuaternion(
            self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w
        )

    def __neg__(self) -> "DualQuaternion":
        """
        Calculate the negation of the dual quaternion.

        :return:    The negation of the dual quaternion.
        """
        return DualQuaternion(-self.w, -self.x, -self.y, -self.z)

    def __repr__(self) -> str:
        """
        Get the formal string representation of the dual quaternion.

        :return:    The formal string representation of the dual quaternion.
        """
        return "DualQuaternion({}, {}, {}, {})".format(repr(self.w), repr(self.x), repr(self.y), repr(self.z))

    def __rmul__(self, factor: Union[DualNumber, float]) -> "DualQuaternion":
        """
        Scale the dual quaternion by the specified factor.

        :param factor:  The scaling factor.
        :return:        A scaled version of the dual quaternion.
        """
        if type(factor) is float:
            factor = DualNumber(factor)

        return DualQuaternion(factor * self.w, factor * self.x, factor * self.y, factor * self.z)

    def __str__(self) -> str:
        """
        Get the informal string representation of the dual quaternion.

        :return:    The informal string representation of the dual quaternion.
        """
        return "[{}, {}, {}, {}]".format(self.w, self.x, self.y, self.z)

    # PUBLIC STATIC METHODS

    @staticmethod
    def angle_between_rotations(q1: "DualQuaternion", q2: "DualQuaternion") -> float:
        """
        Calculate the angle needed to rotate one dual quaternion representing a pure rotation into another.

        .. note:
            This can function as a distance metric between two pure rotations.

        :param q1:  The first dual quaternion.
        :param q2:  The second dual quaternion.
        :return:    The angle needed to rotate one dual quaternion into the other.
        """
        q1toq2 = (q2 * q1.conjugate()).normalised()  # type: DualQuaternion
        return np.linalg.norm(q1toq2.get_rotation_vector())

    @staticmethod
    def close(lhs: "DualQuaternion", rhs: "DualQuaternion", tolerance: float = 1e-4) -> bool:
        """
        Check whether two dual quaternions are approximately equal, up to a tolerance.

        :param lhs:         The first dual quaternion.
        :param rhs:         The second dual quaternion.
        :param tolerance:   The tolerance value.
        :return:            True, if the two dual quaternions are approximately equal, or False otherwise.
        """
        return \
            DualNumber.close(lhs.w, rhs.w, tolerance) and \
            DualNumber.close(lhs.x, rhs.x, tolerance) and \
            DualNumber.close(lhs.y, rhs.y, tolerance) and \
            DualNumber.close(lhs.z, rhs.z, tolerance)

    @staticmethod
    def from_axis_angle(axis, angle: float) -> "DualQuaternion":
        """
        Construct a dual quaternion that represents a rotation of a particular angle about an axis.

        :param axis:            The rotation axis.
        :param angle:           The rotation angle.
        :return:                The dual quaternion.
        :raises RuntimeError:   If the rotation axis is invalid.
        """
        # Make sure that the axis is a numpy array.
        axis = np.array(axis).astype(float)  # type: np.ndarray

        # If the axis needs to be normalised:
        axis_length_squared = np.dot(axis, axis)  # type: float
        if abs(axis_length_squared - 1) > 1e-9:
            # If it can be normalised:
            if axis_length_squared > 1e-6:
                # Normalise it.
                axis_length = math.sqrt(axis_length_squared)  # type: float
                axis /= axis_length
            # Otherwise, raise an exception.
            else:
                raise RuntimeError("Could not construct dual quaternion - bad rotation axis")

        # Construct the dual quaternion itself.
        cos_half_theta = math.cos(angle / 2)  # type: float
        sin_half_theta = math.sin(angle / 2)  # type: float
        return DualQuaternion(
            cos_half_theta, sin_half_theta * axis[0], sin_half_theta * axis[1], sin_half_theta * axis[2]
        )

    @staticmethod
    def from_point(p) -> "DualQuaternion":
        """
        Construct a dual quaternion that represents a 3D point.

        :param p:   The point.
        :return:    The dual quaternion.
        """
        p = np.array(p).astype(float)  # type: np.ndarray

        return DualQuaternion(
            DualNumber(1, 0),
            DualNumber(0, p[0]),
            DualNumber(0, p[1]),
            DualNumber(0, p[2])
        )

    @staticmethod
    def from_rigid_matrix(m: np.ndarray) -> "DualQuaternion":
        """
        Construct a dual quaternion from a rigid-body matrix.

        :param m:   The rigid-body matrix.
        :return:    The dual quaternion.
        """
        rot = Rotation.from_matrix(m[0:3, 0:3]).as_rotvec()  # type: np.ndarray
        trans = m[0:3, 3]  # type: np.ndarray
        return DualQuaternion.from_translation(trans) * DualQuaternion.from_rotation_vector(rot)

    @staticmethod
    def from_rotation_vector(rot) -> "DualQuaternion":
        """
        Construct a dual quaternion that corresponds to a rotation expressed as a Lie rotation vector.

        :param rot: The Lie rotation vector.
        :return:    The dual quaternion.
        """
        rot = np.array(rot).astype(float)  # type: np.ndarray
        length_squared = np.dot(rot, rot)  # type: float
        if length_squared > 1e-6:
            length = math.sqrt(length_squared)  # type: float
            return DualQuaternion.from_axis_angle(rot / length, length)
        else:
            return DualQuaternion.identity()

    @staticmethod
    def from_screw(screw: Screw) -> "DualQuaternion":
        """
        Construct a dual quaternion that corresponds to a transformation expressed in screw form.

        :param screw:   The screw transformation.
        :return:        The dual quaternion.
        """
        # See "Dual-Quaternions: From Classical Mechanics to Computer Graphics and Beyond" by Ben Kenwright.
        c = math.cos(screw.angle / 2)  # type: float
        s = math.sin(screw.angle / 2)  # type: float
        return DualQuaternion(
            DualNumber(c, -screw.pitch * s / 2),
            DualNumber(screw.direction[0] * s, screw.moment[0] * s + screw.pitch * screw.direction[0] * c / 2),
            DualNumber(screw.direction[1] * s, screw.moment[1] * s + screw.pitch * screw.direction[1] * c / 2),
            DualNumber(screw.direction[2] * s, screw.moment[2] * s + screw.pitch * screw.direction[2] * c / 2)
        )

    @staticmethod
    def from_translation(t) -> "DualQuaternion":
        """
        Construct a dual quaternion that represents a translation by a 3D vector.

        :param t:   The translation vector.
        :return:    The dual quaternion.
        """
        t = np.array(t).astype(float)  # type: np.ndarray

        return DualQuaternion(
            DualNumber(1, 0),
            DualNumber(0, t[0] / 2),
            DualNumber(0, t[1] / 2),
            DualNumber(0, t[2] / 2)
        )

    @staticmethod
    def identity() -> "DualQuaternion":
        """
        Make a dual quaternion that corresponds to the identity matrix.

        :return:    A dual quaternion that corresponds to the identity matrix.
        """
        return DualQuaternion(1, 0, 0, 0)

    @staticmethod
    def linear_blend(dqs: List["DualQuaternion"], weights: List[float]) -> "DualQuaternion":
        """
        Perform a weighted linear blend of the specified set of unit dual quaternions.

        :param dqs:     The input dual quaternions.
        :param weights: The corresponding weights.
        :return:        The result of blending the dual quaternions.
        """
        assert len(dqs) == len(weights)

        result = DualQuaternion()  # type: DualQuaternion
        for i in range(len(dqs)):
            result += weights[i] * dqs[i]

        return result.normalised()

    @staticmethod
    def sclerp(lhs: "DualQuaternion", rhs: "DualQuaternion", t: float):
        """
        Interpolate between two dual quaternions using the ScLERP approach.

        :param lhs: The first dual quaternion.
        :param rhs: The second dual quaternion.
        :param t:   The interpolation parameter (in the range [0,1]).
        :return:    The result of the interpolation.
        """
        return lhs * (lhs.conjugate() * rhs).pow(t)

    # PUBLIC METHODS

    def apply(self, p) -> np.ndarray:
        """
        Apply the transformation represented by this dual quaternion to a 3D point.

        :param p:   The 3D point.
        :return:    The transformed point.
        """
        p = np.array(p).astype(float)  # type: np.ndarray
        result = self.copy()  # type: DualQuaternion
        result *= DualQuaternion.from_point(p)
        result *= self.dual_conjugate()
        return result.__to_point()

    def conjugate(self) -> "DualQuaternion":
        """
        Calculate the conjugate of the dual quaternion.

        :return:    The conjugate of the dual quaternion.
        """
        return DualQuaternion(self.w, -self.x, -self.y, -self.z)

    def copy(self) -> "DualQuaternion":
        """
        Make a copy of the dual quaternion.

        :return:    A copy of the dual quaternion.
        """
        return DualQuaternion(self.w.copy(), self.x.copy(), self.y.copy(), self.z.copy())

    def dual_conjugate(self) -> "DualQuaternion":
        """
        Calculate the "dual conjugate" of the dual quaternion.

        .. note::
            This involves applying both quaternion and dual conjugation.

        :return:    The "dual conjugate" of the dual quaternion.
        """
        return DualQuaternion(self.w.conjugate(), -self.x.conjugate(), -self.y.conjugate(), -self.z.conjugate())

    def get_rotation_part(self) -> "DualQuaternion":
        """
        Get a dual quaternion corresponding to the rotation component of the rigid-body transform represented by
        the dual quaternion.

        :return:    A dual quaternion corresponding to the rotation component of the rigid-body transform represented
                    by the dual quaternion.
        """
        return DualQuaternion(self.w.r, self.x.r, self.y.r, self.z.r)

    def get_rotation_vector(self) -> np.ndarray:
        """
        Gets a Lie vector corresponding to the rotation component of the rigid-body transform represented by the
        dual quaternion.

        :return:    A Lie vector corresponding to the rotation component of the rigid-body transform represented by
                    the dual quaternion.
        """
        # Note: The order here is deliberate, in the sense that scipy expects w last.
        return Rotation.from_quat([self.x.r, self.y.r, self.z.r, self.w.r]).as_rotvec()

    def get_translation(self) -> np.ndarray:
        """
        Get the translation component of the rigid-body transform represented by the dual quaternion.

        :return:    The translation component of the rigid-body transform represented by the dual quaternion.
        """
        tp = self.get_translation_part()  # type: DualQuaternion
        return 2.0 * np.array([tp.x.d, tp.y.d, tp.z.d])

    def get_translation_part(self) -> "DualQuaternion":
        """
        Get a dual quaternion corresponding to the translation component of the rigid-body transform represented by
        the dual quaternion.

        :return:    A dual quaternion corresponding to the translation component of the rigid-body transform
                    represented by the dual quaternion.
        """
        return self * self.get_rotation_part().conjugate()

    def norm(self) -> DualNumber:
        """
        Calculate the norm of the dual quaternion.

        :return:    The norm of the dual quaternion.
        """
        return (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()

    def normalised(self) -> "DualQuaternion":
        """
        Calculate a normalised version of the dual quaternion.

        :return:    A normalised version of the dual quaternion.
        """
        inv_norm = self.norm().inverse()  # type: DualNumber
        return inv_norm * self

    def pow(self, exponent: float) -> "DualQuaternion":
        """
        Calculate an exponent of the dual quaternion.

        :param exponent:    The exponent.
        :return:            The result of raising the dual quaternion to the specified power.
        """
        s = self.to_screw()  # type: Screw
        s.angle *= exponent
        s.pitch *= exponent
        return DualQuaternion.from_screw(s)

    def set_from(self, rhs: "DualQuaternion") -> "DualQuaternion":
        """
        Set this dual quaternion's components to the same as those of another one.

        :param rhs: The other dual quaternion.
        :return:    This dual quaternion (after the operation).
        """
        self.w = rhs.w.copy()
        self.x = rhs.x.copy()
        self.y = rhs.y.copy()
        self.z = rhs.z.copy()
        return self

    def to_rigid_matrix(self) -> np.ndarray:
        """
        Calculate the rigid-body matrix corresponding to the dual quaternion.

        :return:    The rigid-body matrix corresponding to the dual quaternion.
        """
        m = np.eye(4)  # type: np.ndarray
        m[0:3, 0:3] = Rotation.from_quat([self.x.r, self.y.r, self.z.r, self.w.r]).as_matrix()
        m[0:3, 3] = self.get_translation()
        return m

    def to_screw(self) -> Screw:
        """
        Calculate a screw representation of the dual quaternion.

        :return:    A screw representation of the dual quaternion.
        """
        # See "Dual-Quaternions: From Classical Mechanics to Computer Graphics and Beyond" by Ben Kenwright.
        vr = np.array([self.x.r, self.y.r, self.z.r])  # type: np.ndarray
        vd = np.array([self.x.d, self.y.d, self.z.d])  # type: np.ndarray
        inv_vr_len = 1 / np.linalg.norm(vr)            # type: float
        wr = self.w.r                                  # type: float
        wd = self.w.d                                  # type: float

        angle = 2 * math.acos(np.clip(wr, -1.0, 1.0))              # type: float
        pitch = -2 * wd * inv_vr_len                               # type: float
        direction = vr * inv_vr_len                                # type: np.ndarray
        moment = (vd - direction * (pitch * wr / 2)) * inv_vr_len  # type: np.ndarray

        return Screw(angle, pitch, direction, moment)

    # PRIVATE METHODS

    def __to_point(self) -> np.ndarray:
        """
        Convert the dual quaternion to a 3D point (assuming that it represents one in the first place).

        :return:    A 3D point corresponding to the dual quaternion.
        """
        return np.array([self.x.d, self.y.d, self.z.d])
