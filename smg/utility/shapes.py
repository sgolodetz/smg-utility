import numpy as np
import vg

from abc import ABC, abstractmethod


# ENUMERATIONS

class EShapeClassification(int):
    """The result of classifying something against a shape."""

    # SPECIAL METHODS

    def __str__(self):
        """
        Get a string representation of the result of classifying something against a shape.

        :return:    A string representation of the result of classifying something against a shape.
        """
        if self == SC_INSIDE:
            return "SC_INSIDE"
        elif self == SC_OUTSIDE:
            return "SC_OUTSIDE"
        else:
            return "SC_SURFACE"


# {Inside/intersecting with} the shape.
SC_INSIDE = EShapeClassification(0)

# {Outside/not touching} the shape.
SC_OUTSIDE = EShapeClassification(1)

# {On the surface of/touching} the shape.
SC_SURFACE = EShapeClassification(2)


# CLASSES

# noinspection PyUnresolvedReferences
class ShapeVisitor(ABC):
    """An abstract base class for geometric shape visitors."""

    # PUBLIC ABSTRACT METHODSs

    @abstractmethod
    def visit_cylinder(self, cylinder: "Cylinder") -> None:
        """
        Visit a cylinder.

        :param cylinder:    The cylinder.
        """
        pass

    @abstractmethod
    def visit_sphere(self, sphere: "Sphere") -> None:
        """
        Visit a sphere.

        :param sphere:  The sphere.
        """
        pass


class Shape(ABC):
    """An abstract base class for geometric shapes."""

    # PUBLIC ABSTRACT METHODS

    @abstractmethod
    def accept(self, visitor: ShapeVisitor) -> None:
        """
        Accept a visitor.

        :param visitor: The visitor.
        """
        pass

    @abstractmethod
    def classify_point(self, p) -> EShapeClassification:
        """
        Classify a point against the shape.

        :param p:   The point.
        :return:    The result of classifying the point against the shape.
        """
        pass

    # noinspection PyUnresolvedReferences
    @abstractmethod
    def expand(self, radius: float) -> "Shape":
        """
        Make an expanded copy of the shape.

        :param radius:  The radius by which to expand the shape.
        :return:        The expanded copy of the shape.
        """
        pass

    @abstractmethod
    def maxs(self) -> np.ndarray:
        """
        Get the upper bounds of an AABB around the shape.

        .. note::
            The AABB will definitely contain the shape, but may not necessarily be tight.

        :return:    The upper bounds of an AABB around the shape.
        """
        pass

    @abstractmethod
    def mins(self) -> np.ndarray:
        """
        Get the lower bounds of an AABB around the shape.

        .. note::
            The AABB will definitely contain the shape, but may not necessarily be tight.

        :return:    The lower bounds of an AABB around the shape.
        """
        pass

    # PUBLIC METHODS

    def classify_sphere(self, centre, radius: float) -> EShapeClassification:
        """
        Classify a sphere against the shape.

        :param centre:  The centre of the sphere.
        :param radius:  The radius of the sphere.
        :return:        The result of classifying the sphere against the shape.
        """
        return self.expand(radius).classify_point(centre)


class Cylinder(Shape):
    """A (conical) cylinder."""

    # CONSTRUCTOR

    def __init__(self, *, base_centre, base_radius: float, top_centre, top_radius: float):
        """
        Construct a (conical) cylinder.

        :param base_centre: The centre of the cylinder's base.
        :param base_radius: The radius of the cylinder's base.
        :param top_centre:  The centre of the cylinder's top.
        :param top_radius:  The radius of the cylinder's top.
        """
        # : np.ndarray
        self.__base_centre = np.array(base_centre)
        # : float
        self.__base_radius = base_radius
        # : np.ndarray
        self.__top_centre = np.array(top_centre)
        # : float
        self.__top_radius = top_radius

    # PROPERTIES

    @property
    def base_centre(self) -> np.ndarray:
        """
        Get the centre of the cylinder's base.

        :return:    The centre of the cylinder's base.
        """
        return self.__base_centre

    @property
    def base_radius(self) -> float:
        """
        Get the radius of the cylinder's base.

        :return:    The radius of the cylinder's base.
        """
        return self.__base_radius

    @property
    def top_centre(self) -> np.ndarray:
        """
        Get the centre of the cylinder's top.

        :return:    The centre of the cylinder's top.
        """
        return self.__top_centre

    @property
    def top_radius(self) -> float:
        """
        Get the radius of the cylinder's top.

        :return:    The radius of the cylinder's top.
        """
        return self.__top_radius

    # PUBLIC METHODS

    def accept(self, visitor: ShapeVisitor) -> None:
        """
        Accept a visitor.

        :param visitor: The visitor.
        """
        visitor.visit_cylinder(self)

    def classify_point(self, p) -> EShapeClassification:
        """
        Classify a point against the cylinder.

        :param p:   The point.
        :return:    The result of classifying the point against the cylinder.
        """
        # Convert the point to a numpy array (if it isn't one already).
        p = np.array(p)

        # Compute the offset of the point from the centre of the cylinder's base.
        # : np.ndarray
        offset = p - self.__base_centre

        # Compute a vector along the axis of the cylinder.
        # : np.ndarray
        axis = self.__top_centre - self.__base_centre

        # Work out how far along the cylinder's axis the closest point on the axis to the input point lies.
        # : float
        t = vg.scalar_projection(offset, axis) / np.linalg.norm(axis)

        # If the closest point on the axis is not outside the cylinder:
        if 0 <= t <= 1:
            # Compute the distance of the input point from the axis, and the radius of the cylinder at that point.
            # : float
            distance_from_axis = np.linalg.norm(vg.reject(offset, axis))
            # : float
            radius = (1 - t) * self.__base_radius + t * self.__top_radius

            # Check whether the input point is strictly within the cylinder or on its surface.
            if 0 < t < 1 and distance_from_axis < radius:
                return SC_INSIDE
            elif distance_from_axis <= radius:
                return SC_SURFACE

        # If we get here, the input point's outside the cylinder.
        return SC_OUTSIDE

    # noinspection PyUnresolvedReferences
    def expand(self, radius: float) -> "Cylinder":
        """
        Make an expanded copy of the cylinder.

        :param radius:  The radius by which to expand the cylinder.
        :return:        The expanded copy of the cylinder.
        """
        # : np.ndarray
        axis = vg.normalize(self.__top_centre - self.__base_centre)
        return Cylinder(
            base_centre=self.__base_centre - axis * radius,
            base_radius=self.__base_radius + radius,
            top_centre=self.__top_centre + axis * radius,
            top_radius=self.__top_radius + radius
        )

    def maxs(self) -> np.ndarray:
        """
        Get the upper bounds of an AABB around the cylinder.

        .. note::
            These upper bounds are not especially tight, but they'll do for now.

        :return:    The upper bounds of an AABB around the cylinder.
        """
        return np.max(
            [
                self.__base_centre + np.full(3, self.__base_radius),
                self.__top_centre + np.full(3, self.__top_radius)
            ],
            axis=0
        )

    def mins(self) -> np.ndarray:
        """
        Get the lower bounds of an AABB around the cylinder.

        .. note::
            These lower bounds are not especially tight, but they'll do for now.

        :return:    The lower bounds of an AABB around the cylinder.
        """
        return np.min(
            [
                self.__base_centre - np.full(3, self.__base_radius),
                self.__top_centre - np.full(3, self.__top_radius)
            ],
            axis=0
        )


class Sphere(Shape):
    """A sphere."""

    # CONSTRUCTOR

    def __init__(self, *, centre, radius: float):
        """
        Construct a sphere.

        :param centre:  The centre of the sphere.
        :param radius:  The radius of the sphere.
        """
        # : np.ndarray
        self.__centre = np.array(centre)
        # : float
        self.__radius = radius

    # PROPERTIES

    @property
    def centre(self) -> np.ndarray:
        """
        Get the centre of the sphere.

        :return:    The centre of the sphere.
        """
        return self.__centre

    @property
    def radius(self) -> float:
        """
        Get the radius of the sphere.

        :return:    The radius of the sphere.
        """
        return self.__radius

    # PUBLIC METHODS

    def accept(self, visitor: ShapeVisitor) -> None:
        """
        Accept a visitor.

        :param visitor: The visitor.
        """
        visitor.visit_sphere(self)

    def classify_point(self, p) -> EShapeClassification:
        """
        Classify a point against the sphere.

        :param p:   The point.
        :return:    The result of classifying the point against the sphere.
        """
        p = np.array(p)

        # : float
        distance = np.linalg.norm(p - self.__centre)

        if distance < self.__radius:
            return SC_INSIDE
        elif distance > self.__radius:
            return SC_OUTSIDE
        else:
            return SC_SURFACE

    # noinspection PyUnresolvedReferences
    def expand(self, radius: float) -> "Sphere":
        """
        Make an expanded copy of the sphere.

        :param radius:  The radius by which to expand the sphere.
        :return:        The expanded copy of the sphere.
        """
        return Sphere(centre=self.__centre, radius=self.__radius + radius)

    def maxs(self) -> np.ndarray:
        """
        Get the upper bounds of an AABB around the sphere.

        :return:    The upper bounds of an AABB around the sphere.
        """
        return self.__centre + np.full(3, self.__radius)

    def mins(self) -> np.ndarray:
        """
        Get the lower bounds of an AABB around the sphere.

        :return:    The lower bounds of an AABB around the sphere.
        """
        return self.__centre - np.full(3, self.__radius)
