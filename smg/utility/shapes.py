# -*- coding: future_annotations -*-

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
SC_INSIDE: EShapeClassification = EShapeClassification(0)

# {Outside/not touching} the shape.
SC_OUTSIDE: EShapeClassification = EShapeClassification(1)

# {On the surface of/touching} the shape.
SC_SURFACE: EShapeClassification = EShapeClassification(2)


# CLASSES

# noinspection PyUnresolvedReferences
class ShapeVisitor(ABC):
    """An abstract base class for geometric shape visitors."""

    # PUBLIC ABSTRACT METHODSs

    @abstractmethod
    def visit_cylinder(self, cylinder: Cylinder) -> None:
        """
        Visit a cylinder.

        :param cylinder:    The cylinder.
        """
        pass

    @abstractmethod
    def visit_sphere(self, sphere: Sphere) -> None:
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
    def expand(self, radius: float) -> Shape:
        """
        Make an expanded copy of the shape.

        :param radius:  The radius by which to expand the shape.
        :return:        The expanded copy of the shape.
        """
        pass

    @abstractmethod
    def maxs(self) -> np.ndarray:
        pass

    @abstractmethod
    def mins(self) -> np.ndarray:
        pass

    # PUBLIC METHODS

    def classify_sphere(self, centre, radius: float) -> EShapeClassification:
        return self.expand(radius).classify_point(centre)


class Cylinder(Shape):
    """A (conical) cylinder."""

    # CONSTRUCTOR

    def __init__(self, *, base_centre, base_radius: float, top_centre, top_radius: float):
        self.__base_centre: np.ndarray = np.array(base_centre)
        self.__base_radius: float = base_radius
        self.__top_centre: np.ndarray = np.array(top_centre)
        self.__top_radius: float = top_radius

    # PROPERTIES

    @property
    def base_centre(self) -> np.ndarray:
        return self.__base_centre

    @property
    def base_radius(self) -> float:
        return self.__base_radius

    @property
    def top_centre(self) -> np.ndarray:
        return self.__top_centre

    @property
    def top_radius(self) -> float:
        return self.__top_radius

    # PUBLIC METHODS

    def accept(self, visitor: ShapeVisitor) -> None:
        """
        Accept a visitor.

        :param visitor: The visitor.
        """
        visitor.visit_cylinder(self)

    def classify_point(self, p) -> EShapeClassification:
        p = np.array(p)
        q: np.ndarray = p - self.__base_centre

        axis: np.ndarray = self.__top_centre - self.__base_centre
        t: float = vg.scalar_projection(q, axis) / np.linalg.norm(axis)
        radius: float = (1 - t) * self.__base_radius + t * self.__top_radius
        distance_from_axis: float = np.linalg.norm(vg.reject(q, axis))

        if 0 < t < 1:
            if distance_from_axis < radius:
                return SC_INSIDE
            elif distance_from_axis > radius:
                return SC_OUTSIDE
            else:
                return SC_SURFACE
        elif t == 0 or t == 1:
            return SC_SURFACE
        else:
            return SC_OUTSIDE

    # noinspection PyUnresolvedReferences
    def expand(self, radius: float) -> Cylinder:
        """
        Make an expanded copy of the cylinder.

        :param radius:  The radius by which to expand the cylinder.
        :return:        The expanded copy of the cylinder.
        """
        axis: np.ndarray = vg.normalize(self.__top_centre - self.__base_centre)
        return Cylinder(
            base_centre=self.__base_centre - axis * radius,
            base_radius=self.__base_radius + radius,
            top_centre=self.__top_centre + axis * radius,
            top_radius=self.__top_radius + radius
        )

    def maxs(self) -> np.ndarray:
        # Note: These upper bounds are not especially tight, but they'll do for now.
        return np.max(
            [
                self.__base_centre + np.full(3, self.__base_radius),
                self.__top_centre + np.full(3, self.__top_radius)
            ],
            axis=0
        )

    def mins(self) -> np.ndarray:
        # Note: These lower bounds are not especially tight, but they'll do for now.
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
        self.__centre: np.ndarray = np.array(centre)
        self.__radius: float = radius

    # PROPERTIES

    @property
    def centre(self) -> np.ndarray:
        return self.__centre

    @property
    def radius(self) -> float:
        return self.__radius

    # PUBLIC METHODS

    def accept(self, visitor: ShapeVisitor) -> None:
        """
        Accept a visitor.

        :param visitor: The visitor.
        """
        visitor.visit_sphere(self)

    def classify_point(self, p) -> EShapeClassification:
        p = np.array(p)

        distance: float = np.linalg.norm(p - self.__centre)

        if distance < self.__radius:
            return SC_INSIDE
        elif distance > self.__radius:
            return SC_OUTSIDE
        else:
            return SC_SURFACE

    # noinspection PyUnresolvedReferences
    def expand(self, radius: float) -> Sphere:
        return Sphere(centre=self.__centre, radius=self.__radius + radius)

    def maxs(self) -> np.ndarray:
        return self.__centre + np.full(3, self.__radius)

    def mins(self) -> np.ndarray:
        return self.__centre - np.full(3, self.__radius)
