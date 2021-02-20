# -*- coding: future_annotations -*-

import numpy as np
import vg

from abc import ABC, abstractmethod


class EPointClassification(int):
    def __str__(self):
        if self == PC_INSIDE:
            return "PC_INSIDE"
        elif self == PC_OUTSIDE:
            return "PC_OUTSIDE"
        else:
            return "PC_SURFACE"


PC_INSIDE: EPointClassification = EPointClassification(0)
PC_OUTSIDE: EPointClassification = EPointClassification(1)
PC_SURFACE: EPointClassification = EPointClassification(2)


# noinspection PyUnresolvedReferences
class ShapeVisitor:
    @abstractmethod
    def visit_cylinder(self, cylinder: Cylinder) -> None: ...

    @abstractmethod
    def visit_sphere(self, sphere: Sphere) -> None: ...


class Shape(ABC):
    @abstractmethod
    def accept(self, visitor: ShapeVisitor) -> None: ...

    @abstractmethod
    def classify_point(self, p: np.ndarray) -> EPointClassification:
        pass


class Cylinder(Shape):
    def __init__(self, *, base_centre, base_radius: float, top_centre, top_radius: float):
        self.__base_centre: np.ndarray = np.array(base_centre)
        self.__base_radius: float = base_radius
        self.__top_centre: np.ndarray = np.array(top_centre)
        self.__top_radius: float = top_radius

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

    def accept(self, visitor: ShapeVisitor) -> None:
        visitor.visit_cylinder(self)

    def classify_point(self, pt) -> EPointClassification:
        p = np.array(pt)

        axis: np.ndarray = self.__top_centre - self.__base_centre
        t: float = vg.scalar_projection(p, axis) / np.linalg.norm(axis)
        radius: float = (1 - t) * self.__base_radius + t * self.__top_radius
        distance_from_axis: float = np.linalg.norm(vg.reject(p, axis))

        if 0 < t < 1:
            if distance_from_axis < radius:
                return PC_INSIDE
            elif distance_from_axis > radius:
                return PC_OUTSIDE
            else:
                return PC_SURFACE
        elif t == 0 or t == 1:
            return PC_SURFACE
        else:
            return PC_OUTSIDE


class Sphere(Shape):
    def __init__(self, *, centre, radius: float):
        self.__centre: np.ndarray = np.array(centre)
        self.__radius: float = radius

    @property
    def centre(self) -> np.ndarray:
        return self.__centre

    @property
    def radius(self) -> float:
        return self.__radius

    def accept(self, visitor: ShapeVisitor) -> None:
        visitor.visit_sphere(self)

    def classify_point(self, pt) -> EPointClassification:
        p = np.array(pt)

        distance: float = np.linalg.norm(p - self.__centre)

        if distance < self.__radius:
            return PC_INSIDE
        elif distance > self.__radius:
            return PC_OUTSIDE
        else:
            return PC_SURFACE
