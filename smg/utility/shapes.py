# -*- coding: future_annotations -*-

import numpy as np
import vg

from abc import ABC, abstractmethod
from typing import Dict, List


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
    def visit_cylinder(self, cylinder: Cylinder) -> None:
        pass

    @abstractmethod
    def visit_sphere(self, sphere: Sphere) -> None:
        pass


class Shape(ABC):
    @abstractmethod
    def accept(self, visitor: ShapeVisitor) -> None:
        pass

    @abstractmethod
    def classify_point(self, p) -> EPointClassification:
        pass

    @abstractmethod
    def classify_sphere(self, centre, radius: float) -> EPointClassification:
        pass

    @abstractmethod
    def maxs(self) -> np.ndarray:
        pass

    @abstractmethod
    def mins(self) -> np.ndarray:
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

    def classify_point(self, p) -> EPointClassification:
        p = np.array(p)
        q: np.ndarray = p - self.__base_centre

        axis: np.ndarray = self.__top_centre - self.__base_centre
        t: float = vg.scalar_projection(q, axis) / np.linalg.norm(axis)
        radius: float = (1 - t) * self.__base_radius + t * self.__top_radius
        distance_from_axis: float = np.linalg.norm(vg.reject(q, axis))

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

    def classify_sphere(self, centre, radius: float) -> EPointClassification:
        axis: np.ndarray = vg.normalize(self.__top_centre - self.__base_centre)
        expanded_cylinder: Cylinder = Cylinder(
            base_centre=self.__base_centre - axis * radius,
            base_radius=self.__base_radius + radius,
            top_centre=self.__top_centre + axis * radius,
            top_radius=self.__top_radius + radius
        )
        return expanded_cylinder.classify_point(centre)

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

    def classify_point(self, p) -> EPointClassification:
        p = np.array(p)

        distance: float = np.linalg.norm(p - self.__centre)

        if distance < self.__radius:
            return PC_INSIDE
        elif distance > self.__radius:
            return PC_OUTSIDE
        else:
            return PC_SURFACE

    def classify_sphere(self, centre, radius: float) -> EPointClassification:
        expanded_sphere: Sphere = Sphere(centre=self.__centre, radius=self.__radius + radius)
        return expanded_sphere.classify_point(centre)

    def maxs(self) -> np.ndarray:
        return self.__centre + np.full(3, self.__radius)

    def mins(self) -> np.ndarray:
        return self.__centre - np.full(3, self.__radius)


class ShapeUtil:
    # @staticmethod
    # def rasterise_shapes(shapes: List[Shape], voxel_size: float) -> List[np.ndarray]:
    #     mins: np.ndarray = np.full(3, np.inf)
    #     maxs: np.ndarray = np.full(3, -np.inf)
    #     for shape in shapes:
    #         mins = np.min([mins, shape.mins()], axis=0)
    #         maxs = np.max([maxs, shape.maxs()], axis=0)
    #
    #     mins = np.floor(mins / voxel_size) * voxel_size
    #     maxs = np.ceil(maxs / voxel_size) * voxel_size
    #
    #     vals: Dict[int, np.ndarray] = {}
    #     for i in range(3):
    #         vals[i] = np.linspace(
    #             mins[i] + voxel_size / 2, maxs[i] - voxel_size / 2, int((maxs[i] - mins[i]) / voxel_size)
    #         )
    #
    #     output: List[np.ndarray] = []
    #
    #     for x in vals[0]:
    #         for y in vals[1]:
    #             for z in vals[2]:
    #                 p: np.ndarray = np.array([x, y, z])
    #                 for shape in shapes:
    #                     if shape.classify_sphere(p, voxel_size / 2) != PC_OUTSIDE:
    #                         output.append(p)
    #                         break
    #
    #     return output

    @staticmethod
    def rasterise_shapes(shapes: List[Shape], voxel_size: float) -> List[np.ndarray]:
        output: List[np.ndarray] = []

        for shape in shapes:
            mins: np.ndarray = np.floor(shape.mins() / voxel_size) * voxel_size
            maxs: np.ndarray = np.ceil(shape.maxs() / voxel_size) * voxel_size

            vals: Dict[int, np.ndarray] = {}
            for i in range(3):
                vals[i] = np.linspace(
                    mins[i] + voxel_size / 2, maxs[i] - voxel_size / 2, int(np.round((maxs[i] - mins[i]) / voxel_size))
                )

            for z in vals[2]:
                for y in vals[1]:
                    for x in vals[0]:
                        p: np.ndarray = np.array([x, y, z])
                        if shape.classify_sphere(p, np.sqrt(3) * voxel_size / 2) != PC_OUTSIDE:
                            output.append(p)

        return output
