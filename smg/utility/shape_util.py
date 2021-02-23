import numpy as np

from typing import Dict, List

from .shapes import SC_OUTSIDE, Shape


class ShapeUtil:
    """TODO"""

    # PUBLIC STATIC METHODS

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
                        if shape.classify_sphere(p, np.sqrt(3) * voxel_size / 2) != SC_OUTSIDE:
                            output.append(p)

        return output
