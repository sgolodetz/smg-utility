import numpy as np

from typing import Dict, List

from .shapes import SC_OUTSIDE, Shape


class ShapeUtil:
    """Utility functions related to geometric shapes."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def rasterise_shapes(shapes: List[Shape], voxel_size: float) -> List[np.ndarray]:
        """
        Make a list of voxels that potentially intersect/touch at least one of the specified shapes.

        .. note::
            The "potentially" refers to the fact that we test the bounding spheres of the voxels (rather than
            the voxels themselves) against the shapes, for speed reasons.
        .. note::
            As currently implemented, the output list may contain duplicates.

        :param shapes:      A list of shapes.
        :param voxel_size:  The voxel size.
        :return:            A list of voxels that potentially intersect/touch at least one of the specified shapes.
        """
        output = []  # type: List[np.ndarray]

        # For each shape:
        for shape in shapes:
            # Round the shape bounds to the nearest voxel coordinates (away from zero).
            mins = np.floor(shape.mins() / voxel_size) * voxel_size  # type: np.ndarray
            maxs = np.ceil(shape.maxs() / voxel_size) * voxel_size   # type: np.ndarray

            # Work out the x, y and z values for the centres of the voxels that will need to be tested.
            vals = {}  # type: Dict[int, np.ndarray]
            for i in range(3):
                vals[i] = np.linspace(
                    mins[i] + voxel_size / 2, maxs[i] - voxel_size / 2, int(np.round((maxs[i] - mins[i]) / voxel_size))
                )

            # For each voxel that needs to be tested:
            for z in vals[2]:
                for y in vals[1]:
                    for x in vals[0]:
                        # Test its bounding sphere against the shape. If it intersects/touches the shape,
                        # add the voxel to the output list.
                        voxel_centre = np.array([x, y, z])  # type: np.ndarray
                        if shape.classify_sphere(voxel_centre, np.sqrt(3) * voxel_size / 2) != SC_OUTSIDE:
                            output.append(voxel_centre)

        return output
