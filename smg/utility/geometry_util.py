import math
import numpy as np

from itertools import product
from numba import cuda
from typing import Dict, List, Tuple

from .image_util import ImageUtil
from .numba_util import NumbaUtil


class GeometryUtil:
    """Utility functions related to geometry."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def compute_world_points_image(depth_image: np.ndarray, depth_mask: np.ndarray, pose: np.ndarray,
                                   fx: float, fy: float, cx: float, cy: float, ws_points: np.ndarray) -> None:
        """
        Compute a world-space points image from a depth image, camera pose, and set of camera intrinsics.

        .. note::
            The origin, i.e. (0, 0, 0), is stored for pixels whose depth is zero.

        :param depth_image:     The depth image.
        :param depth_mask:      A mask indicating which pixels have valid depths (non-zero means valid).
        :param pose:            The camera pose (denoting a transformation from camera space to world space).
        :param fx:              The horizontal focal length.
        :param fy:              The vertical focal length.
        :param cx:              The x component of the principal point.
        :param cy:              The y component of the principal point.
        :param ws_points:       The output world-space points image.
        """
        NumbaUtil.launch_kernel_2d(
            GeometryUtil.__ck_compute_world_points_image, depth_image, depth_mask, pose,
            fx, fy, cx, cy, ws_points, grid_size=depth_image.shape
        )

    @staticmethod
    def compute_world_points_image_fast(depth_image: np.ndarray, pose: np.ndarray,
                                        intrinsics: Tuple[float, float, float, float],
                                        ws_points: np.ndarray) -> None:
        """
        Compute a world-space points image from a depth image, camera pose, and set of camera intrinsics.

        .. note::
            The origin, i.e. (0, 0, 0), is stored for pixels whose depth is zero.

        :param depth_image:     The depth image (pixels with zero depth are treated as invalid).
        :param pose:            The camera pose (denoting a transformation from camera space to world space).
        :param intrinsics:      The camera intrinsics, as an (fx, fy, cx, cy) tuple.
        :param ws_points:       The output world-space points image.
        """
        # TODO: This should ultimately replace the compute_world_points_image function above.
        height, width = depth_image.shape
        fx, fy, cx, cy = intrinsics
        xl = np.array(range(width))
        yl = np.array(range(height))
        al = np.tile((xl - cx) / fx, height).reshape(height, width) * depth_image
        bl = np.transpose(np.tile((yl - cy) / fy, width).reshape(width, height)) * depth_image
        for i in range(3):
            ws_points[:, :, i] = pose[i, 0] * al + pose[i, 1] * bl + pose[i, 2] * depth_image

    @staticmethod
    def make_depths_orthogonal(depth_image: np.ndarray, intrinsics: Tuple[float, float, float, float]) -> None:
        """
        Convert the depth values in a depth image from Euclidean distances from the camera centre
        to orthogonal distances to the image plane.

        :param depth_image: The depth image.
        :param intrinsics:  The depth camera intrinsics.
        """
        NumbaUtil.launch_kernel_2d(
            GeometryUtil.__ck_make_depths_orthogonal, depth_image, *intrinsics, grid_size=depth_image.shape
        )

    @staticmethod
    def make_point_cloud(colour_image: np.ndarray, depth_image: np.ndarray, depth_mask: np.ndarray,
                         intrinsics: Tuple[float, float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make a colourised point cloud for visualisation purposes.

        .. note::
            For efficiency (bearing in mind that we're only trying to visualise the point cloud, not use it for
            downstream processing), we avoid pruning any points. Instead, we set the position and colour of any
            invalid points to the camera origin and white, respectively. This won't affect anything because there
            aren't any other points closer than the near plane of the camera.

        :param depth_image:   The depth image whose pixels are to be back-projected to make the point cloud.
        :param colour_image:  The colour image to use to colourise the point cloud (assumed to be in BGR format).
        :param depth_mask:    A mask indicating which pixels have valid depths (non-zero means valid).
        :param intrinsics:    The camera intrinsics (assumed to be the same for both the colour/depth cameras).
        """
        # Create a camera-space points image for the frame by using the identity pose.
        height, width = depth_image.shape
        cs_points: np.ndarray = np.zeros((height, width, 3), dtype=np.float32)
        # noinspection PyTypeChecker
        GeometryUtil.compute_world_points_image(depth_image, depth_mask, np.eye(4), *intrinsics, cs_points)

        # Allocate the output point cloud arrays.
        pcd_points: np.ndarray = np.zeros((height * width, 3))
        pcd_colours: np.ndarray = np.ones((height * width, 3))

        # Launch the kernel.
        NumbaUtil.launch_kernel_2d(
            GeometryUtil.__ck_make_point_cloud, cs_points, ImageUtil.flip_channels(colour_image), depth_mask,
            pcd_points, pcd_colours, grid_size=(height, width)
        )

        return pcd_points, pcd_colours

    @staticmethod
    def make_voxel_grid_endpoints(mins: List[float], maxs: List[float], voxel_size: List[float]) -> \
            Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
        """
        Make the endpoints of the lines needed for a wireframe voxel grid.

        :param mins:        The minimum bounds of the voxel grid.
        :param maxs:        The maximum bounds of the voxel grid.
        :param voxel_size:  The voxel size.
        :return:            The endpoints of the lines needed for the voxel grid.
        """
        vals: Dict[int, np.ndarray] = {}
        for i in range(3):
            maxs[i] = mins[i] + math.ceil((maxs[i] - mins[i]) / voxel_size[i]) * voxel_size[i]
            vals[i] = np.linspace(mins[i], maxs[i], int((maxs[i] - mins[i]) / voxel_size[i]) + 1)

        xpts1: List[Tuple[float, float, float]] = [(mins[0], y, z) for y, z in product(vals[1], vals[2])]
        xpts2: List[Tuple[float, float, float]] = [(maxs[0], y, z) for y, z in product(vals[1], vals[2])]

        ypts1: List[Tuple[float, float, float]] = [(x, mins[1], z) for x, z in product(vals[0], vals[2])]
        ypts2: List[Tuple[float, float, float]] = [(x, maxs[1], z) for x, z in product(vals[0], vals[2])]

        zpts1: List[Tuple[float, float, float]] = [(x, y, mins[2]) for x, y in product(vals[0], vals[1])]
        zpts2: List[Tuple[float, float, float]] = [(x, y, maxs[2]) for x, y in product(vals[0], vals[1])]

        pts1: List[Tuple[float, float, float]] = xpts1 + ypts1 + zpts1
        pts2: List[Tuple[float, float, float]] = xpts2 + ypts2 + zpts2

        return pts1, pts2

    # PRIVATE STATIC CUDA KERNELS

    @staticmethod
    @cuda.jit
    def __ck_compute_world_points_image(depth_image, depth_mask, pose, fx: float, fy: float, cx: float, cy: float,
                                        ws_points):
        """
        Compute a world-space points image from a depth image, known camera pose and known intrinsics.

        .. note::
            This CUDA kernel must be invoked using numba.

        .. note::
            The origin, i.e. (0, 0, 0), is stored for pixels whose depth is zero.

        :param depth_image:   The depth image.
        :param depth_mask:    A mask indicating which pixels have valid depths (non-zero means valid).
        :param pose:          The camera pose (denoting a transformation from camera space to world space).
        :param fx:            The horizontal focal length.
        :param fy:            The vertical focal length.
        :param cx:            The x component of the principal point.
        :param cy:            The y component of the principal point.
        :param ws_points:     The output world-space points image.
        """
        # noinspection PyArgumentList
        y, x = cuda.grid(2)
        if y < depth_image.shape[0] and x < depth_image.shape[1]:
            # If the pixel has an invalid depth, store the origin as its world-space point.
            if depth_mask[y, x] == 0:
                ws_points[y, x] = 0, 0, 0
                return

            # Back-project the pixel using the depth.
            depth = depth_image[y, x]
            a = (x - cx) * depth / fx
            b = (y - cy) * depth / fy
            c = depth

            # Transform the back-projected point into world space.
            for i in range(3):
                ws_points[y, x, i] = pose[i, 0] * a + pose[i, 1] * b + pose[i, 2] * c + pose[i, 3]

    @staticmethod
    @cuda.jit
    def __ck_make_depths_orthogonal(depth_image, fx: float, fy: float, cx: float, cy: float):
        """
        Convert the depth values in a depth image from Euclidean distances from the camera centre
        to orthogonal distances to the image plane.

        .. note::
            This CUDA kernel must be invoked using numba.

        :param depth_image: The depth image.
        :param fx:          The horizontal focal length.
        :param fy:          The vertical focal length.
        :param cx:          The x component of the principal point.
        :param cy:          The y component of the principal point.
        """
        # noinspection PyArgumentList
        y, x = cuda.grid(2)
        if y < depth_image.shape[0] and x < depth_image.shape[1]:
            # Compute the position (a,b,1)^T of the pixel on the image plane.
            a: float = (x - cx) / fx
            b: float = (y - cy) / fy

            # Compute the distance from the camera centre to the pixel, and then divide by it.
            dist_to_pixel = math.sqrt(a ** 2 + b ** 2 + 1 ** 2)
            depth_image[y, x] /= dist_to_pixel

    @staticmethod
    @cuda.jit
    def __ck_make_point_cloud(cs_points, colour_image, depth_mask, pcd_points, pcd_colours):
        """
        Make a colourised point cloud for visualisation purposes.

        .. note::
                This CUDA kernel must be invoked using numba.

        :param cs_points:           The camera-space points.
        :param colour_image:        The image to use to colour the point cloud.
        :param depth_mask:          A mask indicating which pixels have valid depths (non-zero means valid).
        :param pcd_points:          The output points array.
        :param pcd_colours:         The output colours array.
        """
        # noinspection PyArgumentList
        y, x = cuda.grid(2)
        if y >= cs_points.shape[0] or x >= cs_points.shape[1]:
            return

        # If the pixel's depth was invalid, so is its camera-space point, so early out.
        if depth_mask[y, x] == 0:
            return

        # Get the camera-space point for the pixel.
        p = cs_points[y, x]

        # If the point is far away, early out (far-flung points break the Open3D visualiser).
        for i in range(3):
            if p[i] > 100 or p[i] < -100:
                return

        # Determine the pixel's offset in the output point and colour arrays.
        k = y * cs_points.shape[1] + x

        # Add the point and its colour to the output arrays.
        pcd_points[k] = p[0], p[1], p[2]
        pcd_colours[k] = colour_image[y, x, 0] / 255, colour_image[y, x, 1] / 255, colour_image[y, x, 2] / 255
