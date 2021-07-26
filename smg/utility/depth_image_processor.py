import cv2
import math
import numpy as np

from numba import cuda
from typing import Dict, Tuple

from .numba_util import NumbaUtil


class DepthImageProcessor:
    """Utility functions for post-processing depth images."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def remove_small_regions(depth_image: np.ndarray, segmentation: np.ndarray, stats: np.ndarray,
                             *, min_region_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove small regions from a depth image (as identified by a previous segmentation of the depth image).

        :param depth_image:     The depth image.
        :param segmentation:    A previous segmentation of the depth image.
        :param stats:           The region statistics associated with the segmentation.
        :param min_region_size: The minimum size of region to keep (that is, regions smaller than this will have
                                their depths set to zero).
        :return:                A copy of the depth image from which the small regions have been removed.
        """
        replace = {}  # type: Dict[int, int]
        for i in range(stats.shape[0]):
            area = stats[i, cv2.CC_STAT_AREA]  # type: np.intc
            if area < min_region_size:
                replace[i] = 0

        indexer = np.array([replace.get(i, i) for i in range(0, np.max(segmentation) + 1)])  # type: np.ndarray

        segmentation = indexer[segmentation]
        depth_image = np.where(segmentation != 0, depth_image, 0.0)

        return depth_image, segmentation

    @staticmethod
    def segment_depth_image(depth_image: np.ndarray, *, max_depth_difference: float) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segment a depth image into regions such that all of the pixels in each region have similar depth.

        .. note::
            We use a 5x5 neighbourhood rather than a 3x3 one, as it leads to more reliable depth edges.

        :param depth_image:             The depth image to segment.
        :param max_depth_difference:    The maximum depth difference to allow between two neighbouring pixels in the
                                        same region.
        :return:                        A tuple consisting of the segmentation image, the associated statistics of its
                                        regions, and the depth edge map that was used to construct the segmentation.
        """
        # Make the depth edge map.
        depth_edges = np.full_like(depth_image, 255, dtype=np.uint8)  # type: np.ndarray
        DepthImageProcessor.__make_depth_edge_map(depth_image, depth_edges, max_depth_difference)

        # Find the connected components of the depth edge map.
        _, segmentation, stats, _ = cv2.connectedComponentsWithStats(depth_edges)

        return segmentation, stats, depth_edges

    # PRIVATE STATIC METHODS

    @staticmethod
    def __make_depth_edge_map(depth_image, depth_edges, max_depth_difference: float) -> None:
        """
        Make a depth edge map in which a pixel is marked as a depth edge if the maximum absolute difference between
        its own depth and that of one of its neighbours exceeds the specified threshold.

        :param depth_image:             The input depth image.
        :param depth_edges:             The output depth edge map.
        :param max_depth_difference:    The maximum depth difference to allow between two neighbouring pixels in the
                                        same region.
        """
        NumbaUtil.launch_kernel_2d(
            DepthImageProcessor.__ck_make_depth_edge_map, depth_image, depth_edges, max_depth_difference,
            grid_size=depth_image.shape
        )

    # PRIVATE STATIC CUDA KERNELS

    @staticmethod
    @cuda.jit
    def __ck_make_depth_edge_map(depth_image, depth_edges, max_depth_difference: float):
        """
        Make a depth edge map in which a pixel is marked as a depth edge if the maximum absolute difference between
        its own depth and that of one of its neighbours exceeds the specified threshold.

        .. note::
            This CUDA kernel must be invoked using numba.

        :param depth_image:             The input depth image.
        :param depth_edges:             The output depth edge map.
        :param max_depth_difference:    The maximum depth difference to allow between two neighbouring pixels in the
                                        same region.
        """
        # noinspection PyArgumentList
        cy, cx = cuda.grid(2)

        if cy < depth_image.shape[0] and cx < depth_image.shape[1]:
            # Get the depth of the current pixel.
            cdepth: float = depth_image[cy, cx]

            # For each potential neighbour of the current pixel:
            k_squared: int = 25
            k: int = int(math.sqrt(float(k_squared)))
            centre: int = k_squared // 2
            half_k: int = k // 2

            for i in range(k_squared):
                dy, dx = divmod(i, k)
                x = cx - half_k + dx
                y = cy - half_k + dy

                # Skip the current pixel itself and any of its neighbours that are outside the image bounds.
                if not(i != centre and 0 <= y < depth_image.shape[0] and 0 <= x < depth_image.shape[1]):
                    continue

                # If the difference between the neighbour's depth and the current pixel's depth is above a threshold:
                if math.fabs(depth_image[y, x] - cdepth) > max_depth_difference:
                    # Mark the current pixel as a depth edge and early out.
                    depth_edges[cy, cx] = 0
                    break
