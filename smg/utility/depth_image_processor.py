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
    def remove_isolated_regions(depth_image: np.ndarray, segmentation: np.ndarray, stats: np.ndarray,
                                *, min_region_size: int) -> Tuple[np.ndarray, np.ndarray]:
        replace = {}  # type: Dict[int, int]
        for i in range(stats.shape[0]):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_region_size:
                replace[i] = 0

        indexer = np.array([replace.get(i, i) for i in range(0, np.max(segmentation) + 1)])

        segmentation = indexer[segmentation]
        depth_image = np.where(segmentation != 0, depth_image, 0.0)

        return depth_image, segmentation

    @staticmethod
    def segment_depth_image(depth_image: np.ndarray, *, threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Make the depth edge map.
        depth_edges = np.full_like(depth_image, 255, dtype=np.uint8)  # type: np.ndarray
        DepthImageProcessor.__make_depth_edge_map(depth_image, depth_edges, threshold)

        # Find the connected components of the depth edge map.
        _, segmentation, stats, _ = cv2.connectedComponentsWithStats(depth_edges)

        return segmentation, stats, depth_edges

    # PRIVATE STATIC METHODS

    @staticmethod
    def __make_depth_edge_map(depth_image, depth_edges, threshold: float) -> None:
        NumbaUtil.launch_kernel_2d(
            DepthImageProcessor.__ck_make_depth_edge_map, depth_image, depth_edges, threshold,
            grid_size=depth_image.shape
        )

    # PRIVATE STATIC CUDA KERNELS

    @staticmethod
    @cuda.jit
    def __ck_make_depth_edge_map(depth_image, depth_edges, threshold: float):
        # noinspection PyArgumentList
        cy, cx = cuda.grid(2)
        if cy < depth_image.shape[0] and cx < depth_image.shape[1]:
            cdepth: float = depth_image[cy, cx]

            k_squared: int = 25
            k: int = int(math.sqrt(float(k_squared)))
            centre: int = k_squared // 2
            half_k: int = k // 2

            # For each potential neighbour of the current pixel:
            for i in range(k_squared):
                dy, dx = divmod(i, k)
                x = cx - half_k + dx
                y = cy - half_k + dy

                # Skip the current pixel and any neighbours that are outside the image bounds.
                if not(i != centre and 0 <= y < depth_image.shape[0] and 0 <= x < depth_image.shape[1]):
                    continue

                if math.fabs(depth_image[y, x] - cdepth) > threshold:
                    depth_edges[cy, cx] = 0
                    break
