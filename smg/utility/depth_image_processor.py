import cv2
import math
import numpy as np

from numba import cuda
from typing import Dict, Optional, Tuple

from .geometry_util import GeometryUtil
from .numba_util import NumbaUtil


class DepthImageProcessor:
    """Utility functions for post-processing depth images."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def postprocess_depth_image(depth_image: np.ndarray, *, max_depth: float, max_depth_difference: float,
                                median_filter_radius: int, min_region_size: int, min_valid_fraction: float) \
            -> Optional[np.ndarray]:
        """
        Try to post-process the specified depth image to reduce the amount of noise it contains.

        .. note::
            This function will return None if the input depth image does not have depth values for enough pixels.

        :param depth_image:             The input depth image.
        :param max_depth:               The maximum depth values to keep (pixels with depth values greater than this
                                        will have their depths set to zero).
        :param max_depth_difference:    The maximum depth difference to allow between two neighbouring pixels in the
                                        same segmentation region.
        :param median_filter_radius:    The radius of the median filter to use to reduce impulsive noise at the end
                                        of the post-processing operation.
        :param min_region_size:         The minimum size of region to keep from the depth segmentation (that is,
                                        regions smaller than this will have their depths set to zero).
        :param min_valid_fraction:      The minimum fraction of pixels for which the input depth image must have
                                        depth values for the post-processing operation to succeed. (Note that we
                                        remove pixels whose depth values are greater than the specified maximum
                                        depth before performing this test.)
        :return:                        The post-processed depth image, if possible, or None otherwise.
        """
        # Limit the depth range (more distant points can be unreliable).
        depth_image = np.where(depth_image <= max_depth, depth_image, 0.0)

        # If we have depth values for more than the specified fraction of the remaining pixels:
        if np.count_nonzero(depth_image) / np.product(depth_image.shape) >= min_valid_fraction:
            # Segment the depth image into regions such that all of the pixels in each region have similar depth.
            segmentation, stats, _ = DepthImageProcessor.segment_depth_image(
                depth_image, max_depth_difference=max_depth_difference
            )

            # Remove any regions that are smaller than the specified size.
            depth_image, _ = DepthImageProcessor.remove_small_regions(
                depth_image, segmentation, stats, min_region_size=min_region_size
            )

            # Median filter the depth image to help mitigate impulsive noise.
            depth_image = cv2.medianBlur(depth_image, median_filter_radius)

            return depth_image

        # Otherwise, discard the depth image.
        else:
            return None

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
    def remove_temporal_inconsistencies(
        current_depth_image: np.ndarray, current_w_t_c: np.ndarray, current_world_points: np.ndarray,
        previous_depth_image: np.ndarray, previous_w_t_c: np.ndarray, previous_world_points: np.ndarray,
        intrinsics: Tuple[float, float, float, float], *,
        debug: bool = False, distance_threshold: float
    ) -> np.ndarray:
        """
        Make a filtered version of the current depth image by removing any pixel that either does not have a
        corresponding pixel in the previous world-space points image at all, or else does not have one whose
        world-space point is sufficiently close to the current world-space point.

        .. note::
            Since this makes use of the previous frame, it's deliberately not part of the normal post-processing
            we perform on individual depth images.

        :param current_depth_image:     The current depth image.
        :param current_w_t_c:           The current camera pose (as a camera -> world transform).
        :param current_world_points:    The current world-space points image.
        :param previous_depth_image:    The previous depth image.
        :param previous_w_t_c:          The previous camera pose (as a camera -> world transform).
        :param previous_world_points:   The previous world-space points image.
        :param intrinsics:              The camera intrinsics, as an (fx, fy, cx, cy) tuple.
        :param debug:                   Whether to show the internal images to aid debugging.
        :param distance_threshold:      The threshold (in m) defining what's meant by "sufficiently" close to the
                                        current world-space point (see above).
        :return:                        The filtered version of the current depth image.
        """
        # Reproject the previous depth image and world-space points image into the current image plane.
        selection_image = GeometryUtil.find_reprojection_correspondences(
            current_depth_image, current_w_t_c, previous_w_t_c, intrinsics
        )  # type: np.ndarray

        reprojected_depth_image = GeometryUtil.select_pixels_from(
            previous_depth_image, selection_image
        )  # type: np.ndarray

        reprojected_world_points = GeometryUtil.select_pixels_from(
            previous_world_points, selection_image
        )  # type: np.ndarray

        # Compute the distances between the reprojected world-space points from the previous frame and the current
        # world-space points.
        distance_image = np.linalg.norm(reprojected_world_points - current_world_points, axis=2)  # type: np.ndarray

        # Make a filtered version of the current depth image by removing any pixel that either does not have a
        # corresponding pixel in the previous world-space points image at all, or else does not have one whose
        # world-space point is close to the current world-space point.
        filtered_depth_image = np.where(
            (reprojected_depth_image > 0.0) & (distance_image <= distance_threshold), current_depth_image, 0.0
        )  # type: np.ndarray

        # If we're debugging, show the internal images.
        if debug:
            cv2.imshow("Unfiltered Depth Image", current_depth_image / 5)
            cv2.imshow("Warped Depth Image", reprojected_depth_image / 5)
            cv2.imshow("World-Space Points Distance Image", distance_image)
            cv2.waitKey(1)

        return filtered_depth_image

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
            cdepth = depth_image[cy, cx]  # type: float

            # For each potential neighbour of the current pixel:
            k_squared = 25                        # type: int
            k = int(math.sqrt(float(k_squared)))  # type: int
            centre = k_squared // 2               # type: int
            half_k = k // 2                       # type: int

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
