import math
import numpy as np
import vg

from collections import defaultdict
from itertools import product
from numba import cuda
from typing import Dict, List, Optional, Tuple, Union

from .dual_quaternion import DualQuaternion
from .image_util import ImageUtil
from .numba_util import NumbaUtil


class GeometryUtil:
    """Utility functions related to geometry."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def angle_between(v1: np.ndarray, v2: np.ndarray) -> Optional[float]:
        """
        Try to calculate the angle (in radians) between two vectors.

        .. note::
            If the magnitude of either vector is too small, this will return None.

        :param v1:  The first vector.
        :param v2:  The second vector.
        :return:    The angle (in radians) between the two vectors, if possible, or None otherwise.
        """
        # Calculate the magnitudes of the two vectors.
        v1_length = np.linalg.norm(v1)  # type: float
        v2_length = np.linalg.norm(v2)  # type: float

        # If both are long enough, calculate and return the angle between them.
        if v1_length >= 1e-4 and v2_length >= 1e-4:
            return np.arccos(np.clip(np.dot(v1, v2) / (v1_length * v2_length), -1.0, 1.0))

        # Otherwise, return None.
        else:
            return None

    @staticmethod
    def apply_rigid_transform(mat4x4: np.ndarray, vec3: np.ndarray) -> np.ndarray:
        """
        Apply a rigid-body transform expressed as a 4x4 matrix to a 3D vector.

        :param mat4x4:  The 4x4 matrix.
        :param vec3:    The 3D vector.
        :return:        The result of applying the rigid-body transform to the 3D vector.
        """
        return GeometryUtil.to_3x4(mat4x4) @ np.array([*vec3, 1.0])

    @staticmethod
    def blend_rigid_transforms(transforms: List[np.ndarray]) -> np.ndarray:
        """
        Linearly blend a set of rigid-body transforms together to construct a refined transform.

        :param transforms:  The transforms to blend.
        :return:            The refined transform.
        """
        dqs = [DualQuaternion.from_rigid_matrix(m) for m in transforms]  # type: List[DualQuaternion]
        weight = 1.0 / len(transforms)                                   # type: float
        weights = [weight for _ in transforms]                           # type: List[float]
        return DualQuaternion.linear_blend(dqs, weights).to_rigid_matrix()

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
                                        intrinsics: Tuple[float, float, float, float]) -> np.ndarray:
        """
        Compute a world-space points image from a depth image, camera pose, and set of camera intrinsics.

        .. note::
            The origin, i.e. (0, 0, 0), is stored for pixels whose depth is zero.

        :param depth_image: The depth image (pixels with zero depth are treated as invalid).
        :param pose:        The camera pose (denoting a transformation from camera space to world space).
        :param intrinsics:  The camera intrinsics, as an (fx, fy, cx, cy) tuple.
        :return:            The world-space points image.
        """
        # TODO: This should ultimately replace the compute_world_points_image function above.
        height, width = depth_image.shape
        ws_points = np.zeros((height, width, 3), dtype=float)  # type: np.ndarray
        fx, fy, cx, cy = intrinsics
        xl = np.array(range(width))
        yl = np.array(range(height))
        al = np.tile((xl - cx) / fx, height).reshape(height, width) * depth_image
        bl = np.transpose(np.tile((yl - cy) / fy, width).reshape(width, height)) * depth_image
        cl = (depth_image > 0.0).astype(np.uint8)
        for i in range(3):
            ws_points[:, :, i] = pose[i, 0] * al + pose[i, 1] * bl + pose[i, 2] * depth_image + pose[i, 3] * cl
        return ws_points

    @staticmethod
    def distance_to_line_segment(point, start, end) -> float:
        """
        Calculate the distance between the specified point and the closest point to it on a line segment.

        :param point:   The specified point.
        :param start:   One endpoint of the line segment.
        :param end:     The other endpoint of the line segment.
        :return:        The distance between the specified point and the closest point to it on the line segment.
        """
        point = np.array(point)  # type: np.ndarray
        start = np.array(start)  # type: np.ndarray
        end = np.array(end)      # type: np.ndarray

        closest_point: np.ndarray = GeometryUtil.find_closest_point_on_line_segment(point, start, end)
        return np.linalg.norm(closest_point - point)

    @staticmethod
    def estimate_rigid_transform(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Estimate the rigid body transform between two sets of corresponding 3D points (using the Kabsch algorithm).

        .. note::
            This function was adapted from SemanticPaint: for academic/non-commercial use only.

        :param p:   The first set of 3D points.
        :param q:   The second set of 3D points.
        :return:    The estimated rigid body transform between the two sets of points.
        """
        # Step 1: Count the correspondences.
        n = p.shape[1]  # type: int

        # Step 2: Compute the centroids of the two sets of points. For n = 3:
        #
        # centroid = (x1 x2 x3) * (1/3) = ((x1 + x2 + x3) / 3) = (cx)
        #            (y1 y2 y3) * (1/3) = ((y1 + y2 + y3) / 3) = (cy)
        #            (z1 z2 z3) * (1/3) = ((z1 + z2 + z3) / 3) = (cz)
        nths = np.full((n, 1), 1 / n)  # type: np.ndarray
        centroid_p = p.dot(nths)       # type: np.ndarray
        centroid_q = q.dot(nths)       # type: np.ndarray

        # Step 3: Translate the points in each set so that their centroid coincides with the origin
        #         of the coordinate system. To do this, we subtract the centroid from each point.
        #
        # centred = (x1 x2 x3) - (cx) * (1 1 1) = (x1 x2 x3) - (cx cx cx) = (x1-cx x2-cx x3-cx)
        #           (y1 y2 y3)   (cy)             (y1 y2 y3)   (cy cy cy)   (y1-cy y2-cy y3-cy)
        #           (z1 z2 z3)   (cz)             (z1 z2 z3)   (cz cz cz)   (z1-cz z2-cz z3-cz)
        ones_t = np.ones((1, n))                # type: np.ndarray
        centred_p = p - centroid_p.dot(ones_t)  # type: np.ndarray
        centred_q = q - centroid_q.dot(ones_t)  # type: np.ndarray

        # Step 4: Compute the cross-covariance between the two matrices of centred points.
        a = centred_p.dot(centred_q.transpose())  # type: np.ndarray

        # Step 5: Calculate the SVD of the cross-covariance matrix: a = v * s * w^T.
        v, s, w_t = np.linalg.svd(a, full_matrices=True)
        w = w_t.transpose()

        # Step 6: Decide whether or not we need to correct our rotation matrix, and set the i matrix accordingly.
        i = np.eye(3)
        if np.linalg.det(np.dot(v, w).transpose()) < 0:
            i[2, 2] = -1

        # Step 7: Recover the rotation and translation estimates.
        r = w.dot(i).dot(v.transpose())
        t = centroid_q - r.dot(centroid_p)

        # Step 8: Combine the estimates into a 4x4 homogeneous transformation, and return it.
        m = np.eye(4)  # type: np.ndarray
        m[0:3, 0:3] = r
        m[0:3, 3] = np.transpose(t)
        return m

    @staticmethod
    def find_closest_point_on_half_ray(point, start, direction) -> np.ndarray:
        """
        Find the closest point on a half-ray to the specified point.

        :param point:       The specified point.
        :param start:       The start of the half-ray.
        :param direction:   The direction of the half-ray.
        :return:            The closest point on the half-ray to the specified point.
        """
        point = np.array(point)          # type: np.ndarray
        start = np.array(start)          # type: np.ndarray
        direction = np.array(direction)  # type: np.ndarray

        direction = vg.normalize(direction)
        t = np.dot(point - start, direction)  # type: float
        return start + t * direction if t > 0 else start

    @staticmethod
    def find_closest_point_on_line_segment(point, start, end) -> np.ndarray:
        """
        Find the closest point on a line segment to the specified point.

        :param point:   The specified point.
        :param start:   One endpoint of the line segment.
        :param end:     The other endpoint of the line segment.
        :return:        The closest point on the line segment to the specified point.
        """
        point = np.array(point)  # type: np.ndarray
        start = np.array(start)  # type: np.ndarray
        end = np.array(end)      # type: np.ndarray

        direction = end - start                                                # type: np.ndarray
        direction_length = np.linalg.norm(direction)                           # type: float
        if direction_length <= 1e-4:
            return start

        t = vg.scalar_projection(point - start, direction) / direction_length  # type: float
        if t <= 0:
            return start
        elif t >= 1:
            return end
        else:
            return start + t * direction

    @staticmethod
    def find_largest_cluster(transforms: List[np.ndarray], *, rotation_threshold: float = 20 * math.pi / 180,
                             translation_threshold: float = 0.05) -> Optional[List[int]]:
        """
        Try to cluster the specified rigid-body transforms and return the indices of the transforms in a largest
        cluster.

        .. note::
            If the list of input transforms is empty, there is no largest cluster, so this returns None.
        .. note::
            If there's more than one largest cluster, only one of them will be returned, but the result will always
            be the same for the same (ordered) list of input transforms (i.e. there's no randomness involved).

        :param transforms:              The input transforms.
        :param rotation_threshold:      The angular threshold to use when comparing rotations.
        :param translation_threshold:   The distance threshold to use when comparing translations.
        :return:                        The indices of the transforms in a largest cluster, if possible, or None
                                        otherwise.
        """
        clusters = defaultdict(list)  # type: Dict[int, List[int]]

        largest_cluster_index = -1    # type: int
        largest_cluster_size = 0      # type: int

        # For each transform:
        for i in range(len(transforms)):
            # Compute the cluster for the transform.
            for j in range(len(transforms)):
                if GeometryUtil.transforms_are_similar(
                    transforms[i], transforms[j],
                    rotation_threshold=rotation_threshold, translation_threshold=translation_threshold
                ):
                    clusters[i].append(j)

            # Update which cluster is the largest as necessary.
            cluster_size = len(clusters[i])  # type: int
            if cluster_size > largest_cluster_size:
                largest_cluster_index = i
                largest_cluster_size = cluster_size

        # Return the largest cluster, if there is one, or None otherwise.
        return clusters[largest_cluster_index] if largest_cluster_index != -1 else None

    @staticmethod
    def find_plane_intersection(start, direction, plane: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        """
        Find the unique intersection point (if any) between a line and a plane.

        :param start:       The start point of the line.
        :param direction:   The direction vector of the line.
        :param plane:       The plane, as an (a,b,c,d) tuple denoting the plane ax + by + cz - d = 0.
        :return:            The unique intersection point (if any) between the line and the plane, or None otherwise.
        """
        s = np.array(start).astype(float)           # type: np.ndarray
        v = np.array(direction).astype(float)       # type: np.ndarray
        a, b, c, d = np.array(plane).astype(float)
        n = np.array([a, b, c])                     # type: np.ndarray

        denom = np.dot(n, v)                        # type: float
        if np.fabs(denom) <= 1e-4:
            return None

        t = (d - np.dot(n, s)) / denom              # type: float
        return s + t * v

    @staticmethod
    def find_reprojection_correspondences(
        source_depth_image: np.ndarray, world_from_source: np.ndarray, world_from_target: np.ndarray,
        source_intrinsics: Tuple[float, float, float, float], *,
        target_image_size: Optional[Tuple[int, int]] = None,
        target_intrinsics: Optional[Tuple[float, float, float, float]] = None
    ) -> np.ndarray:
        """
        Make a reprojection correspondence image by back-projecting the pixels in a source depth image,
        transforming them into the camera space of a target image, and reprojecting them into that image.

        .. note::
            The reprojection correspondence image contains a tuple (x',y') for each source pixel (x,y) that denotes
            the target pixel corresponding to it. If a source pixel does not have a valid reprojection correspondence,
            (-1,-1) will be stored for it.
        .. note::
            There are two reasons why a source pixel might not have a valid reprojection correspondence:
             (i) Its depth is 0, so it can't be back-projected.
            (ii) It has a correspondence that's in the target image plane, but not within the target image bounds.

        :param source_depth_image:  The source depth image.
        :param world_from_source:   A transformation from source camera space to world space.
        :param world_from_target:   A transformation from target camera space to world space.
        :param source_intrinsics:   The source camera intrinsics.
        :param target_image_size:   The target image size, if not the same as the source image size.
        :param target_intrinsics:   The target camera intrinsics, if not the same as the source camera intrinsics.
        :return:                    The reprojection correspondence image.
        """
        # Back-project the points from the source depth image and transform them into the camera space
        # of the target image so that they can be reprojected down onto that image.
        target_from_source = np.linalg.inv(world_from_target) @ world_from_source  # type: np.ndarray
        target_points = GeometryUtil.compute_world_points_image_fast(
            source_depth_image, target_from_source, source_intrinsics
        )  # type: np.ndarray

        # Reproject the points down onto the target image to find the correspondences. For each point, the relevant
        # equations are x = fx * X / Z + cx and y = fy * Y / Z + cy. Note that Z can be 0 for a particular pixel,
        # which is difficult to deal with in a vectorised operation. However, numpy will simply yield a NaN if we
        # divide by zero, and we can suppress the warnings that are produced, so that's good enough. Note that if
        # we then convert one of the NaNs to an int, we get INT_MIN in practice.
        if target_intrinsics is None:
            target_intrinsics = source_intrinsics
        fx, fy, cx, cy = target_intrinsics
        source_height, source_width = source_depth_image.shape
        correspondence_image = np.zeros((source_height, source_width, 2), dtype=int)  # type: np.ndarray
        np.seterr(divide="ignore", invalid="ignore")
        xs = np.round(fx * target_points[:, :, 0] / target_points[:, :, 2] + cx).astype(int)
        ys = np.round(fy * target_points[:, :, 1] / target_points[:, :, 2] + cy).astype(int)
        np.seterr(divide="warn", invalid="warn")
        correspondence_image[:, :, 0] = xs
        correspondence_image[:, :, 1] = ys

        # Replace any correspondences that are not within the image bounds with (-1, -1). Note that this will also
        # filter out points that had a Z of 0, since (-INT_MIN, -INT_MIN) is definitely outside the image bounds.
        if target_image_size is None:
            target_image_size = (source_height, source_width)
        target_height, target_width = target_image_size
        correspondence_image = np.where(np.atleast_3d(0 <= xs), correspondence_image, -1)
        correspondence_image = np.where(np.atleast_3d(xs < target_width), correspondence_image, -1)
        correspondence_image = np.where(np.atleast_3d(0 <= ys), correspondence_image, -1)
        correspondence_image = np.where(np.atleast_3d(ys < target_height), correspondence_image, -1)

        return correspondence_image

    @staticmethod
    def intrinsics_to_matrix(intrinsics: Tuple[float, float, float, float]) -> np.ndarray:
        """
        Convert camera intrinsics expressed as an (fx,fy,cx,cy) tuple to 3x3 matrix form.

        :param intrinsics:  The camera intrinsics in tuple form.
        :return:            The camera intrinsics in matrix form.
        """
        fx, fy, cx, cy = intrinsics
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

    @staticmethod
    def intrinsics_to_tuple(intrinsics: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Convert camera intrinsics expressed as a 3x3 matrix to an (fx,fy,cx,cy) tuple.

        :param intrinsics:  The camera intrinsics in matrix form.
        :return:            The camera intrinsics in tuple form.
        """
        return intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2]

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

        :param colour_image:  The colour image to use to colourise the point cloud (assumed to be in BGR format).
        :param depth_image:   The depth image whose pixels are to be back-projected to make the point cloud.
        :param depth_mask:    A mask indicating which pixels have valid depths (non-zero means valid).
        :param intrinsics:    The camera intrinsics (assumed to be the same for both the colour/depth cameras).
        """
        # Create a camera-space points image for the frame by using the identity pose.
        height, width = depth_image.shape
        cs_points = np.zeros((height, width, 3), dtype=np.float32)  # type: np.ndarray
        # noinspection PyTypeChecker
        GeometryUtil.compute_world_points_image(depth_image, depth_mask, np.eye(4), *intrinsics, cs_points)

        # Allocate the output point cloud arrays.
        pcd_points = np.zeros((height * width, 3))  # type: np.ndarray
        pcd_colours = np.ones((height * width, 3))  # type: np.ndarray

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
        vals = {}  # type: Dict[int, np.ndarray]
        for i in range(3):
            maxs[i] = mins[i] + math.ceil((maxs[i] - mins[i]) / voxel_size[i]) * voxel_size[i]
            vals[i] = np.linspace(mins[i], maxs[i], int((maxs[i] - mins[i]) / voxel_size[i]) + 1)

        xpts1 = [(mins[0], y, z) for y, z in product(vals[1], vals[2])]  # type: List[Tuple[float, float, float]]
        xpts2 = [(maxs[0], y, z) for y, z in product(vals[1], vals[2])]  # type: List[Tuple[float, float, float]]

        ypts1 = [(x, mins[1], z) for x, z in product(vals[0], vals[2])]  # type: List[Tuple[float, float, float]]
        ypts2 = [(x, maxs[1], z) for x, z in product(vals[0], vals[2])]  # type: List[Tuple[float, float, float]]

        zpts1 = [(x, y, mins[2]) for x, y in product(vals[0], vals[1])]  # type: List[Tuple[float, float, float]]
        zpts2 = [(x, y, maxs[2]) for x, y in product(vals[0], vals[1])]  # type: List[Tuple[float, float, float]]

        pts1 = xpts1 + ypts1 + zpts1  # type: List[Tuple[float, float, float]]
        pts2 = xpts2 + ypts2 + zpts2  # type: List[Tuple[float, float, float]]

        return pts1, pts2

    @staticmethod
    def rescale_intrinsics(old_intrinsics: Tuple[float, float, float, float], old_image_size: Tuple[int, int],
                           new_image_size: Tuple[int, int]):
        """
        Rescale a set of camera intrinsics to allow them to be used with a different window size.

        :param old_intrinsics:  The old camera intrinsics, as an (fx, fy, cx, cy) tuple.
        :param old_image_size:  The old window size.
        :param new_image_size:  The new window size.
        :return:                The new camera intrinsics.
        """
        fx, fy, cx, cy = old_intrinsics
        fractions = (
            new_image_size[0] / old_image_size[0], new_image_size[1] / old_image_size[1]
        )  # type: Tuple[float, float]
        return fx * fractions[0], fy * fractions[1], cx * fractions[0], cy * fractions[1]

    @staticmethod
    def select_pixels_from(target_image: np.ndarray, selection_image: np.ndarray, *, invalid_value=0) -> np.ndarray:
        """
        Select pixels from a target image based on a selection image.

        .. note::
            Each pixel in the selection image contains a tuple (x,y) denoting a pixel that should be selected
            from the target image. Note that (-1,-1) can be used to denote an invalid pixel, i.e. one for which
            we do not want to select a target pixel).
        .. note::
            The output image will be the same size as the selection image.

        :param target_image:    The target image.
        :param selection_image: The selection image.
        :param invalid_value:   The value to store in the output image for invalid pixels.
        :return:                The output image.
        """
        # Copy the corresponding pixels from the target image into the output image. Note that the correspondence
        # image contains (-1, -1) for pixels without a valid correspondence, so we can safely index the target
        # with this, but we need to then filter out those pixels after the fact.
        output_image = target_image[selection_image[:, :, 1], selection_image[:, :, 0]]  # type: np.ndarray

        # Filter out the invalid pixels. Different implementations are needed depending on the number of channels.
        if len(output_image.shape) > 2 and output_image.shape[2] == 3:
            for i in range(2):
                output_image = np.where(np.atleast_3d(selection_image[:, :, i] >= 0), output_image, invalid_value)
        else:
            for i in range(2):
                output_image = np.where(selection_image[:, :, i] >= 0, output_image, invalid_value)

        return output_image

    @staticmethod
    def to_3x4(mat4x4: np.ndarray) -> np.ndarray:
        """
        Convert a 4*4 matrix representation of a rigid-body transform to its 3*4 equivalent.

        :param mat4x4:  The 4*4 matrix representation of the rigid-body transform.
        :return:        The 3*4 matrix representation of the rigid-body transform.
        """
        return mat4x4[0:3, :]

    @staticmethod
    def to_4x4(mat3x4: np.ndarray) -> np.ndarray:
        """
        Convert a 3*4 matrix representation of a rigid-body transform to its 4*4 equivalent.

        :param mat3x4:  The 3*4 matrix representation of the rigid-body transform.
        :return:        The 4*4 matrix representation of the rigid-body transform.
        """
        mat4x4 = np.eye(4)  # type: np.ndarray
        mat4x4[0:3, :] = mat3x4
        return mat4x4

    @staticmethod
    def transforms_are_similar(transform1: Union[DualQuaternion, np.ndarray],
                               transform2: Union[DualQuaternion, np.ndarray], *,
                               rotation_threshold: float = 20 * math.pi / 180,
                               translation_threshold: float = 0.05) -> bool:
        """
        Determine whether or not two SE(3) transforms are similar.

        .. note::
            Similarity is defined in terms of both the rotations and translations involved. Rotation similarity is
            assessed by looking at the relative rotation mapping one of the two input rotations to the other, and
            thresholding the angle involved. Translation similarity is assessed by thresholding the distance between
            the two input translations. Iff both their rotations and translations are similar, so are the transforms.

        :param transform1:              The first rigid-body transform.
        :param transform2:              The second rigid-body transform.
        :param rotation_threshold:      The angular threshold to use when comparing the rotations.
        :param translation_threshold:   The distance threshold to use when comparing the translations.
        :return:                        True, if the transforms are similar, or False otherwise.
        """
        dq1 = transform1 if type(transform1) is DualQuaternion \
            else DualQuaternion.from_rigid_matrix(transform1)  # type: DualQuaternion
        dq2 = transform2 if type(transform2) is DualQuaternion \
            else DualQuaternion.from_rigid_matrix(transform2)  # type: DualQuaternion

        rot = DualQuaternion.angle_between_rotations(dq1.get_rotation_part(), dq2.get_rotation_part())  # type: float
        trans = np.linalg.norm(dq1.get_translation() - dq2.get_translation())                           # type: float

        return rot <= rotation_threshold and trans <= translation_threshold

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
            a = (x - cx) / fx  # type: float
            b = (y - cy) / fy  # type: float

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
