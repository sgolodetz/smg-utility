import math
import numpy as np

from scipy.spatial.transform import Rotation
from typing import List, Tuple

# from smg.utility.geometry_util import GeometryUtil


class TrajectoryUtil:
    """Utility functions related to trajectories."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def load_tum_trajectory(filename: str) -> List[Tuple[float, np.ndarray]]:
        """
        Load a TUM trajectory from a file.

        :param filename:    The name of the file containing the trajectory.
        :return:            The trajectory, as a list of (timestamp, pose) pairs.
        """
        # Read in all of the lines in the file.
        with open(filename, "r") as f:
            lines = f.read().split("\n")

        # Convert the lines into an n*8 data array, where n is the trajectory length and each row is of
        # the form "timestamp tx ty tz qx qy qz qw".
        data: np.ndarray = np.array([list(map(float, line.split(" "))) for line in lines if line])

        # Construct and return the output list.
        result: List[Tuple[float, np.ndarray]] = []
        for i in range(data.shape[0]):
            timestamp, tx, ty, tz, qx, qy, qz, qw = data[i, :]
            r: Rotation = Rotation.from_quat([qx, qy, qz, qw])
            pose: np.ndarray = np.eye(4)
            pose[0:3, 0:3] = r.as_matrix()
            pose[0:3, 3] = [tx, ty, tz]
            result.append((timestamp, pose))

        return result

    @staticmethod
    def smooth_trajectory(trajectory: List[Tuple[float, np.ndarray]], *, neighbourhood_size: int = 35) \
            -> List[Tuple[float, np.ndarray]]:
        """
        Smooth a trajectory using Laplacian smoothing.

        :param trajectory:          The trajectory to smooth.
        :param neighbourhood_size:  The neighbourhood size for the Laplacian smoothing.
        :return:                    The smoothed trajectory.
        """
        half_neighbourhood_size: int = neighbourhood_size // 2
        new_trajectory: List[Tuple[float, np.ndarray]] = []
        for i in range(len(trajectory)):
            low: int = max(i - half_neighbourhood_size, 0)
            high: int = min(i + half_neighbourhood_size, len(trajectory) - 1)
            t: np.ndarray = np.zeros(3)
            for j in range(low, high + 1):
                _, pose_j = trajectory[j]
                t += pose_j[0:3, 3]
            timestamp_i, pose_i = trajectory[i]
            new_pose_i: np.ndarray = pose_i.copy()
            new_pose_i[0:3, 3] = t / (high + 1 - low)
            new_trajectory.append((timestamp_i, new_pose_i))
        return new_trajectory

    # @staticmethod
    # def transform_trajectory(trajectory: np.ndarray, *, pre: np.ndarray = np.eye(4), post: np.ndarray = np.eye(4)) \
    #         -> None:
    #     """
    #     Apply the specified rigid-body transforms to each pose in a trajectory (in-place).
    #
    #     :param trajectory:  The trajectory whose poses should be transformed.
    #     :param pre:         The rigid-body transform with which to pre-multiply each pose (expressed as a 4*4 matrix).
    #     :param post:        The rigid-body transform with which to post-multiply each pose (expressed as a 4*4 matrix).
    #     """
    #     for frame_idx in range(trajectory.shape[0]):
    #         trajectory[frame_idx] = GeometryUtil.to_3x4(pre @ GeometryUtil.to_4x4(trajectory[frame_idx]) @ post)

    @staticmethod
    def write_tum_pose(f, timestamp: float, pose: np.ndarray) -> None:
        """
        Write a timestamped pose to a TUM trajectory file.

        :param f:           The TUM trajectory file.
        :param timestamp:   The timestamp.
        :param pose:        The pose.
        """
        r: Rotation = Rotation.from_matrix(pose[0:3, 0:3])
        t: np.ndarray = pose[0:3, 3]
        f.write(" ".join([str(timestamp)] + list(map(str, t)) + list(map(str, r.as_quat()))))
        f.write("\n")
