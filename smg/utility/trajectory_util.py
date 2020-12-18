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

    # @staticmethod
    # def load_trajectory(filename: str, *, canonicalise_poses: bool = False) -> np.ndarray:
    #     """
    #     Load a KITTI or TUM trajectory from a file.
    #
    #     :param filename:            The name of the file containing the trajectory.
    #     :param canonicalise_poses:  Whether or not to canonicalise the poses in the trajectory (i.e. transform each
    #                                 pose by the inverse of the first pose in the trajectory to ensure that the new
    #                                 first pose is the identity).
    #     :return:                    The trajectory (as an n*3*4 numpy array, where n is the sequence length).
    #     """
    #     # Read in all of the lines in the file.
    #     with open(filename, "r") as f:
    #         lines = f.read().split("\n")
    #
    #     # Convert the lines into an n*m float array, where n is the sequence length and m is the number of elements
    #     # in each line. The file will either be in KITTI format, in which case each line will contain a 3*4 rigid body
    #     # transform (in row major order), and m will be 12, or it will be in TUM format, in which case each line will
    #     # contain a timestamp, translation vector and quaternion, and be of the form "timestamp tx ty tz qx qy qz qw",
    #     # and m will be 8.
    #     transforms: np.ndarray = np.array([list(map(float, line.split(" "))) for line in lines if line])
    #
    #     # noinspection PyUnusedLocal
    #     new_transforms: np.ndarray
    #
    #     if transforms.shape[1] == 12:
    #         # If the file was in KITTI format, the n*12 array can simply be reshaped to n*3*4.
    #         new_transforms = transforms.reshape((-1, 3, 4))
    #     else:
    #         # If the file was in TUM format, we need to determine the 3*4 matrix manually for each frame.
    #         new_transforms = np.zeros((transforms.shape[0], 3, 4))
    #         for frame_idx, transform in enumerate(transforms):
    #             new_transforms[frame_idx] = np.eye(4)[0:3, :]
    #             new_transforms[frame_idx, 0:3, 3] = transform[1:4]
    #             r: Rotation = Rotation.from_quat(transform[4:])
    #             new_transforms[frame_idx, 0:3, 0:3] = r.as_matrix()
    #
    #     if canonicalise_poses:
    #         TrajectoryUtil.transform_trajectory(
    #             new_transforms, pre=np.linalg.inv(GeometryUtil.to_4x4(new_transforms[0]))
    #         )
    #
    #     return new_transforms
    #
    # @staticmethod
    # def load_tum_timestamps(filename: str) -> List[float]:
    #     """
    #     Load the frame timestamps for a TUM trajectory from a file.
    #
    #     :param filename:    The name of the file containing the trajectory.
    #     :return:            The frame timestamps for the trajectory.
    #     """
    #     # Read in all of the lines in the file.
    #     with open(filename, "r") as f:
    #         lines = f.read().split("\n")
    #
    #     # Make a list of the timestamps and return them.
    #     timestamps: List[float] = []
    #     for line in lines:
    #         if line:
    #             timestamp: float = float(line.split(" ")[0])
    #             timestamps.append(timestamp)
    #
    #     return timestamps
    #
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
