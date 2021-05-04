import numpy as np

from scipy.spatial.transform import Rotation
from typing import List, Tuple

from .trajectory_smoother import TrajectorySmoother


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
        data = np.array([list(map(float, line.split(" "))) for line in lines if line])  # type: np.ndarray

        # Construct and return the output list.
        result = []  # type: List[Tuple[float, np.ndarray]]
        for i in range(data.shape[0]):
            timestamp, tx, ty, tz, qx, qy, qz, qw = data[i, :]
            r = Rotation.from_quat([qx, qy, qz, qw])  # type: Rotation
            pose = np.eye(4)  # type: np.ndarray
            pose[0:3, 0:3] = r.as_matrix()
            pose[0:3, 3] = [tx, ty, tz]
            result.append((timestamp, pose))

        return result

    @staticmethod
    def smooth_trajectory(trajectory: List[Tuple[float, np.ndarray]], *, neighbourhood_size: int = 25) \
            -> List[Tuple[float, np.ndarray]]:
        """
        Smooth a trajectory using Laplacian smoothing.

        :param trajectory:          The trajectory to smooth.
        :param neighbourhood_size:  The neighbourhood size for the Laplacian smoothing.
        :return:                    The smoothed trajectory.
        """
        smoother = TrajectorySmoother(neighbourhood_size=neighbourhood_size)  # type: TrajectorySmoother
        for timestamp, pose in trajectory:
            smoother.append(timestamp, pose)
        return smoother.get_smoothed_trajectory()

    @staticmethod
    def write_tum_pose(f, timestamp: float, pose: np.ndarray) -> None:
        """
        Write a timestamped pose to a TUM trajectory file.

        :param f:           The TUM trajectory file.
        :param timestamp:   The timestamp.
        :param pose:        The pose.
        """
        r = Rotation.from_matrix(pose[0:3, 0:3])  # type: Rotation
        t = pose[0:3, 3]  # type: np.ndarray
        f.write(" ".join([str(timestamp)] + list(map(str, t)) + list(map(str, r.as_quat()))))
        f.write("\n")
