import itertools
import numpy as np

from typing import List


class PoseUtil:
    """Utility functions relating to poses."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def load_pose(filename: str) -> np.ndarray:
        """
        Load a 4x4 pose matrix from the specified file.

        :param filename:    The name of the pose file.
        :return:            The 4x4 pose matrix.
        """
        with open(filename, "r") as file:
            # : List[str]
            data = file.read().split()

        if len(data) != 16:
            raise Exception('Cannot parse pose in "{}".'.format(filename))

        # : np.ndarray
        pose = np.eye(4, dtype=np.float32)

        for y, x in itertools.product(range(3), range(4)):
            pose[y, x] = float(data[y * 4 + x])

        return pose

    @staticmethod
    def save_pose(pose_filename: str, pose: np.ndarray) -> None:
        """
        Save a 6D pose to disk.

        :param pose_filename:   The name of the file to which to save it.
        :param pose:            The pose, as a 4x4 matrix.
        """
        with open(pose_filename, "w") as f:
            for i in range(4):
                line = " ".join(map(str, pose[i])) + "\r"
                f.write(line)
