import numpy as np


class PoseUtil:
    """Utility functions relating to poses."""

    # PUBLIC STATIC METHODS

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
