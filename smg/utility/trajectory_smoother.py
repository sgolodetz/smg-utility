import numpy as np

from typing import List, Optional, Tuple


class TrajectorySmoother:
    """Used to smooth trajectories using Laplacian smoothing."""

    # CONSTRUCTOR

    def __init__(self, *, neighbourhood_size: Optional[int] = None):
        """
        Construct a trajectory smoother.

        :param neighbourhood_size:  The neighbourhood size for the Laplacian smoothing.
        """
        self.__half_neighbourhood_size: Optional[int] = neighbourhood_size // 2 \
            if neighbourhood_size is not None else None
        self.__raw_trajectory: List[Tuple[float, np.ndarray]] = []
        self.__smoothed_trajectory: List[Tuple[float, np.ndarray]] = []

    # PUBLIC METHODS

    def append(self, timestamp: float, pose: np.ndarray) -> None:
        """
        Append a timestamped pose to the raw trajectory that is being smoothed.

        :param timestamp:   The timestamp.
        :param pose:        The pose.
        """
        # Append the timestamped pose to the raw trajectory.
        self.__raw_trajectory.append((timestamp, pose))

        # If a neighbourhood size was specified (and therefore smoothing is enabled):
        if self.__half_neighbourhood_size is not None:
            # If possible, smooth an earlier timestamped pose and append it to the smoothed trajectory.
            high: int = len(self.__raw_trajectory) - 1
            i: int = high - self.__half_neighbourhood_size
            low: int = i - self.__half_neighbourhood_size

            if low >= 0:
                t: np.ndarray = np.zeros(3)

                for j in range(low, high + 1):
                    _, pose_j = self.__raw_trajectory[j]
                    t += pose_j[0:3, 3]

                t /= (high + 1 - low)

                timestamp_i, pose_i = self.__raw_trajectory[i]
                smoothed_pose_i: np.ndarray = pose_i.copy()
                smoothed_pose_i[0:3, 3] = t
                self.__smoothed_trajectory.append((timestamp_i, smoothed_pose_i))
        else:
            # If smoothing is not in use, simply append the raw pose to the smoothed trajectory.
            self.__smoothed_trajectory.append((timestamp, pose))

    def get_raw_trajectory(self) -> List[Tuple[float, np.ndarray]]:
        """
        Get the raw trajectory that is being smoothed.

        :return:    The raw trajectory that is being smoothed.
        """
        return self.__raw_trajectory

    def get_smoothed_trajectory(self) -> List[Tuple[float, np.ndarray]]:
        """
        Get the smoothed trajectory.

        :return:    The smoothed trajectory.
        """
        return self.__smoothed_trajectory
