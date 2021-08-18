import numpy as np

from typing import Dict, Optional

from .geometry_util import GeometryUtil


class MarkerUtil:
    """Utility functions to calculate transformations between different spaces based on a marker."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def estimate_space_to_marker_transform(marker_positions: Dict[str, np.ndarray], *,
                                           half_width: float = 0.0705) -> Optional[np.ndarray]:
        """
        Try to estimate the transformation from a space S to the space M associated with a square marker
        (e.g. an ArUco marker) by making use of the known positions of the marker's corners in S.

        .. note::
            The ArUco marker I use is 14.1cm across, which is why the default half-width is set to 7.05cm.

        :param marker_positions:    The positions of the marker's corners in S (where known)
        :param half_width:          Half the width of the marker (in m).
        :return:                    The transformation from S to M, if possible, or None otherwise.
        """
        # If the positions in S of all of the marker's corners are known, estimate the transformation.
        if all(key in marker_positions for key in ["0_0", "0_1", "0_2", "0_3"]):
            p: np.ndarray = np.column_stack([
                marker_positions["0_0"],
                marker_positions["0_1"],
                marker_positions["0_2"],
                marker_positions["0_3"]
            ])

            q: np.ndarray = np.array([
                [-half_width, -half_width, 0],
                [half_width, -half_width, 0],
                [half_width, half_width, 0],
                [-half_width, half_width, 0]
            ]).transpose()

            return GeometryUtil.estimate_rigid_transform(p, q)

        # Otherwise, return None.
        else:
            return None

    @staticmethod
    def estimate_space_to_space_transform(marker_positions_s: Dict[str, np.ndarray],
                                          marker_positions_t: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        Try to calculate the transformation from a source space S to a target space T by making use of the
        known positions of the corners of a square marker (e.g. an ArUco marker) in both spaces.

        :param marker_positions_s:  The positions of the marker's corners in S (where known).
        :param marker_positions_t:  The positions of the marker's corners in T (where known).
        :return:                    The transformation from S to T, if possible, or None otherwise.
        """
        # If the positions in both S and T of all of the marker's corners are known, estimate the transformation.
        if all(key in marker_positions_s and key in marker_positions_t for key in ["0_0", "0_1", "0_2", "0_3"]):
            p: np.ndarray = np.column_stack([
                marker_positions_s["0_0"],
                marker_positions_s["0_1"],
                marker_positions_s["0_2"],
                marker_positions_s["0_3"]
            ])

            q: np.ndarray = np.column_stack([
                marker_positions_t["0_0"],
                marker_positions_t["0_1"],
                marker_positions_t["0_2"],
                marker_positions_t["0_3"]
            ])

            return GeometryUtil.estimate_rigid_transform(p, q)

        # Otherwise, return None.
        else:
            return None
