import numpy as np

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class MonocularDepthEstimator(ABC):
    """The interface for a monocular depth estimator."""

    # PUBLIC ABSTRACT METHODS

    @abstractmethod
    def estimate_depth(self, colour_image: np.ndarray, tracker_w_t_c: np.ndarray, *, postprocess: bool = False) \
            -> Optional[np.ndarray]:
        """
        Try to estimate a depth image corresponding to the colour image passed in.

        :param colour_image:    The colour image.
        :param tracker_w_t_c:   The camera pose corresponding to the colour image (as a camera -> world transform).
        :param postprocess:     Whether or not to apply any optional post-processing to the depth image.
        :return:                The estimated depth image, if possible, or None otherwise.
        """
        pass

    @abstractmethod
    def set_intrinsics(self, intrinsics: np.ndarray) -> "MonocularDepthEstimator":
        """
        Set the camera intrinsics.

        :param intrinsics:  The 3x3 camera intrinsics matrix.
        :return:            The current object.
        """
        pass

    # PUBLIC METHODS

    def get_keyframes(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get the current set of keyframes.

        :return:    The current set of keyframes.
        """
        # FIXME: Only some monocular depth estimators are currently able to return their keyframes.
        return []
