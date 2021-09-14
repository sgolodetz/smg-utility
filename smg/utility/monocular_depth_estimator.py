import numpy as np

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class MonocularDepthEstimator(ABC):
    """TODO"""

    # PUBLIC ABSTRACT METHODS

    @abstractmethod
    def estimate_depth(self, colour_image: np.ndarray, tracker_w_t_c: np.ndarray) -> Optional[np.ndarray]:
        """
        Try to estimate a depth image corresponding to the colour image passed in.

        :param colour_image:    The colour image.
        :param tracker_w_t_c:   The camera pose corresponding to the colour image (as a camera -> world transform).
        :return:                The estimated depth image, if possible, or None otherwise.
        """
        pass

    @abstractmethod
    def postprocess_depth_image(self, depth_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Try to post-process the specified depth image to reduce the amount of noise it contains.

        .. note::
            This function will return None if the input depth image does not have depth values for enough pixels.

        :param depth_image: The input depth image.
        :return:            The post-processed depth image, if possible, or None otherwise.
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
