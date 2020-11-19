import numpy as np


class ImageUtil:
    """Utility functions related to images."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def flip_channels(img: np.ndarray) -> np.ndarray:
        """
        Convert a BGR image to RGB, or vice-versa.

        :param img: The input image.
        :return:    The output image.
        """
        return np.ascontiguousarray(img[:, :, [2, 1, 0]])
