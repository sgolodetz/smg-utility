import cv2
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

    @staticmethod
    def load_depth_image(filename: str, *, depth_scale_factor: float = 1000.0) -> np.ndarray:
        """
        Load a depth image from disk.

        :param filename:            The name of the file from which to load it.
        :param depth_scale_factor:  The factor by which the depths were scaled when they were saved.
        :return:                    The loaded depth image.
        """
        return cv2.imread(filename, cv2.IMREAD_UNCHANGED) / depth_scale_factor

    @staticmethod
    def save_depth_image(filename: str, depth_image: np.ndarray, *, depth_scale_factor: float = 1000) -> None:
        """
        Save a depth image to disk.

        :param filename:            The name of the file to which to save it.
        :param depth_image:         The depth image to save.
        :param depth_scale_factor:  The factor by which to scale the depths before saving them.
        """
        scaled_depth_image: np.ndarray = (depth_image * depth_scale_factor).astype(np.uint16)
        cv2.imwrite(filename, scaled_depth_image)
