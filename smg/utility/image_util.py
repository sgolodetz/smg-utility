import cv2
import numpy as np

from typing import Any


# MAIN CLASS

class ImageUtil:
    """Utility functions related to images."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def fill_border(image: np.ndarray, border_size: int, value: Any) -> np.ndarray:
        """
        Make a copy of the input image in which a border of the size specified has been filled with the specified value.

        :param image:       The input image.
        :param border_size: The border size (in pixels).
        :param value:       The value with which to fill the border.
        :return:            The output image.
        """
        height, width = image.shape
        image_copy = image.copy()  # type: np.ndarray
        image_copy[:border_size, :] = value
        image_copy[height - border_size:, :] = value
        image_copy[:, :border_size] = value
        image_copy[:, width - border_size:] = value
        return image_copy

    @staticmethod
    def flip_channels(image: np.ndarray) -> np.ndarray:
        """
        Convert a BGR image to RGB, or vice-versa.

        :param image:   The input image.
        :return:        The output image.
        """
        return np.ascontiguousarray(image[:, :, [2, 1, 0]])

    @staticmethod
    def from_short_depth(short_depth_image: np.ndarray, *, depth_scale_factor: float = 1000) -> np.ndarray:
        """
        Convert an unsigned short depth image to a floating-point one.

        :param short_depth_image:   The unsigned short depth image.
        :param depth_scale_factor:  The factor by which to divide the depths during the conversion.
        :return:                    The floating-point depth image.
        """
        return short_depth_image / depth_scale_factor

    @staticmethod
    def load_depth_image(filename: str, *, depth_scale_factor: float = 1000.0) -> np.ndarray:
        """
        Load a depth image from disk.

        :param filename:            The name of the file from which to load it.
        :param depth_scale_factor:  The factor by which the depths were scaled when they were saved.
        :return:                    The loaded depth image.
        """
        return ImageUtil.from_short_depth(
            cv2.imread(filename, cv2.IMREAD_UNCHANGED), depth_scale_factor=depth_scale_factor
        )

    @staticmethod
    def save_depth_image(filename: str, depth_image: np.ndarray, *, depth_scale_factor: float = 1000) -> None:
        """
        Save a depth image to disk.

        :param filename:            The name of the file to which to save it.
        :param depth_image:         The depth image to save.
        :param depth_scale_factor:  The factor by which to scale the depths before saving them.
        """
        cv2.imwrite(filename, ImageUtil.to_short_depth(depth_image, depth_scale_factor=depth_scale_factor))

    @staticmethod
    def to_short_depth(depth_image: np.ndarray, *, depth_scale_factor: float = 1000) -> np.ndarray:
        """
        Convert a floating-point depth image to an unsigned short one.

        :param depth_image:         The floating-point depth image.
        :param depth_scale_factor:  The factor by which to multiply the depths during the conversion.
        :return:                    The unsigned short depth image.
        """
        return (depth_image * depth_scale_factor).astype(np.uint16)
