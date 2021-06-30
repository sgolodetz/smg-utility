import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from numba import cuda
from PIL import Image
from typing import Any, Optional

from .numba_util import NumbaUtil


# MAIN CLASS

class ImageUtil:
    """Utility functions related to images."""

    # PRIVATE STATIC CONSTANTS

    __CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    # PUBLIC STATIC METHODS

    @staticmethod
    def colourise_segmentation(segmentation: np.ndarray, *, may_load_default_palette: bool = True,
                               palette: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Make a colourised version of the specified segmentation.

        .. note::
            If a palette is passed in, it will be used. If not, but this function is allowed to load the
            default palette, that will be used. Otherwise, a colour map from pyplot wil be used.

        :param segmentation:                The segmentation to colourise.
        :param may_load_default_palette:    Whether or not this function is allowed to load the default palette
                                            if no palette is passed in.
        :param palette:                     An optional palette.
        :return:                            The colourised segmentation.
        """
        # If no palette was passed in, but we're allowed to load the default palette, load it.
        if palette is None and may_load_default_palette:
            palette = ImageUtil.load_default_palette()

        # If we now have a palette, use it to colourise the segmentation. If not, use a colour map from pyplot
        # to colourise the segmentation instead.
        if palette is not None:
            height, width = segmentation.shape[:2]
            segmentation_bgr = np.zeros((height, width, 3), dtype=np.uint8)  # type: np.ndarray
            NumbaUtil.launch_kernel_2d(
                ImageUtil.__ck_apply_palette, palette, segmentation, segmentation_bgr, grid_size=(height, width)
            )
            return segmentation_bgr
        else:
            cm = plt.get_cmap("tab20")
            return (ImageUtil.flip_channels(cm(np.mod(segmentation, 20))) * 255).astype(np.uint8)

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
    def load_default_palette() -> np.ndarray:
        """
        Load the default palette.

        :return:    The default palette.
        """
        palette_image_pil = Image.open(os.path.join(ImageUtil.__CURRENT_DIR, "palette.png"))
        return np.array(palette_image_pil.getpalette()).reshape((256, 3))

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

    # PRIVATE STATIC CUDA KERNELS

    @staticmethod
    @cuda.jit
    def __ck_apply_palette(palette, segmentation, segmentation_bgr) -> None:
        """
        Apply a palette to the specified segmentation to produce a colourised version of it.

        :param palette:             The palette.
        :param segmentation:        The segmentation.
        :param segmentation_bgr:    An output image into which to write the colourised version of the segmentation.
        """
        # noinspection PyArgumentList
        y, x = cuda.grid(2)

        if y < segmentation.shape[0] and x < segmentation.shape[1]:
            # Determine the colour from the palette to use for this pixel.
            idx = segmentation[y, x] % palette.shape[0]  # type: int

            # Write the specified colour into the relevant pixel of the output image. Note that the colours in
            # the palette are RGB, whereas our output image is BGR, so we flip the channels when doing this.
            for i in range(3):
                segmentation_bgr[y, x, i] = palette[idx, 2 - i]
