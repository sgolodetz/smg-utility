import json
import os

from typing import Any, Dict, Optional, Tuple


class CameraParameters:
    """A set of camera parameters for one or more cameras."""

    # CONSTRUCTOR

    def __init__(self):
        """Construct a set of camera parameters."""
        self.__data = {}  # type: Dict[str, Dict[str, Any]]

    # PUBLIC STATIC METHODS

    # noinspection PyUnresolvedReferences
    @staticmethod
    def try_load(filename: str) -> Optional["CameraParameters"]:
        """
        Try to load a set of camera parameters from a JSON file.

        :param filename:    The name of the file.
        :return:            The loaded camera parameters, if loading was successful, or None otherwise.
        """
        return CameraParameters().__try_load(filename)

    # PUBLIC METHODS

    def delete(self, camera_name: str) -> bool:
        """
        Delete the parameters for the camera with the specified name (if it exists).

        :param camera_name: The name of a camera.
        :return:            True, if the camera existed, or False otherwise.
        """
        if self.__data.get(camera_name) is not None:
            del self.__data[camera_name]
            return True
        else:
            return False

    def get_image_size(self, camera_name: str) -> Tuple[int, int]:
        """
        Get the size of the images produced by the camera with the specified name.

        :param camera_name: The name of a camera.
        :return:            The size of the images produced by the camera, as a (width, height) tuple.
        """
        camera_data = self.__data.get(camera_name)  # type: Optional[Dict[str, Any]]
        if camera_data is not None:
            return camera_data["width"], camera_data["height"]
        else:
            raise RuntimeError("Cannot get image size for unknown camera '{}'".format(camera_name))

    def get_intrinsics(self, camera_name: str) -> Tuple[float, float, float, float]:
        """
        Get the intrinsics of the camera with the specified name.

        :param camera_name: The name of a camera.
        :return:            The intrinsics of the camera.
        """
        camera_data = self.__data.get(camera_name)  # type: Optional[Dict[str, Any]]
        if camera_data is not None:
            return camera_data["fx"], camera_data["fy"], camera_data["cx"], camera_data["cy"]
        else:
            raise RuntimeError("Cannot get intrinsics for unknown camera '{}'".format(camera_name))

    def save(self, filename: str) -> None:
        """
        Save the camera parameters to a JSON file.

        :param filename:    The name of the file.
        """
        with open(filename, "w") as f:
            json.dump(self.__data, f, indent=4)

    def set(self, camera_name: str, width: int, height: int, fx: float, fy: float, cx: float, cy: float) -> None:
        """
        Set the parameters of the camera with the specified name.

        :param camera_name: The name of the camera.
        :param width:       The width of the camera's images.
        :param height:      The height of the camera's images.
        :param fx:          The horizontal focal length of the camera.
        :param fy:          The vertical focal length of the camera.
        :param cx:          The x component of the camera's principal point.
        :param cy:          The y component of the camera's principal point.
        """
        self.__data[camera_name] = {"width": width, "height": height, "fx": fx, "fy": fy, "cx": cx, "cy": cy}

    # PRIVATE METHODS

    # noinspection PyUnresolvedReferences
    def __try_load(self, filename: str) -> Optional["CameraParameters"]:
        """
        Try to load a set of camera parameters from a JSON file.

        :param filename:    The name of the file.
        :return:            The current object, if loading was successful, or None otherwise.
        """
        if os.path.exists(filename):
            with open(filename, "r") as f:
                self.__data = json.load(f)

            return self
        else:
            return None
