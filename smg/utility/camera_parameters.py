from __future__ import annotations

import json

from typing import Any, Dict, Optional, Tuple


class CameraParameters:
    """The parameters for one or more cameras."""

    # CONSTRUCTOR

    def __init__(self):
        """TODO"""
        self.__data: Dict[str, Dict[str, Any]] = {}

    # PUBLIC STATIC METHODS

    @staticmethod
    def load(filename: str) -> CameraParameters:
        """
        TODO

        :param filename:    TODO
        :return:            TODO
        """
        return CameraParameters().__load(filename)

    # PUBLIC METHODS

    def get_image_size(self, camera_name: str) -> Tuple[int, int]:
        """
        TODO

        :param camera_name: TODO
        :return:            TODO
        """
        camera_data: Optional[Dict[str, Any]] = self.__data.get(camera_name)
        if camera_data is not None:
            return camera_data["width"], camera_data["height"]
        else:
            raise RuntimeError(f"Cannot get image size for unknown camera '{camera_name}'")

    def get_intrinsics(self, camera_name: str) -> Tuple[float, float, float, float]:
        """
        TODO

        :param camera_name: TODO
        :return:            TODO
        """
        camera_data: Optional[Dict[str, Any]] = self.__data.get(camera_name)
        if camera_data is not None:
            return camera_data["fx"], camera_data["fy"], camera_data["cx"], camera_data["cy"]
        else:
            raise RuntimeError(f"Cannot get intrinsics for unknown camera '{camera_name}'")

    def save(self, filename: str) -> None:
        """
        TODO

        :param filename:    TODO
        """
        with open(filename, "w") as f:
            json.dump(self.__data, f, indent=4)

    def set(self, camera_name: str, width: int, height: int, fx: float, fy: float, cx: float, cy: float) -> None:
        """
        TODO

        :param camera_name: TODO
        :param width:       TODO
        :param height:      TODO
        :param fx:          TODO
        :param fy:          TODO
        :param cx:          TODO
        :param cy:          TODO
        """
        self.__data[camera_name] = {"width": width, "height": height, "fx": fx, "fy": fy, "cx": cx, "cy": cy}

    # PRIVATE METHODS

    def __load(self, filename: str) -> CameraParameters:
        """
        TODO

        :param filename:    TODO
        :return:            TODO
        """
        with open(filename, "r") as f:
            self.__data = json.load(f)

        return self
