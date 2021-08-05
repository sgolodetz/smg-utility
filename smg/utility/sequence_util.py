import cv2
import numpy as np
import os

from typing import Any, Dict, Optional, Tuple

from .camera_parameters import CameraParameters
from .image_util import ImageUtil
from .pose_util import PoseUtil


class SequenceUtil:
    """Utility functions relating to sequences."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def make_calibration_filename(sequence_dir: str) -> str:
        """
        Make the name of the file containing the camera parameters for a sequence.

        :param sequence_dir:    The sequence directory.
        :return:                The name of the file containing the camera parameters for the sequence.
        """
        return os.path.join(sequence_dir, "calib.json")

    @staticmethod
    def save_rgbd_calibration(sequence_dir: str, colour_image_size: Tuple[int, int], depth_image_size: Tuple[int, int],
                              colour_intrinsics: Tuple[float, float, float, float],
                              depth_intrinsics: Tuple[float, float, float, float]) -> None:
        """
        Save to disk the parameters of the colour and depth cameras used to capture an RGB-D sequence.

        :param sequence_dir:        The sequence directory.
        :param colour_image_size:   The size of the images captured by the colour camera, as a (width, height) tuple.
        :param depth_image_size:    The size of the imaegs captured by the depth camera, as a (width, height) tuple.
        :param colour_intrinsics:   The intrinsics of the colour camera, as an (fx, fy, cx, cy) tuple.
        :param depth_intrinsics:    The intrinsics of the depth camera, as an (fx, fy, cx, cy) tuple.
        """
        camera_params = CameraParameters()  # type: CameraParameters
        camera_params.set("colour", *colour_image_size, *colour_intrinsics)
        camera_params.set("depth", *depth_image_size, *depth_intrinsics)
        camera_params.save(SequenceUtil.make_calibration_filename(sequence_dir))

    @staticmethod
    def save_rgbd_frame(frame_idx: int, sequence_dir: str, colour_image: np.ndarray,
                        depth_image: np.ndarray, world_from_camera: np.ndarray, *,
                        colour_intrinsics: Optional[Tuple[float, float, float, float]] = None,
                        depth_intrinsics: Optional[Tuple[float, float, float, float]] = None) -> None:
        """
        Save a frame from an RGB-D sequence to disk.

        .. note::
            The colour and depth cameras are assumed to have the same pose.
        .. note::
            The camera intrinsics are stored at a sequence level, and the most sensible way to achieve
            that is to call save_calibration, but for convenience, we also allow them to be saved (into
            the central file) via this method.

        :param frame_idx:           The frame index.
        :param sequence_dir:        The sequence directory.
        :param colour_image:        The colour image.
        :param depth_image:         The depth image.
        :param world_from_camera:   The camera pose, as a transformation from camera space to world space.
        :param colour_intrinsics:   The intrinsics of the colour camera, as an (fx, fy, cx, cy) tuple.
        :param depth_intrinsics:    The intrinsics of the depth camera, as an (fx, fy, cx, cy) tuple.
        """
        # Ensure that the sequence directory exists.
        os.makedirs(sequence_dir, exist_ok=True)

        # Save the camera intrinsics into the central file (if they've been provided).
        if colour_intrinsics is not None and depth_intrinsics is not None:
            calib_filename = SequenceUtil.make_calibration_filename(sequence_dir)   # type: str
            if not os.path.exists(calib_filename):
                colour_image_size = (colour_image.shape[1], colour_image.shape[0])  # type: Tuple[int, int]
                depth_image_size = (depth_image.shape[1], depth_image.shape[0])     # type: Tuple[int, int]
                SequenceUtil.save_rgbd_calibration(
                    sequence_dir, colour_image_size, depth_image_size, colour_intrinsics, depth_intrinsics
                )

        # Save the colour image, depth image and pose for the frame.
        colour_filename = os.path.join(sequence_dir, "frame-{:06d}.color.png".format(frame_idx))  # type: str
        depth_filename = os.path.join(sequence_dir, "frame-{:06d}.depth.png".format(frame_idx))   # type: str
        pose_filename = os.path.join(sequence_dir, "frame-{:06d}.pose.txt".format(frame_idx))     # type: str

        cv2.imwrite(colour_filename, colour_image)
        ImageUtil.save_depth_image(depth_filename, depth_image)
        PoseUtil.save_pose(pose_filename, world_from_camera)

    @staticmethod
    def try_load_calibration(sequence_dir: str) -> Optional[CameraParameters]:
        """
        Try to load the parameters of the cameras that were used to capture a sequence.

        :param sequence_dir:    The sequence directory.
        :return:                The camera parameters, if possible, or None otherwise.
        """
        return CameraParameters.try_load(SequenceUtil.make_calibration_filename(sequence_dir))

    @staticmethod
    def try_load_rgbd_frame(frame_idx: int, sequence_dir: str) -> Optional[Dict[str, Any]]:
        """
        Try to load a frame from an RGB-D sequence.

        .. note::
            The RGB-D frame is returned as a mapping from strings to pieces of data:
                "colour_image" -> np.ndarray (an 8UC3 image)
                "depth_image" -> np.ndarray (a float image)
                "world_from_camera" -> np.ndarray (a 4x4 transformation matrix)

        :param frame_idx:       The frame index.
        :param sequence_dir:    The sequence directory.
        :return:                The RGB-D frame, if possible, or None otherwise.
        """
        # Determine the names of the colour image, depth image and pose files.
        colour_filename = os.path.join(sequence_dir, "frame-{:06d}.color.png".format(frame_idx))  # type: str
        depth_filename = os.path.join(sequence_dir, "frame-{:06d}.depth.png".format(frame_idx))   # type: str
        pose_filename = os.path.join(sequence_dir, "frame-{:06d}.pose.txt".format(frame_idx))     # type: str

        # If any one of the files doesn't exist, early out.
        if not os.path.exists(colour_filename) \
                or not os.path.exists(depth_filename) \
                or not os.path.exists(pose_filename):
            return None

        # Otherwise, load and return the frame.
        return {
            "colour_image": cv2.imread(colour_filename),
            "depth_image": ImageUtil.load_depth_image(depth_filename),
            "world_from_camera": PoseUtil.load_pose(pose_filename)
        }
