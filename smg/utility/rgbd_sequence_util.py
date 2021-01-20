import cv2
import numpy as np
import os

from typing import Any, Dict, Optional, Tuple

from smg.utility import CameraParameters, ImageUtil, PoseUtil


class RGBDSequenceUtil:
    """Utility functions relating to RGB-D sequences."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def save_calibration(sequence_dir: str, colour_image_size: Tuple[int, int], depth_image_size: Tuple[int, int],
                         colour_intrinsics: Tuple[float, float, float, float],
                         depth_intrinsics: Tuple[float, float, float, float]) -> None:
        camera_params: CameraParameters = CameraParameters()
        camera_params.set("colour", *colour_image_size, *colour_intrinsics)
        camera_params.set("depth", *depth_image_size, *depth_intrinsics)
        camera_params.save(RGBDSequenceUtil.__make_calibration_filename(sequence_dir))

    @staticmethod
    def save_frame(frame_idx: int, sequence_dir: str, colour_image: np.ndarray,
                   depth_image: np.ndarray, world_from_camera: np.ndarray, *,
                   colour_intrinsics: Optional[Tuple[float, float, float, float]] = None,
                   depth_intrinsics: Optional[Tuple[float, float, float, float]] = None) -> None:
        os.makedirs(sequence_dir, exist_ok=True)

        if colour_intrinsics is not None and depth_intrinsics is not None:
            calib_filename: str = RGBDSequenceUtil.__make_calibration_filename(sequence_dir)
            if not os.path.exists(calib_filename):
                colour_image_size: Tuple[int, int] = (colour_image.shape[1], colour_image.shape[0])
                depth_image_size: Tuple[int, int] = (depth_image.shape[1], depth_image.shape[0])
                RGBDSequenceUtil.save_calibration(
                    sequence_dir, colour_image_size, depth_image_size, colour_intrinsics, depth_intrinsics
                )

        colour_filename: str = os.path.join(sequence_dir, f"frame-{frame_idx:06d}.color.png")
        depth_filename: str = os.path.join(sequence_dir, f"frame-{frame_idx:06d}.depth.png")
        pose_filename: str = os.path.join(sequence_dir, f"frame-{frame_idx:06d}.pose.txt")

        cv2.imwrite(colour_filename, colour_image)
        ImageUtil.save_depth_image(depth_filename, depth_image)
        PoseUtil.save_pose(pose_filename, world_from_camera)

    @staticmethod
    def try_load_calibration(sequence_dir: str) -> Optional[CameraParameters]:
        return CameraParameters.try_load(RGBDSequenceUtil.__make_calibration_filename(sequence_dir))

    @staticmethod
    def try_load_frame(frame_idx: int, sequence_dir: str) -> Optional[Dict[str, Any]]:
        colour_filename: str = os.path.join(sequence_dir, f"frame-{frame_idx:06d}.color.png")
        depth_filename: str = os.path.join(sequence_dir, f"frame-{frame_idx:06d}.depth.png")
        pose_filename: str = os.path.join(sequence_dir, f"frame-{frame_idx:06d}.pose.txt")

        # If the colour image doesn't exist, early out.
        if not os.path.exists(colour_filename):
            return None

        return {
            "colour_image": cv2.imread(colour_filename),
            "depth_image": ImageUtil.load_depth_image(depth_filename),
            "world_from_camera": PoseUtil.load_pose(pose_filename)
        }

    # PRIVATE STATIC METHODS

    @staticmethod
    def __make_calibration_filename(sequence_dir: str) -> str:
        return os.path.join(sequence_dir, "calib.json")
