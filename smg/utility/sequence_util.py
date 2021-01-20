import cv2
import numpy as np
import os

from typing import Any, Dict, Optional, Tuple

from smg.utility import CameraParameters, ImageUtil, PoseUtil


class SequenceUtil:
    """Utility functions relating to sequences."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def load_frame() -> Dict[str, Any]:
        pass

    @staticmethod
    def save_calibration(output_dir: Optional[str], colour_image_size: Tuple[int, int],
                         depth_image_size: Tuple[int, int], colour_intrinsics: Tuple[float, float, float, float],
                         depth_intrinsics: Tuple[float, float, float, float]) -> bool:
        if output_dir is None:
            return False

        camera_params: CameraParameters = CameraParameters()
        camera_params.set("colour", *colour_image_size, *colour_intrinsics)
        camera_params.set("depth", *depth_image_size, *depth_intrinsics)
        camera_params.save(SequenceUtil.__make_calibration_filename(output_dir))

        return True

    @staticmethod
    def save_frame(frame_idx: int, output_dir: Optional[str], colour_image: np.ndarray, depth_image: np.ndarray,
                   pose_w_t_c: np.ndarray, *, colour_intrinsics: Optional[Tuple[float, float, float, float]] = None,
                   depth_intrinsics: Optional[Tuple[float, float, float, float]] = None) -> bool:
        if output_dir is None:
            return False

        os.makedirs(output_dir, exist_ok=True)

        if colour_intrinsics is not None and depth_intrinsics is not None:
            calib_filename: str = SequenceUtil.__make_calibration_filename(output_dir)
            if not os.path.exists(calib_filename):
                colour_image_size: Tuple[int, int] = (colour_image.shape[1], colour_image.shape[0])
                depth_image_size: Tuple[int, int] = (depth_image.shape[1], depth_image.shape[0])
                SequenceUtil.save_calibration(
                    output_dir, colour_image_size, depth_image_size, colour_intrinsics, depth_intrinsics
                )

        colour_filename: str = os.path.join(output_dir, f"frame-{frame_idx:06d}.color.png")
        depth_filename: str = os.path.join(output_dir, f"frame-{frame_idx:06d}.depth.png")
        pose_filename: str = os.path.join(output_dir, f"frame-{frame_idx:06d}.pose.txt")

        cv2.imwrite(colour_filename, colour_image)
        ImageUtil.save_depth_image(depth_filename, depth_image)
        PoseUtil.save_pose(pose_filename, pose_w_t_c)

        return True

    # PRIVATE STATIC METHODS

    @staticmethod
    def __make_calibration_filename(output_dir: str) -> str:
        return os.path.join(output_dir, "calib.json")
