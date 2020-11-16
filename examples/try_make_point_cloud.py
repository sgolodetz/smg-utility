import cv2
import numpy as np
import open3d as o3d

from typing import Optional, Tuple

from smg.geometry import GeometryUtil
from smg.openni.openni_camera import OpenNICamera


def add_axis(vis: o3d.visualization.Visualizer, pose: np.ndarray, *,
             colour: Optional[Tuple[float, float, float]] = None, size: float = 1.0) -> None:
    """
    Add to the specified Open3D visualisation a set of axes for the specified pose.

    :param vis:     The Open3D visualisation.
    :param pose:    The pose (specified in camera space).
    :param colour:  An optional colour with which to paint the axes.
    :param size:    The size to give the axes (defaults to 1).
    """
    # noinspection PyArgumentList
    axes: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    if colour is not None:
        axes.paint_uniform_color(colour)
    axes.transform(pose)
    # noinspection PyTypeChecker
    vis.add_geometry(axes)


def main():
    with OpenNICamera(mirror_images=True) as camera:
        intrinsics: Tuple[float, float, float, float] = camera.get_colour_intrinsics()
        while True:
            colour_image, depth_image = camera.get_images()
            cv2.imshow("Image", colour_image)
            c: int = cv2.waitKey(1)
            if c == ord('b'):
                cv2.destroyAllWindows()
                break

        depth_mask: np.ndarray = np.where(depth_image != 0, 255, 0).astype(np.uint8)
        pcd_points, pcd_colours = GeometryUtil.make_point_cloud(colour_image, depth_image, depth_mask, intrinsics)

        # Convert the point cloud to Open3D format.
        pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_points)
        pcd.colors = o3d.utility.Vector3dVector(pcd_colours)

        # Set up the visualisation.
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        render_option: o3d.visualization.RenderOption = vis.get_render_option()
        render_option.line_width = 10

        # noinspection PyTypeChecker
        vis.add_geometry(pcd)
        add_axis(vis, np.eye(4), size=0.1)

        # Set the initial pose for the visualiser.
        params = vis.get_view_control().convert_to_pinhole_camera_parameters()
        m = np.eye(4)
        params.extrinsic = m
        vis.get_view_control().convert_from_pinhole_camera_parameters(params)

        # Run the visualiser.
        vis.run()


if __name__ == "__main__":
    main()
