"""
Toolbox for projecting points and drawing on images
"""

from copy import Error
import cv2

import numpy as np

class Projector:
    def __init__(self, camera_matrix: np.ndarray, distortion: np.ndarray):
        self.camera_matrix = camera_matrix
        self.distortion = distortion


    def project_point(self, point) -> np.ndarray:
        """
        Project a 3D position to 2D pixel coordinates, with camera intrinsics and distortion coeffs.
        """

        position = np.asarray(point, dtype=float).reshape(3)
        if not np.isfinite(position).all():
            return None

        if position[2] <= 0:
            Error(f"Point {position} is behind the camera (z <= 0).")
            return None

        # use projectPoints
        image_points, _ = cv2.projectPoints(
            position.reshape(1, 1, 3),
            np.zeros(3, dtype=np.float),
            np.zeros(3, dtype=np.float),
            self.camera_matrix,
            self.distortion,
        )

        pixel_x, pixel_y = image_points.reshape(2)
        if not np.isfinite([pixel_x, pixel_y]).all():
            return None

        pixel = np.asarray([int(pixel_x), int(pixel_y)])
        return pixel

            
    def project_line(line_points, camera_matrix, distortion) -> np.ndarray:
        """
        Project a 3D line to 2D pixel coordinates, with camera intrinsics and distortion coeffs.
        line_points: 2x3 array of 3D points defining the line
        """

        line_points = np.asarray(line_points, dtype=float).reshape(2, 3)
        if not np.isfinite(line_points).all():
            return None

        # use projectPoints
        image_points, _ = cv2.projectPoints(
            line_points.reshape(1, 2, 3),
            np.zeros(3, dtype=np.float),
            np.zeros(3, dtype=np.float),
            camera_matrix,
            distortion,
        )

        pixel_line = image_points.reshape(2, 2).astype(int)
        if not np.isfinite(pixel_line).all():
            return None

        return pixel_line


    def draw_point(frame: np.ndarray, pixel: np.ndarray, color=(0, 255, 0), radius=5) -> np.ndarray:
        """
        Takes a point in 3D space and projects it onto 2D image frame.
        Returns a frame with with point drawn.
        """
        width, height = frame.shape[1], frame.shape[0]

        if pixel[0] > width or pixel[1] > height or pixel[0] < 0 or pixel[1] < 0:
            raise ValueError(f"Pixel {pixel} is out of bounds for image of size {width}x{height}.")
        
        frame = cv2.circle(frame, center=(pixel[0], pixel[1]), radius=radius, color=color, thickness=-1
                        )
        return frame


    def draw_coordinate_axis(frame, position: np.ndarray, axis_length=10.0):
        """
        POSITION needs to be already in camera coordinates. 
        Draws coordinate axis on the frame, with origin at end effector position.
        XYZ -> RGB
        """

        AXIS_COLORS = ((0, 0, 255), (0, 255, 0), (255, 0, 0))  # BGR

        origin = position[:3,3] # translation
        rotation = position[:3,:3] # rotation matrix
        print(origin)
        print(rotation)

        x_axis = origin + rotation[:, 0] * axis_length
        y_axis = origin + rotation[:, 1] * axis_length
        z_axis = origin + rotation[:, 2] * axis_length

        

        




    def draw_text(frame, text:str, position: tuple, font_scale=1.0, color=(0, 255, 0), thickness=2) -> np.ndarray:
        """
        Writes text over frame in green unless otherwise specified
        """
        
        x = int(position[0])
        y = int(position[1])
        cv2.putText(frame, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, lineType=cv2.LINE_AA)

        return frame
        
    def project_plane(frame, transform: np.ndarray):

        return