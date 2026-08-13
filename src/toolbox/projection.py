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


    def project_point(self, point) -> np.ndarray | None:
        """
        Project a 3D position to 2D pixel coordinates. 
        Point already in camera frame.
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
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            self.camera_matrix,
            self.distortion,
        )

        pixel_x, pixel_y = image_points.reshape(2)
        if not np.isfinite([pixel_x, pixel_y]).all():
            return None

        pixel = np.asarray([int(pixel_x), int(pixel_y)])
        return pixel

    def project_points(self, points) -> np.ndarray | None:
        """
        Project more than 1 point from 3D to 2D pixel coordinates.
        Accepts an Nx3 array of points, returns an Nx2 array of pixel coordinates.
        """
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected an Nx3 array of points; got shape {points.shape}.")

        pixel_points =[]
        for point in points:
            pixel = self.project_point(point)
            if pixel is not None:
                pixel_points.append(pixel)

        if not pixel_points:
            return None

        return np.array(pixel_points)

    
    def project_line(self, line_points):
        """
        Project a 3D line to 2D pixel coordinates, with camera intrinsics and distortion coeffs.
        line_points: 2x3 array of START and END 3D points defining the line
        TS can literally be replaced by just 2 calls of project point
        """

        line_points = np.asarray(line_points, dtype=np.float32).reshape(2, 3)
        if not np.isfinite(line_points).all():
            return None

        # use projectPoints
        image_points, _ = cv2.projectPoints(
            line_points.reshape(1, 2, 3),
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            self.camera_matrix,
            self.distortion,
        )

        pixel_line = image_points.reshape(2, 2).astype(int)
        if not np.isfinite(pixel_line).all():
            return None

        return pixel_line


    def draw_point(self, frame: np.ndarray, pixel: np.ndarray, color=(0, 255, 0), radius=5) -> np.ndarray:
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


    def draw_coordinate_axis(self, frame, position: np.ndarray, axis_length=10.0):
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

        # project points to 2D
        origin_pixel = self.project_point(origin)
        x_pixel = self.project_point(x_axis)            
        y_pixel = self.project_point(y_axis)
        z_pixel = self.project_point(z_axis)

        if origin_pixel is None:
            raise ValueError(f"Origin point {origin} is not projectable to image.")
        
        if x_pixel is None or y_pixel is None or z_pixel is None:
            raise ValueError(f"One of the axis points is not projectable to image: x={x_axis}, y={y_axis}, z={z_axis}.")

        cv2.line(frame, tuple(map(int, origin_pixel)), tuple(map(int, x_pixel)), AXIS_COLORS[0], 2)
        cv2.line(frame, tuple(map(int, origin_pixel)), tuple(map(int, y_pixel)), AXIS_COLORS[1], 2)
        cv2.line(frame, tuple(map(int, origin_pixel)), tuple(map(int, z_pixel)), AXIS_COLORS[2], 2)

        return frame



    def draw_text(self,frame, text:str, position: tuple, font_scale=1.0, color=(0, 255, 0), thickness=2) -> np.ndarray:
        """
        Writes text over frame in green unless otherwise specified
        """
        
        x = int(position[0])
        y = int(position[1])
        
        cv2.putText(frame, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, lineType=cv2.LINE_AA)

        return frame
        