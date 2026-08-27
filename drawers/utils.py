"""
A utility module providing functions for drawing shapes on video frames.
this module includes functions to draw triangles and ellipses on frames, which can be used
to represent various annotations such as player positions or ball locations in sports analysis.
"""

# Imports OpenCV for computer vision and graphics drawing operations.
import cv2

# Imports NumPy for array manipulations and contour point structures.
import numpy as np

# Imports sys module to adjust module search paths dynamically.
import sys

# Adds parent directory to sys.path so helper modules can be imported.
sys.path.append('../')

# Imports bounding box utility functions for calculating centers, widths, and bottom ground positions.
from utils import get_center_of_bbox, get_bbox_width, get_foot_position


def draw_traingle(frame, bbox, color):
    """
    Draws a filled triangle on the given frame at the specified bounding box location.
    Args:
        frame (numpy.ndarray): The frame on which to draw the triangle.
        bbox (tuple): A tuple representing the bounding box (x, y, width, height).
        color (tuple): The color of the triangle in BGR format.
    Returns:
        numpy.ndarray: The frame with the triangle drawn on it.
    """

    # Extracts top y-coordinate of bounding box for triangle vertex positioning.
    y = int(bbox[1])

    # Obtains horizontal midpoint of the bounding box.
    x, _ = get_center_of_bbox(bbox)

    # Defines 2D vertex array forming an upside-down triangle pointing down at point (x, y).
    triangle_points = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20],])

    # Draws and fills the inside of the triangle contour with the specified BGR color.
    cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)

    # Draws a 2-pixel black border outline around the triangle contour for clarity.
    cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

    # Returns annotated frame array containing the drawn triangle marker.
    return frame


def draw_ellipse(frame, bbox, color, track_id=None):
    """
    Draws an ellipse and an optional rectangle with a track ID on the given frame at the specified bounding box location.
    Args:
        frame (numpy.ndarray): The frame on which to draw the ellipse.
        bbox (tuple): A tuple representing the bounding box (x, y, width, height).
        color (tuple): The color of the ellipse in BGR format.
        track_id (int, optional): The track ID to display inside a rectangle. Defaults to None.
    Returns:
        numpy.ndarray: The frame with the ellipse and optional track ID drawn on it.
    """

    # Extracts bottom y-coordinate (feet level) of bounding box.
    y2 = int(bbox[3])

    # Calculates horizontal center of the bounding box.
    x_center, _ = get_center_of_bbox(bbox)

    # Computes width of bounding box to scale the ellipse appropriately.
    width = get_bbox_width(bbox)

    # Draws an elliptical ring marker at player feet on the court ground plane.
    cv2.ellipse(frame, center=(x_center, y2), axes=(int(width), int(0.35 * width)),
                angle=0.0, startAngle=-45, endAngle=235, color=color, thickness=2, lineType=cv2.LINE_4)

    # Sets width of player ID badge rectangle.
    rectangle_width = 40

    #  Sets height of player ID badge rectangle.
    rectangle_height = 20

    # Calculates left boundary of ID badge rectangle centered horizontally.
    x1_rect = x_center - rectangle_width // 2

    # Calculates right boundary of ID badge rectangle centered horizontally.
    x2_rect = x_center + rectangle_width // 2

    # Calculates top boundary of ID badge rectangle below player feet position.
    y1_rect = (y2 - rectangle_height // 2) + 15

    # Calculates bottom boundary of ID badge rectangle below player feet position.
    y2_rect = (y2 + rectangle_height // 2) + 15

    # Executes badge rendering block if valid player tracking ID is provided.
    if track_id is not None:
        # Draws filled rectangle badge matching player/team color.
        cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED)

        # Sets initial left padding for text alignment inside badge box.
        x1_text = x1_rect + 12

        # Adjusts text offset leftward to accommodate 3-digit tracking IDs properly.
        if track_id > 99:
            x1_text -= 10

        # Renders black tracking ID text integer inside badge rectangle.
        cv2.putText(frame,f"{track_id}",(int(x1_text), int(y1_rect + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 0),2)

    # Returns frame array populated with drawn foot-ring ellipse and tracking badge.
    return frame