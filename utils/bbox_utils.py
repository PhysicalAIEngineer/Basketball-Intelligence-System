"""
Utility functions for bounding box calculations and measurements.
"""


def get_center_of_bbox(bbox):
    """
    Calculate the center coordinates of a bounding box.
    Args:
        bbox: Bounding box in format (x1, y1, x2, y2).
    Returns:
        Tuple containing center (x, y).
    """

    # top-left (x1, y1) and bottom-right (x2, y2) coordinates from the bbox
    x1, y1, x2, y2 = bbox

    # Computes the average x and y values, casts them to integers, and returns the center coordinate tuple (center_x, center_y).
    return (int((x1 + x2) / 2),int((y1 + y2) / 2))

def get_bbox_width(bbox):
    """
    Calculate the width of a bounding box.
    """
    # Subtracts the top-left x coordinate (x1 at index 0) from the bottom-right x coordinate (x2 at index 2).
    return bbox[2] - bbox[0]


def measure_distance(p1, p2):
    """
    Calculate Euclidean distance between two points.
    """

    # Implements the standard 2D Euclidean distance formula: sqrt((x1 - x2)^2 + (y1 - y2)^2).
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def measure_xy_distance(p1, p2):
    """
    Calculate x and y distance between two points.
    """
    # Returns a tuple containing signed horizontal (dx) and vertical (dy) differences between point 1 and point 2.
    return (p1[0] - p2[0], p1[1] - p2[1])

def get_foot_position(bbox):
    """
    Calculate the bottom-center point of a bounding box.
    """

    # Unpacks the bounding box coordinates (x1, y1, x2, y2).
    x1, y1, x2, y2 = bbox

    # Calculates the horizontal midpoint (center x) and pairs it with the bottom edge (y2) to locate the foot/base point.
    return (int((x1 + x2) / 2), int(y2))