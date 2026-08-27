# Imports NumPy for numerical matrix operations and coordinate array manipulations.
import numpy as np

# Imports OpenCV for computing homography transformation matrices and projecting 2D spatial points.
import cv2

# Defines a class to encapsulate homography matrix computation and point transformations between coordinate planes.
class Homography:
    # Constructor taking corresponding 2D source (e.g., video frame pixels) and target (e.g., 2D court model) reference points.
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        # Validates that source and target array dimensions match exactly.
        if source.shape != target.shape:
            raise ValueError("Source and target must have the same shape.")

        # Ensures input point matrices contain 2D coordinates (x, y).
        if source.shape[1] != 2:
            raise ValueError("Source and target points must be 2D coordinates.")

        # Converts source point coordinates to 32-bit floating-point format required by OpenCV functions.
        source = source.astype(np.float32)

        # Converts target point coordinates to 32-bit floating-point format required by OpenCV functions.
        target = target.astype(np.float32)

        # Computes the 3x3 homography transformation matrix 'm' that maps source points to target points.
        self.m, _ = cv2.findHomography(source, target)

        # Raises an error if OpenCV fails to compute a valid transformation matrix (e.g., degenerate or collinear points).
        if self.m is None:
            raise ValueError("Homography matrix could not be calculated.")

    # Transforms an array of 2D coordinates from source space to target space using the computed homography matrix.
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        # Early exit returning input as-is if the points array is empty.
        if points.size == 0:
            return points

        # Validates that the input points array contains 2D (x, y) coordinates.
        if points.shape[1] != 2:
            raise ValueError("Points must be 2D coordinates.")

        # Reshapes points array into `(N, 1, 2)` shape with float32 type required by `cv2.perspectiveTransform`.
        points = points.reshape(-1, 1, 2).astype(np.float32)

        # Applies perspective matrix transformation to project the 2D coordinates into target space.
        points = cv2.perspectiveTransform(points, self.m)

        # Reshapes transformed points back into a standard `(N, 2)` float32 array and returns them.
        return points.reshape(-1, 2).astype(np.float32)