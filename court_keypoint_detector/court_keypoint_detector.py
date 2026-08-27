# Imports the YOLO class from Ultralytics framework for deep learning keypoint detection.
from ultralytics import YOLO

# Imports the Supervision library for computer vision utility functions and annotation tools.
import supervision as sv

# Imports sys module to enable dynamic modification of Python runtime search paths.
import sys

# Appends parent directory to sys.path to allow importing modules from neighboring directories.
sys.path.append('../')

# Imports helper functions 'read_stub' and 'save_stub' to handle caching detection outputs to disk.
from utils import read_stub, save_stub


class CourtKeypointDetector:
    """
    The CourtKeypointDetector class uses a YOLO model to detect court keypoints in image frames.
    It also provides functionality to draw these detected keypoints on the frames.
    """

    def __init__(self, model_path):
        # Initializes detector instance by loading weights from `model_path` into a YOLO model object.
        self.model = YOLO(model_path)

    def get_court_keypoints(self, frames, read_from_stub=False, stub_path=None):
        """
        Detect court keypoints for a batch of frames using the YOLO model. If requested,
        attempts to read previously detected keypoints from a stub file before running the model.
        Args:
            frames (list of numpy.ndarray): A list of frames (images) on which to detect keypoints.
            read_from_stub (bool, optional): Indicates whether to read keypoints from a stub file
                instead of running the detection model. Defaults to False.
            stub_path (str, optional): The file path for the stub file. If None, a default path may be used.
                Defaults to None.
        Returns:
            list: A list of detected keypoints for each input frame.
        """

        # Attempts to load cached keypoints from disk using the specified stub file path.
        court_keypoints = read_stub(read_from_stub, stub_path)

        # Checks if valid stub data was successfully loaded from disk.
        if court_keypoints is not None:

            # Validates that cached keypoints match the input frame count and returns them immediately if true.
            if len(court_keypoints) == len(frames):
                return court_keypoints

        # Sets batch size to 20 frames per model prediction step to optimize inference speed and memory.
        batch_size = 20

        # List to store detected keypoints across all frames in the input sequence.
        court_keypoints = []

        # Iterates through frames in increments of `batch_size`.
        for i in range(0, len(frames), batch_size):

            # Runs batch keypoint inference on current frame chunk with a minimum confidence threshold of 0.5.
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.5)

            # Iterates through individual frame detection outputs within the current batch.
            for detection in detections_batch:
                # Extracts detected keypoint predictions for the frame and appends to results list.
                court_keypoints.append(detection.keypoints)

        # Saves the computed court keypoints to disk at `stub_path` for future cached runs.
        save_stub(stub_path, court_keypoints)

        # Returns final list containing court keypoints for each input frame.
        return court_keypoints