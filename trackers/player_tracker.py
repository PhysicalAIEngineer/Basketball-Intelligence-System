# Imports the YOLO object detection model class from the Ultralytics library.
from ultralytics import YOLO

# Imports Roboflow's Supervision utility library for tracking algorithms (ByteTrack)
import supervision as sv

# Imports the caching utility functions to load/save
from utils import read_stub, save_stub

class PlayerTracker:
    """
    A class that handles player detection and tracking using YOLO and ByteTrack.
    this class combines YOLO object detection with ByteTrack tracking to maintain consistent
    player identities across frames while processing detections in batches.
    """

    def __init__(self, model_path):
        """
        Initialize the PlayerTracker with YOLO model and ByteTrack tracker.
        Args:
            model_path (str): Path to the YOLO model weights.
        """

        # Loads the trained YOLO model weights into memory using the provided file path.
        self.model = YOLO(model_path)
        # Initializes Supervision's ByteTrack algorithm to assign and persist unique IDs across video frames.
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        """
        Detect players in a sequence of frames using batch processing.
        Args:
            frames (list): List of video frames to process.
        Returns:
            list: YOLO detection results for each frame.
        """

        # Sets the maximum number of frames
        batch_size = 20

        # Initializes an empty list to accumulate detection results from all batches.
        detections = []

        # Iterates through the list of frames in steps defined by 'batch_size'.
        for i in range(0, len(frames), batch_size):
            # Runs YOLO inference on the current batch slice with a 50% confidence threshold filter.
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.5)
            # Appends the batch prediction results to the master detections list.
            detections += detections_batch
        # Returns the list containing raw YOLO prediction objects for every frame in the video.
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Get player tracking results for a sequence of frames with optional caching.
        Args:
            frames (list): List of video frames to process.
            read_from_stub (bool): Whether to attempt reading cached results.
            stub_path (str): Path to the cache file.

        Returns:
            list: List of dictionaries containing player tracking information for each frame,
                where each dictionary maps player IDs to their bounding box coordinates.
        """

        # Attempts to load cached tracking data
        tracks = read_stub(read_from_stub, stub_path)

        # Checks if cache reading was successful.
        if tracks is not None:
            # Verifies the cached data matches the input video length before returning it to skip heavy inference.
            if len(tracks) == len(frames):
                return tracks

        # Runs batch inference to generate predictions across all input frames if no valid cache is found.
        detections = self.detect_frames(frames)

        # Initializes the empty tracking structure for storing frame-by-frame player metadata.
        tracks = []

        # Loops through the YOLO detection output frame by frame along with the frame index.
        for frame_num, detection in enumerate(detections):
            # Retrieves the class index-to-name mapping dictionary
            cls_names = detection.names
            # Inverts the dictionary to allow lookup of class index by name string (e.g., {'Player': 0}).
            cls_names_inv = {v: k for k, v in cls_names.items()}
            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            # Updates ByteTrack state with new detections, updating existing trajectories and assigning track IDs.
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            # Creates an empty dictionary for the current frame index inside the master list.
            tracks.append({})

            # Iterates through every tracked object detected in the current frame.
            for frame_detection in detection_with_tracks:
                # Extracts the bounding box coordinates tuple `[x1, y1, x2, y2]` as a standard Python list.
                bbox = frame_detection[0].tolist()
                # Extracts the predicted class ID integer for the object.
                cls_id = frame_detection[3]
                # Extracts the unique ByteTrack tracking ID integer assigned to this object.
                track_id = frame_detection[4]
                # Filters detections so only objects labeled as 'Player' are saved.
                if cls_id == cls_names_inv['Player']:
                    # Stores the player's bounding box coordinates in the frame's dictionary mapped by their unique track ID.
                    tracks[frame_num][track_id] = {"bbox": bbox}

        # saves the populated tracking dictionary to disk for future fast loads.
        save_stub(stub_path, tracks)

        # Returns the final tracking data list containing player bounding boxes mapped by ID for every frame.
        return tracks