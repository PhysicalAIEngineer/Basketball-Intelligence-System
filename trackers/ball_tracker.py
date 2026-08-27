# Imports the YOLO object detection model class from Ultralytics.
from ultralytics import YOLO

# Imports Supervision utilities for handling object detection formats.
import supervision as sv

# Imports functions to load and save intermediate tracking results
from utils import read_stub, save_stub

class BallTracker:
    """
    A class that handles basketball detection and tracking using YOLO.
    this class provides methods to detect the ball in video frames, process detections
    in batches, and refine tracking results through filtering and interpolation.
    """

    # Initializes the BallTracker instance with a specified model path.
    def __init__(self, model_path):
        # Loads the YOLO model weights trained to recognize basketballs.
        self.model = YOLO(model_path)
        # Initializes a ByteTrack tracker instance (though ball selection in this class relies primarily on highest-confidence detection per frame).
        self.trackers = sv.ByteTrack()

    # Helper function to perform batched prediction across video frames.
    def detect_frames(self, frames):
        """
        Detect the ball in a sequence of frames using batch processing.
        Args:
            frames (list): List of video frames to process.
        Returns:
            list: YOLO detection results for each frame.
        """

        # Defines the batch size
        batch_size = 20

        # Creates an empty list to accumulate detection objects across batches.
        detections = []

        # Iterates over the frames list in increments of 20.
        for i in range(0, len(frames), batch_size):
            # Runs YOLO object detection on the current batch with a 50% confidence floor.
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.5)
            # Appends batch results into the cumulative detections list.
            detections += detections_batch
        # Returns raw YOLO predictions for all video frames.
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Get ball tracking results for a sequence of frames with optional caching.
        Args:
            frames (list): List of video frames to process.
            read_from_stub (bool): Whether to attempt reading cached results.
            stub_path (str): Path to the cache file.
        Returns:
            list: List of dictionaries containing ball tracking information for each frame.
        """

        # Attempts to read cached tracking data
        tracks = read_stub(read_from_stub, stub_path)
        # Checks if valid cached data exists.
        if tracks is not None:
            # Returns cached data if frame count matches the input video.
            if len(tracks) == len(frames):
                return tracks

        # Runs batch inference across all video frames if no valid cache is available.
        detections = self.detect_frames(frames)

        # Initializes the result list for storing frame-by-frame ball tracking dictionaries.
        tracks = []

        # Loops through the YOLO detection result for each frame.
        for frame_num, detection in enumerate(detections):
            # Retrieves class mapping dictionary
            cls_names = detection.names
            # Reverses dictionary to look up class ID by name
            cls_names_inv = {v: k for k, v in cls_names.items()}
            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            # Appends an empty dictionary for the current frame to hold the ball detection.
            tracks.append({})
            # Placeholder for the bounding box with highest confidence in the current frame.
            chosen_bbox = None
            # Tracks the highest confidence score observed for a ball detection in this frame.
            max_confidence = 0

            # Iterates through all detected objects in the current frame.
            for frame_detection in detection_supervision:
                # Converts bounding box coordinates to list `[x1, y1, x2, y2]`.
                bbox = frame_detection[0].tolist()
                # Extracts the class ID for the current detection.
                cls_id = frame_detection[3]
                # Extracts the confidence score for the current detection.
                confidence = frame_detection[2]
                # Filters for detections that match the 'Ball' class label.
                if cls_id == cls_names_inv['Ball']:
                    # Selects the ball bounding box with the highest confidence score in case multiple exist.
                    if max_confidence < confidence:
                        chosen_bbox = bbox
                        max_confidence = confidence
            # Assigns the best ball bounding box to ID key 1 for the current frame if a ball was found.
            if chosen_bbox is not None:
                tracks[frame_num][1] = {"bbox": chosen_bbox}

        # Caches the generated tracking
        save_stub(stub_path, tracks)

        # Returns raw ball track results containing frame-by-frame bounding boxes.
        return tracks

    def remove_wrong_detections(self, ball_positions):
        """
        Filter out incorrect ball detections based on maximum allowed movement distance.
        Args:
            ball_positions (list): List of detected ball positions across frames.
        Returns:
            list: Filtered ball positions with incorrect detections removed.
        """

        # Sets the maximum allowed pixel jump between consecutive frames.
        maximum_allowed_distance = 25

        # Tracks the frame index of the most recent confirmed valid ball detection.
        last_good_frame_index = -1

        # Loops through each frame's ball tracking dictionary.
        for i in range(len(ball_positions)):

            # Extracts current frame ball bounding box coordinates, returning an empty list if none exist.
            current_box = ball_positions[i].get(1, {}).get('bbox', [])

            # Skips frames where no ball was detected.
            if len(current_box) == 0:
                continue

            # Initializes the baseline valid detection frame index.
            if last_good_frame_index == -1:
                # First valid detection
                last_good_frame_index = i
                continue

            #Retrieves the bounding box of the last confirmed valid detection.
            last_good_box = ball_positions[last_good_frame_index].get(1, {}).get('bbox', [])
            # Calculates the number of elapsed frames since the last valid detection.
            frame_gap = i - last_good_frame_index
            # Scales the maximum distance threshold dynamically based on the frame gap.
            adjusted_max_distance = maximum_allowed_distance * frame_gap

            # Calculates Euclidean distance between top-left coordinates; if motion exceeds the threshold, discards it as a false positive.
            else:
            if np.linalg.norm(np.array(last_good_box[:2]) - np.array(current_box[:2])) > adjusted_max_distance:
                ball_positions[i] = {}
            # If motion is within reasonable physical limits, marks current frame as the new last valid detection.
            else:
                last_good_frame_index = i

        # Returns the filtered ball trajectory with outlier detections removed.
        return ball_positions

    def interpolate_ball_positions(self, ball_positions):
        """
        Interpolate missing ball positions to create smooth tracking results.
        Args:
            ball_positions (list): List of ball positions with potential gaps.
        Returns:
            list: List of ball positions with interpolated values filling the gaps.
        """

        # Extracts raw bounding box lists into a flat list, inserting empty lists `[]` for missing frames.
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        # Converts the trajectory into a Pandas DataFrame with 4 spatial coordinate columns.
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Performs linear interpolation to fill NaN/missing entries between valid detection points.
        df_ball_positions = df_ball_positions.interpolate()
        # Backward-fills any missing values at the very start of the video sequence.
        df_ball_positions = df_ball_positions.bfill()

        # Reconstructs the original dictionary structure mapping ball track ID 1 to its interpolated bounding box.
        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        # Returns a continuous, smooth ball trajectory spanning all frames.
        return ball_positions