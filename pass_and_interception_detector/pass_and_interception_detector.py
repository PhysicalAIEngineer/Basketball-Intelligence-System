# Imports Python's built-in math module for geometric and square root calculations.
import math

# Defines a class to analyze ball possession changes and classify them as passes or interceptions.
class PassAndInterceptionDetector:
    # Initializes thresholds for valid distances and minimum frame durations between possession changes.
    def __init__(self, min_pass_distance=20, max_pass_distance=300, min_pass_frames=2):

        # Stores the minimum pixel distance required to count a valid pass.
        self.min_pass_distance = min_pass_distance

        # Stores the maximum distance allowed between a player and ball to consider possession active.
        self.max_pass_distance = max_pass_distance

        # Stores the minimum frame count threshold for validating pass duration.
        self.min_pass_frames = min_pass_frames

    # CENTER OF BBOX
    @staticmethod
    # Helper method to calculate the (x, y) center point of a bounding box [x1, y1, x2, y2].
    def get_center(bbox):

        # Returns None if no bounding box coordinates are provided.
        if bbox is None:
            return None

        # Unpacks top-left (x1, y1) and bottom-right (x2, y2) corners of the bounding box.
        x1, y1, x2, y2 = bbox

        # Returns the calculated midpoint (center x, center y) tuple.
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    # DISTANCE
    @staticmethod
    # Calculates Euclidean distance between two 2D points p1 and p2.
    def distance(p1, p2):

        # Returns infinity if either point is missing or invalid.
        if p1 is None or p2 is None:
            return float("inf")

        # Returns Euclidean distance calculated via Pythagorean theorem.
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    # EXTRACT PLAYER ID FROM POSSESSION
    @staticmethod
    # Normalizes different data formats to extract an integer player ID from possession records.
    def extract_player_id(possession):

        # Returns None if possession record is empty or missing.
        if possession is None:
            return None

        # Case 1: possession is already an integer
        # Checks if possession is directly passed as a number.
        if isinstance(possession,(int, float)):
            return int(possession)

        # Case 2: possession is a dictionary
        # Checks if possession is formatted as a dictionary structure.
        if isinstance(possession, dict):

            # Iterates through common key naming conventions for player IDs.
            for key in ["player_id", "player", "holder", "track_id"]:
                # Checks if the current key exists in the dictionary.
                if key in possession:
                    # Retrieves the value associated with the key.
                    value = possession[key]
                    # Validates that the key value is numeric.
                    if isinstance(value, (int, float)):
                        # Returns the validated player ID integer.
                        return int(value)

            # Checks if dictionary contains a single key representing the player ID.
            if len(possession) == 1:
                # Retrieves the single top-level key.
                key = next(iter(possession))
                # Validates that the dictionary key itself is numeric.
                if isinstance(key, (int, float)):
                    # Returns the key integer as player ID.
                    return int(key)

        # Returns None if no valid player ID could be parsed from the structure.
        return None

    # FIND HOLDER FROM BALL POSITION
    # Geometric fallback that finds the player closest to the ball in a specific frame.
    def find_ball_holder(self, frame_num, player_tracks, ball_tracks):
        # Returns None for negative frame indices.
        if frame_num < 0:
            return None

        # Returns None if frame index exceeds player tracking data bounds.
        if frame_num >= len(player_tracks):
            return None

        # Returns None if frame index exceeds ball tracking data bounds.
        if frame_num >= len(ball_tracks):
            return None

        # Extracts player detections for the requested frame.
        players = player_tracks[frame_num]

        # Extracts ball detections for the requested frame.
        ball_data = ball_tracks[frame_num]

        # Returns None if no players were detected in this frame.
        if not players:
            return None

        # Returns None if the ball was not detected in this frame.
        if not ball_data:
            return None

        # Extract ball bbox
        # Variable to store the ball's bounding box.
        ball_bbox = None

        # Checks if ball tracking output is structured as a dictionary.
        if isinstance(ball_data, dict):

            # Iterates through active ball tracking records.
            for _, track in ball_data.items():
                if isinstance(track, dict):
                    if "bbox" in track:
                        ball_bbox = track["bbox"]
                        # Stops searching after locating the primary ball track bbox.
                        break

        # If ball bbox unavailable
        if ball_bbox is None:
            # Returns None if ball bounding box coordinates were not found.
            return None

        # Calculates center coordinates of the ball.
        ball_center = (self.get_center(ball_bbox))

        # Find nearest player
        # track player ID closest to the ball.
        nearest_player = None
        # Stores minimum distance found, initialized to infinity.
        nearest_distance = float("inf")

        # Loops through all detected players in current frame.
        for player_id, player_data in players.items():
            if not isinstance(player_data, dict):
                # Skips malformed player records.
                continue
            if "bbox" not in player_data:
                # Skips player records missing bounding box data.
                continue

            # Calculates center coordinates of player bounding box.
            player_center = (self.get_center(player_data["bbox"]))

            # Measures Euclidean distance between player center and ball center.
            distance = self.distance(ball_center, player_center)

            # Updates shortest distance found & nearest player ID.
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_player = player_id

        # Reject if ball is too far away
        if nearest_distance > self.max_pass_distance:
            # Rejects holder assignment if nearest player exceeds distance threshold.
            return None

        # Returns player ID deemed to hold possession.
        return nearest_player

    # NORMALIZE POSSESSION
    # Helper to safely retrieve normalized player ID from explicit possession logs.
    def normalize_possession(self,ball_aquisition,frame_num):

        # Returns None if possession dataset is not supplied.
        if ball_aquisition is None:
            return None

        # Returns None if frame index exceeds acquisition array bounds.
        if frame_num >= len(ball_aquisition):
            return None

        # Extracts possession record for current frame.
        possession = (ball_aquisition[frame_num])

        # Parses and returns normalized player ID.
        return self.extract_player_id(possession)

    # DETECT PASSES
    # Evaluates frame sequences to detect valid completed passes between players on the same team.
    def detect_passes(self,player_tracks,ball_tracks,player_assignment=None,ball_aquisition=None):
        # List to store detected pass events.
        passes = []

        # Sets frame processing limit based on shortest available dataset sequence.
        total_frames = min(len(player_tracks),len(ball_tracks))

        # Tracks player ID holding ball in preceding possession frame.
        previous_holder = None

        # Tracks frame index of last verified possession.
        previous_frame = None

        # Iterates frame by frame over tracking sequence.
        for frame_num in range(total_frames):

            # Determine current holder
            current_holder = None

            # Attempts to pull holder from explicit acquisition dataset.
            if ball_aquisition is not None:
                current_holder = (self.normalize_possession(ball_aquisition,frame_num))

            # If possession information is unavailable estimate holder from geometry.
            if current_holder is None:
                current_holder = (self.find_ball_holder(frame_num, player_tracks,ball_tracks))

            # No holder
            if current_holder is None:
                continue

            # First holder recorded possession state and advances loop
            if previous_holder is None:
                previous_holder = (current_holder)
                previous_frame = (frame_num)
                continue

            # Same holder Ignores frame if player continues holding the ball.
            if current_holder == previous_holder:
                continue

            # Holder changed
            previous_team = -1
            current_team = -1

            # Default unassigned team flag.
            if player_assignment is not None:
                if (previous_frame is not None and previous_frame < len(player_assignment)):
                    frame_assignment = (player_assignment[previous_frame])
                    if isinstance(frame_assignment,dict):
                        previous_team = (frame_assignment.get(previous_holder,-1))
                if frame_num < len(player_assignment):
                    frame_assignment = (player_assignment[frame_num])
                    if isinstance(frame_assignment,dict):
                        current_team = (frame_assignment.get(current_holder,-1))

            # Same team = possible pass & Different team = possible interception
            if (previous_team != -1 and current_team != -1):
                if previous_team == current_team:

                    # Appends successful pass event dict when ball transfers between teammates.
                    passes.append({"frame": frame_num, "from_player": previous_holder, "to_player": current_holder, "team": current_team})
            previous_holder = current_holder
            previous_frame = frame_num

        return passes

    # DETECT INTERCEPTIONS
    # Detect interceptions based on a change of ball possession from one team to another.
    def detect_interceptions(self, player_tracks, ball_tracks, player_assignment=None, ball_aquisition=None):

        # List to store detected interception events.
        interceptions = []

        # Determines evaluation sequence length.
        total_frames = min(len(player_tracks), len(ball_tracks))

        previous_holder = None
        previous_frame = None

        for frame_num in range(total_frames):
            current_holder = None

            if ball_aquisition is not None:
                current_holder = (self.normalize_possession(ball_aquisition, frame_num))

            if current_holder is None:
                current_holder = (self.find_ball_holder(frame_num, player_tracks, ball_tracks))

            if current_holder is None:
                continue

            if previous_holder is None:
                previous_holder = (current_holder)
                previous_frame = (frame_num)
                continue

            if current_holder == previous_holder:
                continue

            previous_team = -1
            current_team = -1

            if player_assignment is not None:
                if (previous_frame is not None and previous_frame < len(player_assignment)):
                    previous_team = (player_assignment[previous_frame].get(previous_holder, -1))
                if frame_num < len(player_assignment):
                    current_team = (player_assignment[frame_num].get(current_holder, -1))

            # Different teams = interception
            if (previous_team != -1 and current_team != -1 and previous_team != current_team):
                interceptions.append({"frame": frame_num, "from_player": previous_holder, "to_player": current_holder, "from_team": previous_team, "to_team": current_team})

            previous_holder = current_holder
            previous_frame = frame_num

        return interceptions

