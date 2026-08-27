# Imports the sys module to interact with system-specific parameters and paths.
import sys

# Appends the parent directory to sys.path to enable importing modules from parent directories.
sys.path.append('../')

# Imports helper functions for measuring Euclidean distance and extracting bounding box center points.
from utils.bbox_utils import measure_distance, get_center_of_bbox

class BallAquisitionDetector:
    """
    Detects ball acquisition by players in a basketball game.

    This class determines which player is most likely in possession of the ball
    by analyzing bounding boxes for both the ball and the players. It combines
    distance measurements between the ball and key points of each player's bounding
    box with containment ratios of the ball within a player's bounding box.
    """

    def __init__(self):
        """
        Initialize the BallAquisitionDetector with default thresholds.

        Attributes:
            possession_threshold (int): Maximum distance (in pixels) at which
                a player can be considered to have the ball if containment is insufficient.
            min_frames (int): Minimum number of consecutive frames required for a player
                to be considered in possession of the ball.
            containment_threshold (float): Containment ratio above which a player
                is considered to hold the ball without requiring distance checking.
        """

        # Sets maximum pixel distance threshold (50px) between player key points and ball center.
        self.possession_threshold = 50

        # Sets consecutive frame requirement (13 frames) to confirm valid ball possession.
        self.min_frames = 13

        # Sets ball box area overlap ratio threshold (80%) for high-confidence containment.
        self.containment_threshold = 0.8

    def get_key_basketball_player_assignment_points(self, player_bbox, ball_center):
        """
        Compute a list of key points around a player's bounding box.
        Key points are used to measure distance to the ball more accurately than
        using just the center of the bounding box.
        Args:
            bbox (tuple or list): A bounding box in the format (x1, y1, x2, y2).
        Returns:
            list of tuple: A list of (x, y) coordinates representing key points
            around the bounding box.
        """

        # Extracts X-coordinate of the ball's center position.
        ball_center_x = ball_center[0]

        # Extracts Y-coordinate of the ball's center position.
        ball_center_y = ball_center[1]

        # top-left (x1, y1) and bottom-right (x2, y2) coordinates of the player bounding box.
        x1, y1, x2, y2 = player_bbox

        # Computes width of the player bounding box.
        width = x2 - x1

        # Computes height of the player bounding box.
        height = y2 - y1

        # List to store key point coordinates along player bounding box.
        output_points = []

        # Adds left and right boundary projection points if ball Y-level falls within player box height.
        if ball_center_y > y1 and ball_center_y < y2:
            output_points.append((x1, ball_center_y))
            output_points.append((x2, ball_center_y))

        # Adds top and bottom boundary projection points if ball X-level falls within player box width.
        if ball_center_x > x1 and ball_center_x < x2:
            output_points.append((ball_center_x, y1))
            output_points.append((ball_center_x, y2))

        # Adds 10 fixed geometric key points across corners, midpoints, center, and torso region.
        output_points += [
            (x1 + width // 2, y1),  # top center
            (x2, y1),  # top right
            (x1, y1),  # top left
            (x2, y1 + height // 2),  # center right
            (x1, y1 + height // 2),  # center left
            (x1 + width // 2, y1 + height // 2),  # center point
            (x2, y2),  # bottom right
            (x1, y2),  # bottom left
            (x1 + width // 2, y2),  # bottom center
            (x1 + width // 2, y1 + height // 3),  # mid-top center
        ]

        # Returns total list of reference key points for player distance evaluation.
        return output_points

    def calculate_ball_containment_ratio(self, player_bbox, ball_bbox):
        """
        Calculate how much of the ball is contained within a player's bounding box.
        This is computed as the ratio of the intersection of the bounding boxes
        to the area of the ball's bounding box.
        Args:
            player_bbox (tuple or list): The player's bounding box (x1, y1, x2, y2).
            ball_bbox (tuple or list): The ball's bounding box (x1, y1, x2, y2).
        Returns:
            float: A value between 0.0 and 1.0 indicating what fraction of the
            ball is inside the player's bounding box.
        """

        # player bounding box coordinates.
        px1, py1, px2, py2 = player_bbox

        # ball bounding box coordinates.
        bx1, by1, bx2, by2 = ball_bbox

        # Calculates top-left X of intersection rectangle.
        intersection_x1 = max(px1, bx1)
        # Calculates top-left Y of intersection rectangle.
        intersection_y1 = max(py1, by1)
        # Calculates bottom-right X of intersection rectangle.
        intersection_x2 = min(px2, bx2)
        # Calculates bottom-right Y of intersection rectangle.
        intersection_y2 = min(py2, by2)

        # Returns 0.0 containment ratio if boxes do not intersect.
        if intersection_x2 < intersection_x1 or intersection_y2 < intersection_y1:
            return 0.0

        # Computes pixel area of intersection overlap rectangle.
        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)

        # Computes total pixel area of ball bounding box.
        ball_area = (bx2 - bx1) * (by2 - by1)

        # Returns ratio representing fraction of ball box inside player box (0.0 to 1.0).
        return intersection_area / ball_area

    def find_minimum_distance_to_ball(self, ball_center, player_bbox):
        """
        Compute the minimum distance from any key point on a player's bounding box
        to the center of the ball.
        Args:
            ball_center (tuple): (x, y) coordinates of the center of the ball.
            player_bbox (tuple): A bounding box (x1, y1, x2, y2) for the player.
        Returns:
            float: The smallest distance from the ball center to
            any key point on the player's bounding box.
        """

        # Generates array of geometric reference key points around player bounding box.
        key_points = self.get_key_basketball_player_assignment_points(player_bbox, ball_center)

        # Computes Euclidean distance to ball center for each key point and returns the minimum value.
        return min(measure_distance(ball_center, point) for point in key_points)

    def find_best_candidate_for_possession(self, ball_center, player_tracks_frame, ball_bbox):
        """
        Determine which player in a single frame is most likely to have the ball.
        Players who have a high containment ratio of the ball are prioritized.
        If no player has a high containment ratio, the player with the smallest
        distance to the ball that is below the possession threshold is selected.
        Args:
            ball_center (tuple): (x, y) coordinates of the ball center.
            player_tracks_frame (dict): Mapping from player_id to info about that player,
                including a 'bbox' key with (x1, y1, x2, y2).
            ball_bbox (tuple): Bounding box for the ball (x1, y1, x2, y2).
        Returns:
            int: (best_player_id), or (-1 ) if none found.
        """

        # List to store tuples of (player_id, min_distance) for players meeting high containment criteria.
        high_containment_players = []

        # List to store tuples of (player_id, min_distance) for all other players.
        regular_distance_players = []

        # Iterates over every player track entry present in current frame dictionary.
        for player_id, player_info in player_tracks_frame.items():

            # Retrieves player bounding box list; defaults to empty list if missing.
            player_bbox = player_info.get('bbox', [])

            # Skips iteration if player bounding box is empty or invalid.
            if not player_bbox:
                continue

            # Calculates ball-in-player bounding box containment ratio float.
            containment = self.calculate_ball_containment_ratio(player_bbox, ball_bbox)

            # Finds minimum pixel distance from ball center to player key points.
            min_distance = self.find_minimum_distance_to_ball(ball_center, player_bbox)

            # Categorizes player into high containment list if containment exceeds 80%.
            if containment > self.containment_threshold:
                high_containment_players.append((player_id, min_distance))
            else:
                regular_distance_players.append((player_id, min_distance))

        # Selects player with maximum distance among high containment group (furthest valid keypoint match) and returns ID.
        if high_containment_players:
            best_candidate = max(high_containment_players, key=lambda x: x[1])
            return best_candidate[0]

        # Second priority: players within distance threshold indicating evaluation of secondary proximity-based candidates.
        if regular_distance_players:
            best_candidate = min(regular_distance_players, key=lambda x: x[1])
            if best_candidate[1] < self.possession_threshold:
                return best_candidate[0]

        # Returns -1 if no players meet containment or distance criteria.
        return -1

    def detect_ball_possession(self, player_tracks, ball_tracks):
        """
        Detect which player has the ball in each frame based on bounding box information.
        Loops through all frames, looks up ball bounding boxes and player bounding boxes,
        and uses find_best_candidate_for_possession to determine who has the ball.
        Requires a player to hold possession for at least min_frames consecutive frames
        before confirming possession.
        Args:
            player_tracks (list): A list of dictionaries for each frame, where each dictionary
                maps player_id to player information including 'bbox'.
            ball_tracks (list): A list of dictionaries for each frame, where each dictionary
                maps ball_id to ball information including 'bbox'.
        Returns:
            list: A list of length num_frames with the player_id who has possession,
            or -1 if no one is determined to have possession in that frame.
        """

        # Retrieves total count of frames in the ball tracking dataset.
        num_frames = len(ball_tracks)

        # Initializes output possession list with -1 (no possession) for all frames.
        possession_list = [-1] * num_frames

        # Dictionary tracking consecutive frame counters for potential possessing player IDs.
        consecutive_possession_count = {}

        # Loops through frame indices sequentially across video length.
        for frame_num in range(num_frames):

            # Retrieves ball tracking dictionary for ball object ID 1 in current frame.
            ball_info = ball_tracks[frame_num].get(1, {})

            # Skips to next frame if ball tracking information is missing.
            if not ball_info:
                continue

            # Extracts ball bounding box list coordinates.
            ball_bbox = ball_info.get('bbox', [])

            # Skips to next frame if ball bounding box is empty.
            if not ball_bbox:
                continue

            # Calculates center (x, y) coordinates of the ball bounding box.
            ball_center = get_center_of_bbox(ball_bbox)

            # Identifies best candidate player ID holding possession in this frame.
            best_player_id = self.find_best_candidate_for_possession(ball_center,player_tracks[frame_num], ball_bbox)

            # Increments consecutive possession frame counter for detected candidate player.
            if best_player_id != -1:
                number_of_consecutive_frames = consecutive_possession_count.get(best_player_id, 0) + 1
                # Resets consecutive tracking dictionary to focus exclusively on current candidate.
                consecutive_possession_count = {best_player_id: number_of_consecutive_frames}

                # Confirms and sets player ID in output possession list if held for at least 13 consecutive frames.
                if consecutive_possession_count[best_player_id] >= self.min_frames:
                    possession_list[frame_num] = best_player_id
            else:
                # Resets consecutive possession tracker dictionary if no valid player candidate found.
                consecutive_possession_count = {}

        # Returns final list containing possessing player ID or -1 for each video frame.
        return possession_list