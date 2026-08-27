# Imports the standard OS module for file path manipulations.
import os

# Imports the sys module to modify Python runtime system paths.
import sys

# Imports pathlib for object-oriented filesystem path operations.
import pathlib

# Resolves the absolute path of the directory containing this script.
folder_path = pathlib.Path(__file__).parent.resolve()

# Adds the parent directory to sys.path so project utility modules can be imported.
sys.path.append(os.path.join(folder_path, "../"))

# Imports the measure_distance function to calculate Euclidean distance between two points.
from utils import measure_distance

# Defines a class to convert pixel movements on a 2D court into physical distance (meters) and speed (km/h).
class SpeedAndDistanceCalculator():
    # Constructor initializing pixel dimensions of the 2D court image and its corresponding real-world court dimensions in meters.
    def __init__(self,width_in_pixels,height_in_pixels,width_in_meters,height_in_meters):

        # Stores the 2D court image width in pixels.
        self.width_in_pixels = width_in_pixels
        # Stores the 2D court image height in pixels.
        self.height_in_pixels = height_in_pixels
        # Stores the real-world court length in meters (e.g., 28m for basketball).
        self.width_in_meters = width_in_meters
        # Stores the real-world court width in meters (e.g., 15m for basketball).
        self.height_in_meters = height_in_meters

    # Calculates real-world frame-by-frame movement distances for all players across video frames.
    def calculate_distance(self,tactical_player_positions):
        # Dictionary tracking the most recent 2D pixel position for each player ID.
        previous_players_position = {}
        # List storing player movement distances for every frame.
        output_distances = []
        # Iterates through player positions frame by frame.
        for frame_number, tactical_player_position_frame in enumerate(tactical_player_positions):
            # Initializes an empty dictionary for the current frame's player distance records.
            output_distances.append({})

            # Iterates through each player present in the current frame.
            for player_id, current_player_position in tactical_player_position_frame.items():
                # Checks if the player's position was recorded in a prior frame.
                if player_id in previous_players_position:
                    # Retrieves player's previous 2D pixel position.
                    previous_position = previous_players_position[player_id]
                    # Computes real-world distance in meters between previous and current positions.
                    meter_distance = self.calculate_meter_distance(previous_position, current_player_position)
                    # Stores calculated distance (in meters) for the player in current frame output.
                    output_distances[frame_number][player_id] = meter_distance
                # Updates player's last known position with the current frame position.
                previous_players_position[player_id] = current_player_position

        # Returns list of dictionaries containing frame-wise movement distances in meters for each player.
        return output_distances

    # Converts pixel coordinates into metric coordinates and computes physical distance in meters.
    def calculate_meter_distance(self, previous_pixel_position, current_pixel_position):
        # Unpacks (x, y) pixel coordinates of the previous position.
        previous_pixel_x, previous_pixel_y = previous_pixel_position
        # Unpacks (x, y) pixel coordinates of the current position.
        current_pixel_x, current_pixel_y = current_pixel_position
        # Scales previous x-pixel coordinate to meters.
        previous_meter_x = previous_pixel_x * self.width_in_meters / self.width_in_pixels
        # Scales previous y-pixel coordinate to meters.
        previous_meter_y = previous_pixel_y * self.height_in_meters / self.height_in_pixels
        # Scales current x-pixel coordinate to meters.
        current_meter_x = current_pixel_x * self.width_in_meters / self.width_in_pixels
        # Scales current y-pixel coordinate to meters.
        current_meter_y = current_pixel_y * self.height_in_meters / self.height_in_pixels
        # Computes Euclidean distance between points in metric space.
        meter_distance = measure_distance((current_meter_x, current_meter_y),(previous_meter_x, previous_meter_y))
        # Applies an empirical damping factor (0.4)
        meter_distance = meter_distance * 0.4
        # Returns adjusted distance in meters.
        return meter_distance

    def calculate_speed(self, distances, fps=30):
        """
        Calculate player speeds based on distances covered over the last 5 frames.
        Args:
            distances (list): List of dictionaries containing distance per player per frame,
                            as output by calculate_distance method.
            fps (float): Frames per second of the video, used to calculate elapsed time.
        Returns:
            list: List of dictionaries where each dictionary maps player_id to their
                speed in km/h at that frame.
        """

        # List to store frame-by-frame speed estimates for all players.
        speeds = []
        # Defines window size threshold required to estimate velocity smoothly.
        window_size = 5

        # Iterates through frames in the distance dataset.
        for frame_idx in range(len(distances)):
            # Initializes dictionary for current frame's player speeds.
            speeds.append({})
            # Iterates over all player IDs present in the current frame.
            for player_id in distances[frame_idx].keys():
                # Calculates lookback frame boundary index (up to 15 frames back).
                start_frame = max(0, frame_idx - (window_size * 3) + 1)
                # Initializes accumulator for total distance moved within the window.
                total_distance = 0
                # Counter for number of frame transitions where player was present.
                frames_present = 0
                # Tracks index of the last frame where the player was recorded.
                last_frame_present = None

                # Iterates across the lookback window frames up to current frame.
                for i in range(start_frame, frame_idx + 1):
                    # Checks if player has distance data in frame `i`.
                    if player_id in distances[i]:
                        # Checks if player was present in prior frame inside window.
                        if last_frame_present is not None:
                            # Sums distance covered in frame `i`.
                            total_distance += distances[i][player_id]
                            # Increments valid frame presence count.
                            frames_present += 1
                        # Updates last recorded frame index for player.
                        last_frame_present = i

                # Verifies player was continuously present across at least `window_size` frame intervals.
                if frames_present >= window_size:
                    # Converts frame count duration to seconds based on video FPS.
                    time_in_seconds = frames_present / fps
                    # Converts duration from seconds to hours.
                    time_in_hours = time_in_seconds / 3600
                    # Calculate speed in km/h
                    if time_in_hours > 0:
                        # Converts meters to kilometers and divides by hours to compute speed in km/h.
                        speed_kmh = (total_distance / 1000) / time_in_hours
                        # Assigns calculated speed in km/h to player ID in current frame.
                        speeds[frame_idx][player_id] = speed_kmh
                    else:
                        # Assigns zero speed if time delta is zero.
                        speeds[frame_idx][player_id] = 0
                else:
                    # Assigns zero speed if insufficient frames are available in the lookback window.
                    speeds[frame_idx][player_id] = 0
        # Returns list of dictionaries containing frame-wise player speeds in km/h.
        return speeds