# Imports OpenCV library for frame manipulations and rendering text overlays on images.
import cv2

# Defines a class responsible for drawing running metrics (speed in km/h and accumulated distance in meters) on video frames.
class SpeedAndDistanceDrawer():

    # Default constructor initializing the SpeedAndDistanceDrawer class instance.
    def __init__(self):
        pass

    # Draws speed and cumulative distance text metrics below each player's feet position across video frames.
    def draw(self, video_frames, player_tracks, player_distances_per_frame, player_speed_per_frame):

        # List to store processed video frames containing drawn metric overlays.
        output_video_frames = []

        # Dictionary to track and accumulate total running distance (in meters) for each player ID over time.
        total_distances = {}

        # Iterates through frames while matching player bounding boxes, frame distances, and speed calculations.
        for frame, player_tracks, player_distance, player_speed in zip(video_frames, player_tracks,
                                                                       player_distances_per_frame,
                                                                       player_speed_per_frame):

            # Creates a duplicate copy of the current video frame to prevent modifying the original input array.
            output_frame = frame.copy()

            # Iterates through player distance entries for the current frame.
            for player_id, distance in player_distance.items():
                # Initializes cumulative distance to 0 for newly observed player IDs.
                if player_id not in total_distances:
                    total_distances[player_id] = 0

                # Accumulates frame distance into player's total distance total.
                total_distances[player_id] += distance

            # Iterates through all player bounding boxes present in current frame tracking data.
            for player_id, bbox in player_tracks.items():
                # top-left (x1, y1) and bottom-right (x2, y2) coordinates from player bounding box.
                x1, y1, x2, y2 = bbox['bbox']
                # Calculates ground position at player's feet (horizontal center x, bottom border y).
                position = [int((x1 + x2) / 2), int(y2)]
                # Offsets baseline Y coordinate downward by 40 pixels to position text below player feet.
                position[1] += 40
                # Retrieves cumulative distance in meters for current player ID (returns None if unrecorded).
                distance = total_distances.get(player_id, None)
                # Retrieves calculated speed in km/h for current player ID (returns None if unrecorded).
                speed = player_speed.get(player_id, None)
                # Renders player speed text string in black below player's feet.
                if speed is not None:
                    cv2.putText(output_frame, f"{speed:.2f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                                2)
                # Renders cumulative distance text string  20 pixels below the speed text.
                if distance is not None:
                    cv2.putText(output_frame, f"{distance:.2f} m", (position[0], position[1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Appends frame populated with speed and distance stats to output list.
            output_video_frames.append(output_frame)

        # Returns list of video frames populated with speed and distance metric overlays.
        return output_video_frames