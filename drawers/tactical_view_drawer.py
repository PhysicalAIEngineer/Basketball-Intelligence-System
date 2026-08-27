# Imports OpenCV library for reading image files and drawing graphical elements.
import cv2


# Defines a class responsible for rendering a top-down 2D court overlay populated with keypoints and player positions.
class TacticalViewDrawer:

    # Constructor defining top-left pixel placement coordinates and team marker BGR colors.
    def __init__(self, team_1_color=[255, 245, 238], team_2_color=[128, 0, 0]):

        # Sets top-left horizontal pixel offset for rendering the court mini-map overlay.
        self.start_x = 20

        # Sets top-left vertical pixel offset for rendering the court mini-map overlay.
        self.start_y = 40

        # Sets default BGR color array for Team 1 player markers (default off-white).
        self.team_1_color = team_1_color

        # Sets default BGR color array for Team 2 player markers (default maroon).
        self.team_2_color = team_2_color

    def draw(self, video_frames, court_image_path, width, height,
             tactical_court_keypoints, tactical_player_positions=None, player_assignment=None,
             ball_acquisition=None):
        """
        Draw tactical view with court keypoints and player positions.
        Args:
            video_frames (list): List of video frames to draw on.
            court_image_path (str): Path to the court image.
            width (int): Width of the tactical view.
            height (int): Height of the tactical view.
            tactical_court_keypoints (list): List of court keypoints in tactical view.
            tactical_player_positions (list, optional): List of dictionaries mapping player IDs to
                their positions in tactical view coordinates.
            player_assignment (list, optional): List of dictionaries mapping player IDs to team assignments.
            ball_acquisition (list, optional): List indicating which player has the ball in each frame.
        Returns:
            list: List of frames with tactical view drawn on them.
        """

        # Loads the 2D court image template from file.
        court_image = cv2.imread(court_image_path)

        # Resizes 2D court template image to match defined width and height dimensions.
        court_image = cv2.resize(court_image, (width, height))

        # List to store annotated video frames containing rendered tactical mini-maps.
        output_video_frames = []

        # Iterates through video frames tracking frame index.
        for frame_idx, frame in enumerate(video_frames):

            # Creates a copy of current video frame to prevent mutating the original input.
            frame = frame.copy()

            # Calculates top Y bounding coordinate for overlay region.
            y1 = self.start_y
            # Calculates bottom Y bounding coordinate for overlay region.
            y2 = self.start_y + height
            # Calculates left X bounding coordinate for overlay region.
            x1 = self.start_x
            # Calculates right X bounding coordinate for overlay region.
            x2 = self.start_x + width

            # Defines blending opacity factor for transparent mini-map background.
            alpha = 0.6
            # Extracts ROI region from main video frame corresponding to mini-map position.
            overlay = frame[y1:y2, x1:x2].copy()
            # Blends 2D court graphic with video frame ROI producing 60% transparent court background.
            cv2.addWeighted(court_image, alpha, overlay, 1 - alpha, 0, frame[y1:y2, x1:x2])

            # Draw court keypoints introducing court landmark keypoint rendering section.
            for keypoint_index, keypoint in enumerate(tactical_court_keypoints):
                # keypoint target coordinate tuple.
                x, y = keypoint
                # Shifts keypoint X position by mini-map offset.
                x += self.start_x
                # Shifts keypoint Y position by mini-map offset.
                y += self.start_y
                # Draws red filled circle marker for each court landmark keypoint.
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                # Draws green index number text beside keypoint marker.
                cv2.putText(frame, str(keypoint_index), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Checks if optional player tracking datasets exist and contain data for current frame.
            if tactical_player_positions and player_assignment and frame_idx < len(tactical_player_positions):
                # Extracts player 2D position dictionary for current frame.
                frame_positions = tactical_player_positions[frame_idx]
                # Safely extracts player team assignments for current frame.
                frame_assignments = player_assignment[frame_idx] if frame_idx < len(player_assignment) else {}
                # Retrieves ID of possessing player in current frame, defaulting to -1.
                player_with_ball = ball_acquisition[frame_idx] if ball_acquisition and frame_idx < len(
                    ball_acquisition) else -1

                # Iterates over every player present in top-down tactical coordinates.
                for player_id, position in frame_positions.items():
                    # Get player's team Retrieves team ID for player, defaulting to Team 1 if unassigned.
                    team_id = frame_assignments.get(player_id, 1)  # Default to team 1 if not assigned

                    # Set color based on team chooses Team 1 color or Team 2 color depending on `team_id`.
                    color = self.team_1_color if team_id == 1 else self.team_2_color

                    # Adjust position to overlay coordinates
                    x, y = int(position[0]) + self.start_x, int(position[1]) + self.start_y

                    # Draw player circle
                    player_radius = 8

                    # Draws filled circle representing player on top-down court.
                    cv2.circle(frame, (x, y), player_radius, color, -1)

                    # Draws red outer outline around player dot holding possession.
                    if player_id == player_with_ball:
                        cv2.circle(frame, (x, y), player_radius + 3, (0, 0, 255), 2)

            # Appends frame containing complete tactical court visualizer to output frame list.
            output_video_frames.append(frame)

        # Returns list of video frames populated with tactical court mini-maps.
        return output_video_frames