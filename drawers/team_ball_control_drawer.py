# Imports OpenCV library for frame manipulations and graphics drawing.
import cv2

# Imports NumPy library for high-performance array calculations.
import numpy as np


class TeamBallControlDrawer:
    """
    A class responsible for calculating and drawing team ball control statistics on video frames.
    """

    # Default constructor initializing the TeamBallControlDrawer class instance.
    def __init__(self):
        pass

    def get_team_ball_control(self, player_assignment, ball_aquisition):
        """
        Calculate which team has ball control for each frame.
        Args:
            player_assignment (list): A list of dictionaries indicating team assignments for each player
                in the corresponding frame.
            ball_aquisition (list): A list indicating which player has possession of the ball in each frame.
        Returns:
            numpy.ndarray: An array indicating which team has ball control for each frame
                (1 for Team 1, 2 for Team 2, -1 for no control).
        """

        # List to store frame-by-frame team possession values (1, 2, or -1).
        team_ball_control = []

        # Iterates frame by frame pairing team assignments with ball possession data.
        for player_assignment_frame, ball_aquisition_frame in zip(player_assignment, ball_aquisition):

            # Appends -1 and skips iteration if no player has possession of the ball in this frame.
            if ball_aquisition_frame == -1:
                team_ball_control.append(-1)
                continue

            # Appends -1 and skips iteration if possessing player is not assigned to a team.
            if ball_aquisition_frame not in player_assignment_frame:
                team_ball_control.append(-1)
                continue

            # Appends 1 if possessing player belongs to Team 1.
            if player_assignment_frame[ball_aquisition_frame] == 1:
                team_ball_control.append(1)

            # Appends 2 if possessing player belongs to Team 2.
            else:
                team_ball_control.append(2)

        # Converts team ball control list to a NumPy array for fast operations.
        team_ball_control = np.array(team_ball_control)

        # Returns array containing team possession flags for all frames.
        return team_ball_control

    def draw(self, video_frames, player_assignment, ball_aquisition):
        """
        Draw team ball control statistics on a list of video frames.
        Args:
            video_frames (list): A list of frames (as NumPy arrays or image objects) on which to draw.
            player_assignment (list): A list of dictionaries indicating team assignments for each player
                in the corresponding frame.
            ball_aquisition (list): A list indicating which player has possession of the ball in each frame.
        Returns:
            list: A list of frames with team ball control statistics drawn on them.
        """

        # Calculates ball control NumPy array for the entire video sequence.
        team_ball_control = self.get_team_ball_control(player_assignment, ball_aquisition)

        # List to store annotated video frames with drawn statistics overlays.
        output_video_frames = []

        # Loops through frames and tracks frame index.
        for frame_num, frame in enumerate(video_frames):

            # Draws semi-transparent ball control panel on current frame.
            frame_drawn = self.draw_frame(frame, frame_num, team_ball_control)

            # Appends drawn frame to output frame collection list.
            output_video_frames.append(frame_drawn)

        # Returns processed frames list containing drawn ball control HUD overlays.
        return output_video_frames

    def draw_frame(self, frame, frame_num, team_ball_control):
        """
        Draw a semi-transparent overlay of team ball control percentages on a single frame.

        Args:
            frame (numpy.ndarray): The current video frame on which the overlay will be drawn.
            frame_num (int): The index of the current frame.
            team_ball_control (numpy.ndarray): An array indicating which team has ball control for each frame.

        Returns:
            numpy.ndarray: The frame with the semi-transparent overlay and statistics.
        """

        # Creates duplicate copy of current frame to draw overlay shape.
        overlay = frame.copy()

        # Sets font scaling factor for rendered text overlay.
        font_scale = 0.7

        # Sets stroke thickness for rendered text.
        font_thickness = 2

        # Extracts frame height and width pixel dimensions from shape array.
        frame_height, frame_width = overlay.shape[:2]

        # Computes left X coordinate of stats overlay box (60% width).
        rect_x1 = int(frame_width * 0.60)

        # Computes top Y coordinate of stats overlay box (75% height).
        rect_y1 = int(frame_height * 0.75)

        # Computes right X coordinate of stats overlay box (99% width).
        rect_x2 = int(frame_width * 0.99)

        # Computes bottom Y coordinate of stats overlay box (90% height).
        rect_y2 = int(frame_height * 0.90)

        # Sets horizontal text start coordinate (63% width).
        text_x = int(frame_width * 0.63)

        # Sets baseline Y coordinate for Team 1 stats text.
        text_y1 = int(frame_height * 0.80)

        # Sets baseline Y coordinate for Team 2 stats text.
        text_y2 = int(frame_height * 0.88)

        # Draws filled white rectangle on overlay image copy.
        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)

        # Sets opacity blend weight factor for semi-transparent rendering.
        alpha = 0.8

        # Blends overlay image into original frame to render 80% opaque white background box.
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Slices team ball control array from start up to current frame index.
        team_ball_control_till_frame = team_ball_control[:frame_num + 1]

        # Counts total frames Team 1 held possession up to current frame.
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]

        # Counts total frames Team 2 held possession up to current frame.
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]

        # Calculates Team 1 cumulative possession ratio float.
        team_1 = team_1_num_frames / (team_ball_control_till_frame.shape[0])

        # Calculates Team 2 cumulative possession ratio float.
        team_2 = team_2_num_frames / (team_ball_control_till_frame.shape[0])

        # Renders Team 1 ball control percentage text string inside overlay box.
        cv2.putText(frame, f"Team 1 Ball Control: {team_1 * 100:.2f}%", (text_x, text_y1), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), font_thickness)

        # Renders Team 2 ball control percentage text string inside overlay box.
        cv2.putText(frame, f"Team 2 Ball Control: {team_2 * 100:.2f}%", (text_x, text_y2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), font_thickness)

        # Returns updated frame featuring calculated team ball control statistics overlay box.
        return frame