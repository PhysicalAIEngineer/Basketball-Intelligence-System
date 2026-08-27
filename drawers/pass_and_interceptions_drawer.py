# Imports OpenCV library for frame manipulations and rendering text graphics.
import cv2

# Imports NumPy library for numerical array processing operations.
import numpy as np

class PassInterceptionDrawer:
    """
    A class responsible for calculating and drawing pass and interception statistics
    on a sequence of video frames.
    """

    # Default constructor initializing the PassInterceptionDrawer class instance.
    def __init__(self):
        pass

    def get_stats(self, passes, interceptions):
        """
        Calculate the number of passes and interceptions for Team 1 and Team 2.

        Args:
            passes (list): A list of integers representing pass events at each frame.
                (1 represents a pass by Team 1, 2 represents a pass by Team 2, 0 represents no pass.)
            interceptions (list): A list of integers representing interception events at each frame.
                (1 represents an interception by Team 1, 2 represents an interception by Team 2, 0 represents no interception.)

        Returns:
            tuple: A tuple of four integers (team1_pass_total, team2_pass_total,
                team1_interception_total, team2_interception_total) indicating the total
                number of passes and interceptions for both teams.
        """

        # List to record frame indices where Team 1 completed a pass.
        team1_passes = []

        # List to record frame indices where Team 2 completed a pass.
        team2_passes = []

        # List to record frame indices where Team 1 made an interception.
        team1_interceptions = []

        # List to record frame indices where Team 2 made an interception.
        team2_interceptions = []

        # Iterates through passes and interceptions sequence paired by frame index.
        for frame_num, (pass_frame, interception_frame) in enumerate(zip(passes, interceptions)):

            # Records frame index if Team 1 executed a pass.
            if pass_frame == 1:
                team1_passes.append(frame_num)

            # Records frame index if Team 2 executed a pass.
            elif pass_frame == 2:
                team2_passes.append(frame_num)

            # Records frame index if Team 1 executed an interception.
            if interception_frame == 1:
                team1_interceptions.append(frame_num)

            # Records frame index if Team 2 executed an interception.
            elif interception_frame == 2:
                team2_interceptions.append(frame_num)

        # Returns tuple of counts containing total passes and interceptions for both teams.
        return len(team1_passes), len(team2_passes), len(team1_interceptions), len(team2_interceptions)

    def draw(self, video_frames, passes, interceptions):
        """
        Draw pass and interception statistics on a list of video frames.
        Args:
            video_frames (list): A list of frames (as NumPy arrays or image objects) on which to draw.
            passes (list): A list of integers representing pass events at each frame.
                (1 represents a pass by Team 1, 2 represents a pass by Team 2, 0 represents no pass.)
            interceptions (list): A list of integers representing interception events at each frame.
                (1 represents an interception by Team 1, 2 represents an interception by Team 2, 0 represents no interception.)
        Returns:
            list: A list of frames with pass and interception statistics drawn on them.
        """

        # List to store annotated video frames containing rendered pass/interception HUDs.
        output_video_frames = []

        # Iterates frame by frame while tracking the frame index.
        for frame_num, frame in enumerate(video_frames):

            # Skips drawing on the first frame (frame 0).
            if frame_num == 0:
                continue

            # Draws the semi-transparent pass & interception stats panel on current frame.
            frame_drawn = self.draw_frame(frame, frame_num, passes, interceptions)

            # Appends drawn frame to output list.
            output_video_frames.append(frame_drawn)

        # Returns list of video frames containing pass and interception statistical overlays.
        return output_video_frames

    def draw_frame(self, frame, frame_num, passes, interceptions):
        """
        Draw a semi-transparent overlay of pass and interception counts on a single frame.

        Args:
            frame (numpy.ndarray): The current video frame on which the overlay will be drawn.
            frame_num (int): The index of the current frame.
            passes (list): A list of pass events up to this frame.
            interceptions (list): A list of interception events up to this frame.

        Returns:
            numpy.ndarray: The frame with the semi-transparent overlay and statistics.
        """

        # Creates duplicate copy of current video frame to construct overlay panel.
        overlay = frame.copy()

        # Sets font scaling factor for rendered text strings.
        font_scale = 0.7

        # Sets line thickness for font text strokes.
        font_thickness = 2

        # Extracts height and width pixel dimensions from frame array shape.
        frame_height, frame_width = overlay.shape[:2]
        # Computes left X coordinate of stats overlay box (16% width).
        rect_x1 = int(frame_width * 0.16)
        # Computes top Y coordinate of stats overlay box (75% height).
        rect_y1 = int(frame_height * 0.75)
        # Computes right X coordinate of stats overlay box (55% width).
        rect_x2 = int(frame_width * 0.55)
        # Computes bottom Y coordinate of stats overlay box (90% height).
        rect_y2 = int(frame_height * 0.90)
        # Sets horizontal text alignment coordinate (19% width).
        text_x = int(frame_width * 0.19)
        # Sets baseline Y coordinate for Team 1 stats text.
        text_y1 = int(frame_height * 0.80)
        # Sets baseline Y coordinate for Team 2 stats text.
        text_y2 = int(frame_height * 0.88)

        # Draws filled white rectangle on overlay image copy.
        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)

        # Sets opacity blend weight factor for semi-transparent rendering.
        alpha = 0.8

        # Blends overlay image into original frame creating an 80% opaque white background box.
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Slices pass events array up to current frame index.
        passes_till_frame = passes[:frame_num + 1]

        # Slices interception events array up to current frame index.
        interceptions_till_frame = interceptions[:frame_num + 1]

        # Calculates cumulative pass and interception totals for both teams up to current frame.
        team1_passes, team2_passes, team1_interceptions, team2_interceptions = self.get_stats(
            passes_till_frame, interceptions_till_frame)

        # Renders Team 1 pass and interception counts inside top half of overlay box.
        cv2.putText(frame,f"Team 1 - Passes: {team1_passes} Interceptions: {team1_interceptions}",
            (text_x, text_y1), cv2.FONT_HERSHEY_SIMPLEX, font_scale,(0, 0, 0),font_thickness)

        # Renders Team 2 pass and interception counts inside bottom half of overlay box.
        cv2.putText(frame,f"Team 2 - Passes: {team2_passes} Interceptions: {team2_interceptions}",
            (text_x, text_y2), cv2.FONT_HERSHEY_SIMPLEX, font_scale,(0, 0, 0), font_thickness)

        # Returns modified frame array featuring pass and interception metrics overlay box.
        return frame