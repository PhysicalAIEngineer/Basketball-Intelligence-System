# Imports the helper functions 'draw_ellipse' and 'draw_traingle' from the local 'utils' module.
from .utils import draw_ellipse, draw_traingle

class PlayerTracksDrawer:
    """
    A class responsible for drawing player tracks and ball possession indicators on video frames.
    Attributes:
        default_player_team_id (int): Default team ID used when a player's team is not specified.
        team_1_color (list): RGB color used to represent Team 1 players.
        team_2_color (list): RGB color used to represent Team 2 players.
    """

    # Constructor docstring specifying input color defaults.
    def __init__(self, team_1_color=[255, 245, 238], team_2_color=[128, 0, 0]):
        """
        Initialize the PlayerTracksDrawer with specified team colors.
        Args:
            team_1_color (list, optional): RGB color for Team 1. Defaults to [255, 245, 238].
            team_2_color (list, optional): RGB color for Team 2. Defaults to [128, 0, 0].
        """
        # Sets default team assignment ID to 1 if a player's team is unassigned.
        self.default_player_team_id = 1
        # Stores BGR/RGB color list for Team 1 player markers (default off-white).
        self.team_1_color = team_1_color
        # Stores BGR/RGB color list for Team 2 player markers (default maroon).
        self.team_2_color = team_2_color

    def draw(self, video_frames, tracks, player_assignment, ball_aquisition):
        """
        Draw player tracks and ball possession indicators on a list of video frames.
        Args:
            video_frames (list): A list of frames (as NumPy arrays or image objects) on which to draw.
            tracks (list): A list of dictionaries where each dictionary contains player tracking information
                for the corresponding frame.
            player_assignment (list): A list of dictionaries indicating team assignments for each player
                in the corresponding frame.
            ball_aquisition (list): A list indicating which player has possession of the ball in each frame.
        Returns:
            list: A list of frames with player tracks and ball possession indicators drawn on them.
        """

        # List to store annotated video frames with drawn player markers and possession indicators.
        output_video_frames = []

        # Iterates frame by frame while keeping track of the frame index (`frame_num`).
        for frame_num, frame in enumerate(video_frames):
            # Creates a duplicate copy of the current frame to prevent mutating the original video array.
            frame = frame.copy()
            # Extracts dictionary of player detections and bounding boxes for current frame.
            player_dict = tracks[frame_num]
            # Extracts dictionary mapping player IDs to their team assignments for current frame.
            player_assignment_for_frame = player_assignment[frame_num]
            # Retrieves player ID currently possessing the ball in this frame (or -1/None if unassigned).
            player_id_has_ball = ball_aquisition[frame_num]

            # Iterates through player track ID keys and player tracking dictionary objects.
            for track_id, player in player_dict.items():
                # Retrieves team ID for current player, using `default_player_team_id` if missing.
                team_id = player_assignment_for_frame.get(track_id, self.default_player_team_id)
                # Assigns Team 1 color if player belongs to Team 1.
                if team_id == 1:
                    color = self.team_1_color
                # Assigns Team 2 color if player belongs to Team 2.
                else:
                    color = self.team_2_color
                # Draws an ellipse at player's feet along with track ID badge using assigned team color.
                frame = draw_ellipse(frame, player["bbox"], color, track_id)
                # Draws red pointer triangle above player's head if they currently possess the ball.
                if track_id == player_id_has_ball:
                    frame = draw_traingle(frame, player["bbox"], (0, 0, 255))
            # Appends processed frame containing drawn player rings and possession triangle to output list.
            output_video_frames.append(frame)
        # Returns list of video frames populated with player tracking rings and possession annotations.
        return output_video_frames
