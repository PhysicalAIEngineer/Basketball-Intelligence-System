from drawers import PlayerTracksDrawer
from trackers import PlayerTracker
from utils import read_video, save_video


def main():

    # Read video
    video_frames = read_video(
        "input_video/video_1.mp4"
    )

    # Player tracking
    player_tracker = PlayerTracker(
        "model/basketball_player_detection_training.pt"
    )

    player_tracks = player_tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path="stubs/player_tracks_stubs.pkl"
    )

    # Player assignment
    player_assignment = [
        {}
        for _ in video_frames
    ]

    # Ball acquisition
    ball_aquisition = [
        {}
        for _ in video_frames
    ]

    # Draw
    player_tracks_drawer = PlayerTracksDrawer()

    output_video_frames = player_tracks_drawer.draw(
        video_frames,
        player_tracks,
        player_assignment,
        ball_aquisition
    )

    # Save
    save_video(
        output_video_frames,
        "output_video/video_1.avi"
    )


if __name__ == "__main__":
    main()