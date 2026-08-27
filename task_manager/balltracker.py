from drawers import PlayerTracksDrawer, BallTracksDrawer
from trackers import PlayerTracker, BallTracker
from utils import read_video, save_video


def main():

    # ============================================================
    # 1. Read video
    # ============================================================

    video_path = "input_video/video_1.mp4"

    video_frames = read_video(video_path)

    print(
        f"Total video frames: {len(video_frames)}"
    )


    # ============================================================
    # 2. Initialize Player Tracker
    # ============================================================

    player_tracker = PlayerTracker(
        "model/basketball_player_detection_training.pt"
    )


    # ============================================================
    # 3. Initialize Ball Tracker
    # ============================================================

    ball_tracker = BallTracker(
        "model/basketball_ball_training.pt"
    )


    # ============================================================
    # 4. Player Tracking
    # ============================================================

    player_tracks = player_tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path="stubs/player_tracks_stubs.pkl"
    )

    print(
        f"Player tracking completed: "
        f"{len(player_tracks)} frames"
    )


    # ============================================================
    # 5. Ball Tracking
    # ============================================================

    ball_tracks = ball_tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path="stubs/ball_tracks_stubs.pkl"
    )

    print(
        f"Ball tracking completed: "
        f"{len(ball_tracks)} frames"
    )


    # ============================================================
    # 6. Player Assignment
    # ============================================================

    player_assignment = [
        {}
        for _ in video_frames
    ]


    # ============================================================
    # 7. Ball Acquisition
    # ============================================================

    ball_aquisition = [
        {}
        for _ in video_frames
    ]


    # ============================================================
    # 8. Initialize Drawers
    # ============================================================

    player_tracks_drawer = PlayerTracksDrawer()

    ball_tracks_drawer = BallTracksDrawer()


    # ============================================================
    # 9. Draw Player Tracks
    # ============================================================

    output_video_frames = player_tracks_drawer.draw(
        video_frames,
        player_tracks,
        player_assignment,
        ball_aquisition
    )


    # ============================================================
    # 10. Draw Ball Tracks
    # ============================================================

    output_video_frames = ball_tracks_drawer.draw(
        output_video_frames,
        ball_tracks
    )


    # ============================================================
    # 11. Save Final Video
    # ============================================================

    output_video_path = "output_video/video_2.avi"

    save_video(
        output_video_frames,
        output_video_path
    )

    print(
        f"Output video saved to: "
        f"{output_video_path}"
    )


if __name__ == "__main__":
    main()