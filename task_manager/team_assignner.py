from pathlib import Path

from drawers import PlayerTracksDrawer, BallTracksDrawer
from trackers import PlayerTracker, BallTracker
from utils import read_video, save_video
from team_assigner import TeamAssigner


def main():

    # ============================================================
    # 1. Project paths
    # ============================================================

    BASE_DIR = Path(__file__).resolve().parent

    # Input video
    video_path = (
        BASE_DIR
        / "input_video"
        / "video_1.mp4"
    )

    # Player detection model
    player_model_path = (
        BASE_DIR
        / "model"
        / "basketball_player_detection_training.pt"
    )

    # Basketball detection model
    ball_model_path = (
        BASE_DIR
        / "model"
        / "basketball_ball_training.pt"
    )

    # Player tracking cache
    player_stub_path = (
        BASE_DIR
        / "stubs"
        / "player_tracks_stubs.pkl"
    )

    # Ball tracking cache
    ball_stub_path = (
        BASE_DIR
        / "stubs"
        / "ball_tracks_stubs.pkl"
    )

    # Team assignment cache
    player_assignment_stub_path = (
        BASE_DIR
        / "stubs"
        / "player_assignment_stubs.pkl"
    )

    # Output video
    output_video_path = (
        BASE_DIR
        / "output_video"
        / "video_3.avi"
    )


    # ============================================================
    # 2. Validate required files
    # ============================================================

    print("=" * 60)
    print("CHECKING PROJECT FILES")
    print("=" * 60)

    print(f"Video:        {video_path}")
    print(f"Player model: {player_model_path}")
    print(f"Ball model:   {ball_model_path}")

    # Input video
    if not video_path.exists():
        raise FileNotFoundError(
            f"\nInput video not found:\n{video_path}"
        )

    # Player model
    if not player_model_path.exists():
        raise FileNotFoundError(
            f"\nPlayer model not found:\n"
            f"{player_model_path}"
        )

    # Ball model
    if not ball_model_path.exists():

        print("\nAvailable .pt files:")

        model_directory = (
            BASE_DIR / "model"
        )

        pt_files = list(
            model_directory.glob("*.pt")
        )

        if pt_files:

            for file in pt_files:
                print(
                    f"  - {file.name}"
                )

        else:

            print(
                "  No .pt files found."
            )

        raise FileNotFoundError(
            f"\nBall model not found:\n"
            f"{ball_model_path}\n\n"
            f"Put the correct basketball .pt model "
            f"inside the model directory and update "
            f"ball_model_path."
        )


    # ============================================================
    # 3. Read video
    # ============================================================

    print("\n" + "=" * 60)
    print("READING VIDEO")
    print("=" * 60)

    video_frames = read_video(
        str(video_path)
    )

    if not video_frames:
        raise ValueError(
            "No frames were read from the video."
        )

    print(
        f"Total video frames: "
        f"{len(video_frames)}"
    )


    # ============================================================
    # 4. Initialize Player Tracker
    # ============================================================

    print("\n" + "=" * 60)
    print("INITIALIZING PLAYER TRACKER")
    print("=" * 60)

    player_tracker = PlayerTracker(
        str(player_model_path)
    )

    print(
        "Player tracker initialized."
    )


    # ============================================================
    # 5. Initialize Ball Tracker
    # ============================================================

    print("\n" + "=" * 60)
    print("INITIALIZING BALL TRACKER")
    print("=" * 60)

    ball_tracker = BallTracker(
        str(ball_model_path)
    )

    print(
        "Ball tracker initialized."
    )


    # ============================================================
    # 6. Player Tracking
    # ============================================================

    print("\n" + "=" * 60)
    print("PLAYER TRACKING")
    print("=" * 60)

    player_tracks = (
        player_tracker.get_object_tracks(
            video_frames,
            read_from_stub=True,
            stub_path=str(
                player_stub_path
            )
        )
    )

    print(
        f"Player tracking completed: "
        f"{len(player_tracks)} frames"
    )


    # ============================================================
    # 7. Ball Tracking
    # ============================================================

    print("\n" + "=" * 60)
    print("BALL TRACKING")
    print("=" * 60)

    ball_tracks = (
        ball_tracker.get_object_tracks(
            video_frames,
            read_from_stub=True,
            stub_path=str(
                ball_stub_path
            )
        )
    )

    print(
        f"Ball tracking completed: "
        f"{len(ball_tracks)} frames"
    )


    # ============================================================
    # 8. Team Assignment
    # ============================================================

    print("\n" + "=" * 60)
    print("TEAM ASSIGNMENT")
    print("=" * 60)

    team_assigner = TeamAssigner(
        team_1_class_name="white shirt",
        team_2_class_name="dark blue shirt"
    )

    player_assignment = (
        team_assigner.get_player_teams_across_frames(
            video_frames,
            player_tracks,
            read_from_stub=True,
            stub_path=str(
                player_assignment_stub_path
            )
        )
    )

    print(
        f"Player team assignment completed: "
        f"{len(player_assignment)} frames"
    )

    # Print only the first frame instead of all 117 frames
    if player_assignment:

        print(
            "First frame team assignment:"
        )

        print(
            player_assignment[0]
        )


    # ============================================================
    # 9. Ball Acquisition
    # ============================================================
    #
    # This is currently a placeholder.
    #
    # Later this should determine which player has possession
    # of the basketball.
    #
    # Example:
    #
    # ball_aquisition[frame_num] = player_id
    #
    # ============================================================

    ball_aquisition = [
        None
        for _ in video_frames
    ]


    # ============================================================
    # 10. Initialize Drawers
    # ============================================================

    print("\n" + "=" * 60)
    print("INITIALIZING DRAWERS")
    print("=" * 60)

    player_tracks_drawer = (
        PlayerTracksDrawer()
    )

    ball_tracks_drawer = (
        BallTracksDrawer()
    )


    # ============================================================
    # 11. Draw Player Tracks
    # ============================================================

    print(
        "\nDrawing player tracks..."
    )

    output_video_frames = (
        player_tracks_drawer.draw(
            video_frames,
            player_tracks,
            player_assignment,
            ball_aquisition
        )
    )


    # ============================================================
    # 12. Draw Ball Tracks
    # ============================================================

    print(
        "Drawing ball tracks..."
    )

    output_video_frames = (
        ball_tracks_drawer.draw(
            output_video_frames,
            ball_tracks
        )
    )


    # ============================================================
    # 13. Validate output frames
    # ============================================================

    if not output_video_frames:

        raise ValueError(
            "Output video frames are empty."
        )

    print(
        f"Output frames: "
        f"{len(output_video_frames)}"
    )


    # ============================================================
    # 14. Save Output Video
    # ============================================================

    print("\n" + "=" * 60)
    print("SAVING OUTPUT VIDEO")
    print("=" * 60)

    output_video_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    save_video(
        output_video_frames,
        str(output_video_path)
    )

    print(
        "\nOutput video saved successfully:"
    )

    print(
        output_video_path
    )


    # ============================================================
    # 15. Pipeline completed
    # ============================================================

    print("\n" + "=" * 60)
    print("BASKETBALL INTELLIGENCE PIPELINE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()