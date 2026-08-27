from pathlib import Path

from drawers import PlayerTracksDrawer, BallTracksDrawer
from trackers import PlayerTracker, BallTracker
from utils import read_video, save_video
from team_assigner import TeamAssigner
from ball_aquisition import BallAquisitionDetector


def main():

    # ============================================================
    # 1. Project paths
    # ============================================================

    BASE_DIR = Path(__file__).resolve().parent

    video_path = (
        BASE_DIR
        / "input_video"
        / "video_1.mp4"
    )

    player_model_path = (
        BASE_DIR
        / "model"
        / "basketball_player_detection_training.pt"
    )

    ball_model_path = (
        BASE_DIR
        / "model"
        / "basketball_ball_training.pt"
    )

    player_stub_path = (
        BASE_DIR
        / "stubs"
        / "player_tracks_stubs.pkl"
    )

    ball_stub_path = (
        BASE_DIR
        / "stubs"
        / "ball_tracks_stubs.pkl"
    )

    player_assignment_stub_path = (
        BASE_DIR
        / "stubs"
        / "player_assignment_stubs.pkl"
    )

    ball_aquisition_stub_path = (
        BASE_DIR
        / "stubs"
        / "ball_aquisition_stubs.pkl"
    )

    output_video_path = (
        BASE_DIR
        / "output_video"
        / "video_4.avi"
    )


    # ============================================================
    # 2. Validate files
    # ============================================================

    print("=" * 60)
    print("CHECKING PROJECT FILES")
    print("=" * 60)

    print(f"Video:        {video_path}")
    print(f"Player model: {player_model_path}")
    print(f"Ball model:   {ball_model_path}")

    if not video_path.exists():
        raise FileNotFoundError(
            f"Input video not found:\n{video_path}"
        )

    if not player_model_path.exists():
        raise FileNotFoundError(
            f"Player model not found:\n{player_model_path}"
        )

    if not ball_model_path.exists():

        model_directory = BASE_DIR / "model"

        print("\nAvailable .pt files:")

        pt_files = list(
            model_directory.glob("*.pt")
        )

        if pt_files:
            for file in pt_files:
                print(f"  - {file.name}")
        else:
            print("  No .pt files found.")

        raise FileNotFoundError(
            f"\nBall model not found:\n"
            f"{ball_model_path}"
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

    if player_assignment:
        print(
            "First frame team assignment:",
            player_assignment[0]
        )


    # ============================================================
    # 9. Ball Acquisition / Possession
    # ============================================================

    print("\n" + "=" * 60)
    print("BALL POSSESSION DETECTION")
    print("=" * 60)

    ball_aquisition_detector = (
        BallAquisitionDetector()
    )

    # IMPORTANT:
    # Your class method is called:
    #
    # detect_ball_possession()
    #
    # NOT:
    #
    # detector_possession()

    ball_aquisition = (
        ball_aquisition_detector.detect_ball_possession(
            player_tracks,
            ball_tracks
        )
    )

    print(
        "Ball possession detection completed."
    )

    print(
        f"Ball acquisition result type: "
        f"{type(ball_aquisition)}"
    )


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
    # 13. Validate Output
    # ============================================================

    if not output_video_frames:
        raise ValueError(
            "Output video frames are empty."
        )

    if len(output_video_frames) != len(video_frames):
        raise ValueError(
            "Number of output frames does not match "
            "number of input frames."
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
    # 15. Completed
    # ============================================================

    print("\n" + "=" * 60)
    print("BASKETBALL INTELLIGENCE PIPELINE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()