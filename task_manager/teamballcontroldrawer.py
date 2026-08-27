# Imports Path class from standard library pathlib for cross-platform file system path manipulations.
from pathlib import Path

#  Imports visualization drawer classes responsible for rendering player markers, ball indicators, and team possession stats.
from drawers import (PlayerTracksDrawer, BallTracksDrawer, TeamBallControlDrawer)

# Imports object tracking classes designed to detect and track players and the ball across video frames.
from trackers import (PlayerTracker, BallTracker)

# Imports I/O utility functions for decoding video files into image frame lists and encoding them back to disk.
from utils import (read_video, save_video)

# Imports TeamAssigner class for clustering players into opposing teams based on uniform color features.
from team_assigner import TeamAssigner

# Imports BallAquisitionDetector class for determining frame-by-frame ball possession based on bounding box metrics.
from ball_aquisition import BallAquisitionDetector

# Defines validation helper function to ensure standard frame length consistency across pipeline steps.
def validate_frame_count(frames, expected_count, stage_name):
    """
    Validate frame count after each processing stage.
    """

    # Checks if stage output is None.
    if frames is None:

        # Raises ValueError if processing step returned a None object instead of a list.
        raise ValueError(f"{stage_name} returned None.")

    # Checks if total frame count returned by the stage diverges from the expected count.
    if len(frames) != expected_count:

        # Raises ValueError detailing expected vs received frame counts when mismatches occur.
        raise ValueError(f"\n{stage_name} changed the number of frames.\n"
            f"Expected: {expected_count}\n"
            f"Received: {len(frames)}"
        )

# Defines fallback mechanism function that aligns processed frame list length with original input frames.
def restore_frame_count(
    processed_frames,
    original_frames,
    stage_name
):
    """
    Make sure a drawing stage returns exactly the same
    number of frames as the input.
    If one or more frames are missing, the original frame
    at that position is used.
    """

    # Reads total number of expected frames from original frame list.
    expected_count = len(original_frames)

    # Checks if drawer returned a None object.
    if processed_frames is None:

        # Raises ValueError indicating drawer failed completely and returned None.
        raise ValueError(f"{stage_name} returned None.")

    # Gets frame count produced by current drawer stage.
    current_count = len(processed_frames)

    # ------------------------------------------------------------
    # Correct number of frames
    # ------------------------------------------------------------
    # Evaluates if frame counts match perfectly.
    if current_count == expected_count:
        # Returns frame list untouched if no discrepancy exists.
        return processed_frames


    # ------------------------------------------------------------
    # More frames than expected Comment introducing over-allocation frame correction section.
    # ------------------------------------------------------------
    # Evaluates if drawer output produced surplus frames.
    if current_count > expected_count:
        # Prints warning log detailing excess frame count anomaly.
        print(
            f"WARNING: {stage_name} returned "
            f"{current_count} frames instead of "
            f"{expected_count}."
        )

        # Prints console notification for trimming excess tail frames.
        print(f"Trimming extra frames.")

        # Truncates frame list slice to retain only up to expected_count frames.
        return processed_frames[:expected_count]


    # ------------------------------------------------------------
    # Fewer frames than expected Comment introducing missing frame restoration section.
    # ------------------------------------------------------------
    #  Prints warning log displaying deficient frame count anomaly.
    print(
        f"WARNING: {stage_name} returned "
        f"{current_count} frames instead of "
        f"{expected_count}."
    )

    # Prints log message indicating automated pad frame recovery.
    print(f"Restoring missing frames.")

    # Calculates number of missing frames to generate.
    missing_count = (expected_count - current_count)

    # Append copies of the last processed frame  this keeps the video playable and preserves the expected frame count.
    if current_count > 0:

        # Captures last valid processed frame as baseline template for duplication.
        last_frame = processed_frames[-1]

        # Loops for total count of missing frames.
        for _ in range(missing_count):

            # Appends duplicate copy of last frame to fill missing sequence gap.
            processed_frames.append(last_frame.copy())

    # Fallback branch executed if drawer returned 0 frames.
    else:
        # If the drawer returned zero frames fall back completely to the original frames.
        print(
            f"WARNING: {stage_name} returned "
            f"zero frames."
        )

        # Returns deep copies of original unprocessed frames as safety recovery.
        return [frame.copy() for frame in original_frames]

    # Returns padded frame list matching expected length.
    return processed_frames

# Main execution function for running end-to-end sports vision pipeline.
def main():

    # ============================================================
    # 1. PROJECT PATHS
    # ============================================================
    # Resolves absolute path to script's current parent directory.
    BASE_DIR = Path(__file__).resolve().parent

    # Defines relative path to input raw video file.
    video_path = (BASE_DIR / "input_video" / "video_1.mp4"
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

    output_video_path = (
        BASE_DIR
        / "output_video"
        / "video_5.avi"
    )


    # ============================================================
    # 2. CHECK PROJECT FILES
    # ============================================================

    print("=" * 70)
    print("CHECKING PROJECT FILES")
    print("=" * 70)

    print(
        f"Video        : {video_path}"
    )

    print(
        f"Player model : {player_model_path}"
    )

    print(
        f"Ball model   : {ball_model_path}"
    )


    if not video_path.exists():

        raise FileNotFoundError(
            f"\nInput video not found:\n"
            f"{video_path}"
        )


    if not player_model_path.exists():

        raise FileNotFoundError(
            f"\nPlayer model not found:\n"
            f"{player_model_path}"
        )


    if not ball_model_path.exists():

        model_directory = (
            BASE_DIR / "model"
        )

        print(
            "\nAvailable .pt files:"
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
            f"{ball_model_path}"
        )


    # ============================================================
    # 3. READ VIDEO
    # ============================================================

    print("\n" + "=" * 70)
    print("READING VIDEO")
    print("=" * 70)

    video_frames = read_video(
        str(video_path)
    )

    if not video_frames:

        raise ValueError(
            "No frames were read from the video."
        )

    total_frames = len(
        video_frames
    )

    print(
        f"Total video frames: "
        f"{total_frames}"
    )


    # ============================================================
    # 4. INITIALIZE PLAYER TRACKER
    # ============================================================

    print("\n" + "=" * 70)
    print("INITIALIZING PLAYER TRACKER")
    print("=" * 70)

    player_tracker = PlayerTracker(
        str(player_model_path)
    )

    print(
        "Player tracker initialized successfully."
    )


    # ============================================================
    # 5. INITIALIZE BALL TRACKER
    # ============================================================

    print("\n" + "=" * 70)
    print("INITIALIZING BALL TRACKER")
    print("=" * 70)

    ball_tracker = BallTracker(
        str(ball_model_path)
    )

    print(
        "Ball tracker initialized successfully."
    )


    # ============================================================
    # 6. PLAYER TRACKING
    # ============================================================

    print("\n" + "=" * 70)
    print("PLAYER TRACKING")
    print("=" * 70)

    player_tracks = (
        player_tracker.get_object_tracks(
            video_frames,
            read_from_stub=True,
            stub_path=str(
                player_stub_path
            )
        )
    )

    if player_tracks is None:

        raise ValueError(
            "Player tracker returned None."
        )

    if len(player_tracks) != total_frames:

        raise ValueError(
            f"Player tracking returned "
            f"{len(player_tracks)} frames. "
            f"Expected {total_frames}."
        )

    print(
        f"Player tracking completed: "
        f"{len(player_tracks)} frames"
    )


    # ============================================================
    # 7. BALL TRACKING
    # ============================================================

    print("\n" + "=" * 70)
    print("BALL TRACKING")
    print("=" * 70)

    ball_tracks = (
        ball_tracker.get_object_tracks(
            video_frames,
            read_from_stub=True,
            stub_path=str(
                ball_stub_path
            )
        )
    )

    if ball_tracks is None:

        raise ValueError(
            "Ball tracker returned None."
        )

    if len(ball_tracks) != total_frames:

        raise ValueError(
            f"Ball tracking returned "
            f"{len(ball_tracks)} frames. "
            f"Expected {total_frames}."
        )

    print(
        f"Ball tracking completed: "
        f"{len(ball_tracks)} frames"
    )


    # ============================================================
    # 8. TEAM ASSIGNMENT
    # ============================================================

    print("\n" + "=" * 70)
    print("TEAM ASSIGNMENT")
    print("=" * 70)

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

    if player_assignment is None:

        raise ValueError(
            "Team assignment returned None."
        )

    if len(player_assignment) != total_frames:

        raise ValueError(
            f"Team assignment returned "
            f"{len(player_assignment)} frames. "
            f"Expected {total_frames}."
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
    # 9. BALL POSSESSION
    # ============================================================

    print("\n" + "=" * 70)
    print("BALL POSSESSION DETECTION")
    print("=" * 70)

    ball_aquisition_detector = (
        BallAquisitionDetector()
    )

    ball_aquisition = (
        ball_aquisition_detector.detect_ball_possession(
            player_tracks,
            ball_tracks
        )
    )

    if ball_aquisition is None:

        raise ValueError(
            "Ball possession detector "
            "returned None."
        )

    if len(ball_aquisition) != total_frames:

        raise ValueError(
            f"Ball possession returned "
            f"{len(ball_aquisition)} frames. "
            f"Expected {total_frames}."
        )

    print(
        "Ball possession detection completed."
    )


    # ============================================================
    # 10. INITIALIZE DRAWERS
    # ============================================================

    print("\n" + "=" * 70)
    print("INITIALIZING DRAWERS")
    print("=" * 70)

    player_tracks_drawer = (
        PlayerTracksDrawer()
    )

    ball_tracks_drawer = (
        BallTracksDrawer()
    )

    team_ball_control_drawer = (
        TeamBallControlDrawer()
    )

    print(
        "All drawers initialized successfully."
    )


    # ============================================================
    # 11. DRAW PLAYER TRACKS
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

    output_video_frames = (
        restore_frame_count(
            output_video_frames,
            video_frames,
            "PlayerTracksDrawer"
        )
    )

    print(
        f"Player tracks drawn: "
        f"{len(output_video_frames)} frames"
    )


    # ============================================================
    # 12. DRAW BALL TRACKS
    # ============================================================

    print(
        "\nDrawing ball tracks..."
    )

    frames_before_ball_drawer = (
        output_video_frames
    )

    output_video_frames = (
        ball_tracks_drawer.draw(
            output_video_frames,
            ball_tracks
        )
    )

    output_video_frames = (
        restore_frame_count(
            output_video_frames,
            frames_before_ball_drawer,
            "BallTracksDrawer"
        )
    )

    print(
        f"Ball tracks drawn: "
        f"{len(output_video_frames)} frames"
    )


    # ============================================================
    # 13. DRAW TEAM BALL CONTROL
    # ============================================================

    print(
        "\nDrawing team ball control..."
    )

    frames_before_team_drawer = (
        output_video_frames
    )

    output_video_frames = (
        team_ball_control_drawer.draw(
            output_video_frames,
            player_assignment,
            ball_aquisition
        )
    )

    output_video_frames = (
        restore_frame_count(
            output_video_frames,
            frames_before_team_drawer,
            "TeamBallControlDrawer"
        )
    )

    print(
        f"Team ball control drawn: "
        f"{len(output_video_frames)} frames"
    )


    # ============================================================
    # 14. FINAL VALIDATION
    # ============================================================

    print("\n" + "=" * 70)
    print("FINAL OUTPUT VALIDATION")
    print("=" * 70)

    if not output_video_frames:

        raise ValueError(
            "Output video frames are empty."
        )

    if len(output_video_frames) != total_frames:

        raise ValueError(
            f"Final output contains "
            f"{len(output_video_frames)} frames. "
            f"Expected {total_frames}."
        )

    print(
        f"Input frames : {total_frames}"
    )

    print(
        f"Output frames: "
        f"{len(output_video_frames)}"
    )

    print(
        "Frame count validation passed."
    )


    # ============================================================
    # 15. CREATE OUTPUT DIRECTORY
    # ============================================================

    print("\n" + "=" * 70)
    print("SAVING OUTPUT VIDEO")
    print("=" * 70)

    output_video_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    # ============================================================
    # 16. SAVE VIDEO
    # ============================================================

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
    # 17. COMPLETED
    # ============================================================

    print("\n" + "=" * 70)
    print(
        "BASKETBALL INTELLIGENCE PIPELINE COMPLETED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()