from pathlib import Path

from drawers import (
    PlayerTracksDrawer,
    BallTracksDrawer,
    TeamBallControlDrawer,
    PassInterceptionDrawer
)

from trackers import (
    PlayerTracker,
    BallTracker
)

from utils import (
    read_video,
    save_video
)

from team_assigner import TeamAssigner
from ball_aquisition import BallAquisitionDetector
from pass_and_interception_detector import (
    PassAndInterceptionDetector
)


# ================================================================
# FRAME COUNT VALIDATION
# ================================================================

def validate_frame_count(
    frames,
    expected_count,
    stage_name
):
    """
    Validate that a processing stage returns the expected
    number of frames.
    """

    if frames is None:

        raise ValueError(
            f"{stage_name} returned None."
        )

    if len(frames) != expected_count:

        raise ValueError(
            f"\n{stage_name} changed the number of frames.\n"
            f"Expected: {expected_count}\n"
            f"Received: {len(frames)}"
        )


# ================================================================
# RESTORE FRAME COUNT
# ================================================================

def restore_frame_count(
    processed_frames,
    original_frames,
    stage_name
):
    """
    Ensure that drawing stages return exactly the same number
    of frames as the original video.

    IMPORTANT:
    Your PassInterceptionDrawer intentionally skips frame 0.
    Therefore, when a drawer returns N-1 frames, this function
    restores the ORIGINAL FIRST FRAME instead of duplicating
    the last frame.
    """

    if processed_frames is None:

        raise ValueError(
            f"{stage_name} returned None."
        )

    expected_count = len(original_frames)
    current_count = len(processed_frames)

    # ------------------------------------------------------------
    # Correct number
    # ------------------------------------------------------------

    if current_count == expected_count:

        return processed_frames

    # ------------------------------------------------------------
    # One frame missing
    #
    # Your drawer skips frame 0:
    #
    # for frame_num, frame in enumerate(video_frames):
    #     if frame_num == 0:
    #         continue
    #
    # Therefore restore original frame 0.
    # ------------------------------------------------------------

    if current_count == expected_count - 1:

        print(
            f"WARNING: {stage_name} returned "
            f"{current_count} frames instead of "
            f"{expected_count}."
        )

        print(
            f"Restoring original first frame."
        )

        return [
            original_frames[0].copy()
        ] + processed_frames

    # ------------------------------------------------------------
    # More frames than expected
    # ------------------------------------------------------------

    if current_count > expected_count:

        print(
            f"WARNING: {stage_name} returned "
            f"{current_count} frames instead of "
            f"{expected_count}."
        )

        print(
            "Trimming extra frames."
        )

        return processed_frames[
            :expected_count
        ]

    # ------------------------------------------------------------
    # Fewer frames
    # ------------------------------------------------------------

    print(
        f"WARNING: {stage_name} returned "
        f"{current_count} frames instead of "
        f"{expected_count}."
    )

    print(
        "Restoring missing frames."
    )

    if current_count > 0:

        missing_count = (
            expected_count - current_count
        )

        for i in range(missing_count):

            if i < len(original_frames):

                processed_frames.append(
                    original_frames[
                        current_count + i
                    ].copy()
                )

            else:

                processed_frames.append(
                    processed_frames[-1].copy()
                )

        return processed_frames

    # ------------------------------------------------------------
    # Zero frames
    # ------------------------------------------------------------

    print(
        f"WARNING: {stage_name} returned zero frames."
    )

    return [
        frame.copy()
        for frame in original_frames
    ]


# ================================================================
# CONVERT PASS EVENTS TO FRAME-WISE ARRAY
# ================================================================

def convert_pass_events_to_frame_array(
    passes,
    total_frames
):
    """
    Convert pass event dictionaries into a frame-wise list.

    Input example:

        [
            {
                "frame": 20,
                "from_player": 5,
                "to_player": 7,
                "team": 1
            },
            {
                "frame": 45,
                "from_player": 7,
                "to_player": 3,
                "team": 2
            }
        ]

    Output:

        [0, 0, ..., 1, ..., 2, ...]
    """

    pass_array = [
        0
        for _ in range(total_frames)
    ]

    if passes is None:

        return pass_array

    for event in passes:

        # --------------------------------------------------------
        # Make sure event is a dictionary
        # --------------------------------------------------------

        if not isinstance(
            event,
            dict
        ):

            continue

        frame_num = event.get(
            "frame"
        )

        team = event.get(
            "team"
        )

        # --------------------------------------------------------
        # Validate frame number
        # --------------------------------------------------------

        if frame_num is None:

            continue

        try:

            frame_num = int(
                frame_num
            )

        except (
            TypeError,
            ValueError
        ):

            continue

        if (
            frame_num < 0
            or frame_num >= total_frames
        ):

            continue

        # --------------------------------------------------------
        # Team 1 / Team 2
        # --------------------------------------------------------

        if team in (1, 2):

            pass_array[
                frame_num
            ] = team

    return pass_array


# ================================================================
# CONVERT INTERCEPTION EVENTS TO FRAME-WISE ARRAY
# ================================================================

def convert_interception_events_to_frame_array(
    interceptions,
    total_frames
):
    """
    Convert interception event dictionaries into a frame-wise
    list expected by PassInterceptionDrawer.

    Example output:

        [0, 0, 1, 0, 2, 0, ...]
    """

    interception_array = [
        0
        for _ in range(total_frames)
    ]

    if interceptions is None:

        return interception_array

    for event in interceptions:

        if not isinstance(
            event,
            dict
        ):

            continue

        frame_num = event.get(
            "frame"
        )

        # --------------------------------------------------------
        # For interceptions, use the team that gains possession.
        # --------------------------------------------------------

        team = event.get(
            "to_team"
        )

        if team is None:

            team = event.get(
                "team"
            )

        if frame_num is None:

            continue

        try:

            frame_num = int(
                frame_num
            )

        except (
            TypeError,
            ValueError
        ):

            continue

        if (
            frame_num < 0
            or frame_num >= total_frames
        ):

            continue

        if team in (1, 2):

            interception_array[
                frame_num
            ] = team

    return interception_array


# ================================================================
# MAIN
# ================================================================

def main():

    # ============================================================
    # 1. PROJECT PATHS
    # ============================================================

    BASE_DIR = Path(
        __file__
    ).resolve().parent

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

    output_video_path = (
        BASE_DIR
        / "output_video"
        / "video_7.avi"
    )


    # ============================================================
    # 2. CHECK FILES
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

    validate_frame_count(
        player_tracks,
        total_frames,
        "PlayerTracker"
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

    validate_frame_count(
        ball_tracks,
        total_frames,
        "BallTracker"
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

    validate_frame_count(
        player_assignment,
        total_frames,
        "TeamAssigner"
    )

    print(
        f"Player team assignment completed: "
        f"{len(player_assignment)} frames"
    )

    if player_assignment:

        print(
            "First frame team assignment:"
        )

        print(
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
            "Ball possession detector returned None."
        )

    validate_frame_count(
        ball_aquisition,
        total_frames,
        "BallAquisitionDetector"
    )

    print(
        "Ball possession detection completed."
    )


    # ============================================================
    # 10. PASS AND INTERCEPTION DETECTION
    # ============================================================

    print("\n" + "=" * 70)
    print("PASS AND INTERCEPTION DETECTION")
    print("=" * 70)

    pass_and_interception_detector = (
        PassAndInterceptionDetector()
    )

    # ------------------------------------------------------------
    # Detect passes
    # ------------------------------------------------------------

    passes = (
        pass_and_interception_detector.detect_passes(
            player_tracks,
            ball_tracks,
            player_assignment,
            ball_aquisition
        )
    )

    # ------------------------------------------------------------
    # Detect interceptions
    # ------------------------------------------------------------

    interceptions = (
        pass_and_interception_detector.detect_interceptions(
            player_tracks,
            ball_tracks,
            player_assignment,
            ball_aquisition
        )
    )

    if passes is None:

        passes = []

    if interceptions is None:

        interceptions = []

    print(
        f"Pass events detected: "
        f"{len(passes)}"
    )

    print(
        f"Interception events detected: "
        f"{len(interceptions)}"
    )


    # ============================================================
    # 11. CONVERT EVENTS TO FRAME ARRAYS
    # ============================================================

    print("\n" + "=" * 70)
    print("CONVERTING PASS / INTERCEPTION EVENTS")
    print("=" * 70)

    pass_array = (
        convert_pass_events_to_frame_array(
            passes,
            total_frames
        )
    )

    interception_array = (
        convert_interception_events_to_frame_array(
            interceptions,
            total_frames
        )
    )

    validate_frame_count(
        pass_array,
        total_frames,
        "Pass event array"
    )

    validate_frame_count(
        interception_array,
        total_frames,
        "Interception event array"
    )

    print(
        "Pass event array created."
    )

    print(
        "Interception event array created."
    )

    print(
        f"Team 1 pass frames: "
        f"{pass_array.count(1)}"
    )

    print(
        f"Team 2 pass frames: "
        f"{pass_array.count(2)}"
    )

    print(
        f"Team 1 interception frames: "
        f"{interception_array.count(1)}"
    )

    print(
        f"Team 2 interception frames: "
        f"{interception_array.count(2)}"
    )


    # ============================================================
    # 12. INITIALIZE DRAWERS
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

    pass_interception_drawer = (
        PassInterceptionDrawer()
    )

    print(
        "All drawers initialized successfully."
    )


    # ============================================================
    # 13. DRAW PLAYER TRACKS
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
    # 14. DRAW BALL TRACKS
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
    # 15. DRAW TEAM BALL CONTROL
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
    # 16. DRAW PASS AND INTERCEPTION STATISTICS
    # ============================================================

    print(
        "\nDrawing pass and interception statistics..."
    )

    frames_before_pass_drawer = (
        output_video_frames
    )

    output_video_frames = (
        pass_interception_drawer.draw(
            output_video_frames,
            pass_array,
            interception_array
        )
    )

    output_video_frames = (
        restore_frame_count(
            output_video_frames,
            frames_before_pass_drawer,
            "PassInterceptionDrawer"
        )
    )

    print(
        f"Pass/interception statistics drawn: "
        f"{len(output_video_frames)} frames"
    )


    # ============================================================
    # 17. FINAL VALIDATION
    # ============================================================

    print("\n" + "=" * 70)
    print("FINAL OUTPUT VALIDATION")
    print("=" * 70)

    if not output_video_frames:

        raise ValueError(
            "Output video frames are empty."
        )

    validate_frame_count(
        output_video_frames,
        total_frames,
        "Final Pipeline"
    )

    print(
        f"Input frames : "
        f"{total_frames}"
    )

    print(
        f"Output frames: "
        f"{len(output_video_frames)}"
    )

    print(
        "Frame count validation passed."
    )


    # ============================================================
    # 18. SAVE VIDEO
    # ============================================================

    print("\n" + "=" * 70)
    print("SAVING OUTPUT VIDEO")
    print("=" * 70)

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
    # 19. FINAL SUMMARY
    # ============================================================

    print("\n" + "=" * 70)
    print(
        "BASKETBALL INTELLIGENCE PIPELINE COMPLETED"
    )
    print("=" * 70)

    print(
        f"Frames processed : {total_frames}"
    )

    print(
        f"Pass events      : {len(passes)}"
    )

    print(
        f"Interception events: {len(interceptions)}"
    )

    print(
        f"Output video     : {output_video_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()