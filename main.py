# Imports Path class from Python's standard library pathlib module for object-oriented filesystem path operations.
from pathlib import Path

# Imports graphic visualization drawer modules for rendering player circles, ball markers, possession stats, pass event highlights, court keypoints, 2D mini-maps, and speed/distance overlays.
from drawers import (PlayerTracksDrawer, BallTracksDrawer, TeamBallControlDrawer,
                     PassInterceptionDrawer, CourtKeypointDrawer, TacticalViewDrawer,
                     SpeedAndDistanceDrawer)

# Imports deep learning object tracking modules to detect and follow players and the ball across video frames.
from trackers import (PlayerTracker, BallTracker)

# Imports video I/O helper functions to decode raw video files into image lists and save processed frames to disk.
from utils import (read_video, save_video)

# Imports TeamAssigner class for grouping players into opposing teams based on uniform color analysis.
from team_assigner import TeamAssigner

# Imports BallAquisitionDetector class to identify frame-by-frame ball possession based on player proximity.
from ball_aquisition import (BallAquisitionDetector)

# Imports PassAndInterceptionDetector class to detect completed passes, turnovers, and interceptions.
from pass_and_interception_detector import (PassAndInterceptionDetector)

# Imports CourtKeypointDetector class for locating anatomical court landmarks used in homography mapping.
from court_keypoint_detector import (CourtKeypointDetector)

# Imports TacticalViewConverter class for projecting player and ball coordinates onto a 2D top-down tactical court layout.
from tactical_view_converter import (TacticalViewConverter)

# Imports SpeedAndDistanceCalculator class for computing real-world movement speed and total covered distance.
from speed_and_distance_calculator import (SpeedAndDistanceCalculator)


# ================================================================
# FRAME COUNT VALIDATION : Section header denoting standard data frame length validation utility function.
# ================================================================
# Defines helper function to confirm that a given processing stage outputs the expected number of frames.
def validate_frame_count(frames, expected_count, stage_name,):
    """
    Validate that a processing stage returns the expected number of frames.
    """

    # Checks if the processed frame output is a None object.
    if frames is None:
        # Raises ValueError indicating that the specified pipeline stage returned None instead of a frame list.
        raise ValueError(f"{stage_name} returned None.")

    # Checks if the length of the frame list deviates from the required baseline frame count.
    if len(frames) != expected_count:
        # Raises detailed ValueError displaying expected vs received frame counts when mismatches occur.
        raise ValueError(
            f"\n{stage_name} changed the number of frames.\n"
            f"Expected: {expected_count}\n"
            f"Received: {len(frames)}"
        )


# ================================================================
# RESTORE FRAME COUNT : Section header for frame count fallback recovery function.
# ================================================================
# Defines fallback mechanism function that forces a drawer's output frame list to match original input frame count.
def restore_frame_count(processed_frames, original_frames, stage_name):
    """
    Ensure drawing stages return exactly the same number of frames as their input.
    """

    # Evaluates if drawer stage output is a None object.
    if processed_frames is None:
        # Raises ValueError confirming that the drawer output was None.
        raise ValueError(f"{stage_name} returned None.")

    # Reads required benchmark frame count from original frame array.
    expected_count = len(original_frames)

    # Measures current frame count produced by the drawer stage.
    current_count = len(processed_frames)

    # ------------------------------------------------------------
    # Correct number of frames : Comment block indicating exact match check step.
    # ------------------------------------------------------------
    # Evaluates if processed frame count matches the input frame count.
    if current_count == expected_count:
        # Returns processed frame list without modifications.
        return processed_frames

    # ------------------------------------------------------------
    # One frame missing : Comment block indicating single-frame omission recovery step.
    # ------------------------------------------------------------
    # Checks if output list is missing exactly one frame.
    if current_count == expected_count - 1:

        # Logs warning detailing the single missing frame discrepancy.
        print(f"WARNING: {stage_name} returned "
            f"{current_count} frames instead of "
            f"{expected_count}."
        )

        # Logs notification that original first frame will be prepended.
        print("Restoring original first frame.")

        # Prepends a copy of original input frame at index 0 to repair missing starting frame.
        return [original_frames[0].copy()] + processed_frames

    # ------------------------------------------------------------
    # Too many frames : Comment block indicating over-allocation trim recovery step.
    # ------------------------------------------------------------
    # Checks if drawer returned extra frames beyond the expected length.
    if current_count > expected_count:

        # Logs warning detailing excess frame output anomaly.
        print(
            f"WARNING: {stage_name} returned "
            f"{current_count} frames instead of "
            f"{expected_count}."
        )

        # Logs status message that surplus ending frames will be trimmed.
        print("Trimming extra frames.")

        # Truncates frame list slice to retain only up to expected_count frames.
        return processed_frames[:expected_count]

    # ------------------------------------------------------------
    # Fewer frames : Comment block indicating generic multi-frame restoration branch.
    # ------------------------------------------------------------
    # Logs warning detailing deficient frame output count anomaly.
    print(
        f"WARNING: {stage_name} returned "
        f"{current_count} frames instead of "
        f"{expected_count}."
    )

    # Logs notification that missing frames are being restored to restore length parity.
    print("Restoring missing frames.")

    # Checks if output list contains at least one valid processed frame.
    if current_count > 0:

        # Computes number of missing frames required to match expected count.
        missing_count = (expected_count - current_count)

        # Loops for each missing frame index position needed.
        for i in range(missing_count):

            # Calculates index position to map from original source frames.
            source_index = (current_count + i)

            # Evaluates if calculated target index falls within valid bounds of original frames.
            if source_index < expected_count:

                # Appends copy of original frame at source index position to fill missing gap.
                processed_frames.append(original_frames[source_index].copy())

            # Fallback branch executed if source index exceeds original frame array limits.
            else:

                # Appends duplicate copy of last available processed frame.
                processed_frames.append(processed_frames[-1].copy())

        # Returns output frame list restored to full expected length.
        return processed_frames

    # ------------------------------------------------------------
    # Zero frames : Section header for handling cases where a drawer stage returns an empty list of frames.
    # ------------------------------------------------------------
    #  Prints warning message to console indicating that the drawer function returned zero frames.
    print(
        f"WARNING: {stage_name} returned "
        f"zero frames."
    )

    # Returns a list containing deep copies of all original input frames as a complete fallback.
    return [frame.copy() for frame in original_frames]

# ================================================================
# PASS EVENTS : Section header for converting discrete pass event dictionaries into a frame-by-frame lookup array.
# ================================================================
# Defines helper function to map pass event metadata objects across video frames into an indexed list.
def convert_pass_events_to_frame_array(passes, total_frames,):
    """
    Convert pass events into a frame-wise array.
    0 = no pass
    1 = Team 1 pass
    2 = Team 2 pass
    """
    # Initializes list of zeros matching video frame length to represent default state (0 = no pass).
    pass_array = [0 for _ in range(total_frames)]

    # Returns default zero array if pass events parameter is None.
    if passes is None:
        return pass_array

    # Iterates over each event item inside the passes collection.
    for event in passes:

        # Skips current item if it is not a valid dictionary object.
        if not isinstance(event, dict):
            continue

        # Retrieves frame number value associated with current pass event key.
        frame_num = event.get("frame")

        # Retrieves team identifier (1 or 2) associated with pass event.
        team = event.get("team")

        # Skips current iteration if frame number attribute is missing.
        if frame_num is None:
            continue

        # Casts frame number to integer type safely.
        try:
            frame_num = int(frame_num)

        # Skips iteration if frame number cannot be parsed to an integer.
        except (TypeError, ValueError):
            continue

        # Skips iteration if frame number index lies outside valid video bounds.
        if (frame_num < 0 or frame_num >= total_frames):
            continue

        # Validates that team identifier is strictly Team 1 or Team 2.
        if team in (1, 2):

            # Assigns team ID to target index position within frame array.
            pass_array[frame_num] = team

    # Returns frame-wise pass state array.
    return pass_array


# ================================================================
# INTERCEPTION EVENTS : Section header for converting interception event dictionaries into a frame-by-frame array.
# ================================================================
# Defines helper function to map interception event objects to frame-indexed array entries.
def convert_interception_events_to_frame_array(interceptions, total_frames):
    """
    Convert interception events into a frame-wise array.
    0 = no interception
    1 = Team 1 interception
    2 = Team 2 interception
    """
    # Initializes list of zeros with length matching frame count to represent default state (0 = no interception).
    interception_array = [0 for _ in range(total_frames)]

    # Returns zero array immediately if interceptions input is None.
    if interceptions is None:
        return interception_array

    # Iterates over each item in interceptions list.
    for event in interceptions:

        # Skips iteration if current interception entry is not a dictionary.
        if not isinstance(event, dict):
            continue

        # Extracts frame index corresponding to interception event.
        frame_num = event.get("frame")

        # Extracts target team index capturing the interception.
        team = event.get("to_team")

        # Falls back to alternative 'team' key if 'to_team' key is missing.
        if team is None:
            team = event.get("team")

        # Skips event if frame index value is missing.
        if frame_num is None:
            continue

        # Converts frame index string or float into integer type.
        try:
            frame_num = int(frame_num)

        # Skips event processing if frame index conversion fails.
        except (TypeError, ValueError):
            continue

        # Ignores events that reference frame indices outside valid range.
        if (frame_num < 0 or frame_num >= total_frames):
            continue

        # Confirms that team ID is valid (1 or 2).
        if team in (1, 2):

            # Records team ID at specified frame index in output array.
            interception_array[frame_num] = team

    # Returns complete frame-indexed interception status array.
    return interception_array


# ================================================================
# COURT KEYPOINT DETECTION : Section header for court keypoint extraction helper function.
# ================================================================
# Defines helper function to extract keypoint markers across all video frames using the provided detector object.
def detect_court_keypoints(detector, video_frames,):
    """
    Detect court keypoints using:

        CourtKeypointDetector.get_court_keypoints()
    """

    # Checks if the provided court keypoint detector instance is None.
    if detector is None:

        # Raises ValueError confirming detector instance missing.
        raise ValueError("CourtKeypointDetector is None.")

    # Evaluates if detector object lacks the required 'get_court_keypoints' method attribute.
    if not hasattr(detector, "get_court_keypoints",):

        # Inspects and filters non-private public methods on the detector object for debugging.
        available_methods = [name for name in dir(detector) if not name.startswith("_")]

        # Raises detailed AttributeError showing all alternative public methods available on the object.
        raise AttributeError(
            "\nCourtKeypointDetector does not expose "
            "get_court_keypoints().\n\n"
            "Available public methods:\n"
            + "\n".join(
                f"  - {name}"
                for name in available_methods
            )
        )

    # Prints diagnostic log indicating method execution.
    print("Using CourtKeypointDetector.get_court_keypoints()")

    # Runs keypoint detection algorithm across input video frames.
    court_keypoints = (detector.get_court_keypoints(video_frames))

    # Checks if detection method returned a None object.
    if court_keypoints is None:
        # Raises ValueError indicating court keypoint detection failure.
        raise ValueError("get_court_keypoints() returned None.")

    # Returns extracted frame-by-frame court keypoint data structure.
    return court_keypoints


# ================================================================
# GET TACTICAL PLAYER POSITIONS : Section header for coordinate transformation to tactical 2D court layout.
# ================================================================
# Defines helper function to project pixel player bounding boxes into tactical top-down 2D court positions.
def get_tactical_player_positions(tactical_view_converter, court_keypoints, player_tracks):
    """
    Convert player tracks into tactical-view coordinates.
    TacticalViewConverter exposes:
        transform_players_to_tactical_view(keypoints_list, player_tracks)
    Therefore BOTH court_keypoints and player_tracks must be supplied.
    """

    # Checks if tactical view converter instance is None.
    if tactical_view_converter is None:
        # Raises error confirming tactical view converter instance is missing.
        raise ValueError("TacticalViewConverter is None.")

    # Checks if input court keypoints object is None.
    if court_keypoints is None:
        # Raises error confirming missing court keypoint baseline data.
        raise ValueError("court_keypoints is None.")

    # Checks if input player tracking data is None.
    if player_tracks is None:
        # Raises error confirming missing player track input data.
        raise ValueError("player_tracks is None.")

    # ------------------------------------------------------------
    # Correct API : Comment introducing API validation step.
    # ------------------------------------------------------------
    # Confirms that converter instance possesses the expected transformation method.
    if hasattr(tactical_view_converter, "transform_players_to_tactical_view",):

        # Fetches transformation method dynamically from converter object.
        method = getattr(tactical_view_converter, "transform_players_to_tactical_view")

        # Checks whether fetched method attribute is an executable function.
        if not callable(method):
            # Raises error if method attribute exists but cannot be called as a function.
            raise TypeError("transform_players_to_tactical_view  exists but is not callable.")

        # Prints diagnostic confirmation message to console.
        print("Using TacticalViewConverter transform_players_to_tactical_view()")

        #  Invokes transformation method passing court keypoints and player tracks.
        try:
            result = method(court_keypoints, player_tracks)

        # Catches any error during coordinate homography matrix transformation.
        except Exception as error:
            # Wraps original transformation exception in contextual RuntimeError with parameter details.
            raise RuntimeError(
                "\nFailed to transform player tracks "
                "to tactical-view coordinates.\n\n"
                "Correct API requires:\n"
                "  transform_players_to_tactical_view("
                "court_keypoints, player_tracks)\n\n"
                f"Original error:\n{error}"
            ) from error

        # Checks if coordinate transformation output is None.
        if result is None:

            # Raises ValueError confirming transformation produced no data.
            raise ValueError(
                "transform_players_to_tactical_view() "
                "returned None."
            )

        # Returns transformed 2D tactical coordinate positions for players across frames.
        return result

    # ------------------------------------------------------------
    # Fallback APIs : Section header for discovering and attempting alternative method signatures on TacticalViewConverter.
    # ------------------------------------------------------------
    # Tuple of alternative method names to probe if the primary transformation function is missing.
    possible_methods = ("transform_player_positions", "convert_player_positions",
                        "get_tactical_player_positions","convert_to_tactical_view",
                        "transform_points","transform")

    # Iterates over each candidate method name in the fallback tuple.
    for method_name in possible_methods:
        # Skips to next candidate if current method name does not exist on the converter object.
        if not hasattr(tactical_view_converter, method_name):
            continue

        # Dynamic reflection: fetches reference to method attribute from the converter instance.
        method = getattr(tactical_view_converter, method_name)

        # Skips iteration if the matching attribute exists but is not an executable function.
        if not callable(method):
            continue

        # Logs diagnostic status message identifying fallback method being executed.
        print(
            "Trying tactical player-position "
            f"method: {method_name}"
        )

        # Attempts to run candidate fallback method passing court keypoints and player tracks.
        try:

            # Attempts to run candidate fallback method passing court keypoints and player tracks.
            result = method(court_keypoints, player_tracks)

            # Returns transformed tactical position data immediately if successful and non-null.
            if result is not None:
                return result
        except Exception as error:

            # Logs non-fatal warning capturing exception details when fallback method attempt fails.
            print(
                f"WARNING: {method_name}() failed: "
                f"{error}"
            )

    # Introspects and lists all public methods/attributes on converter instance for debugging diagnostics.
    available_methods = [name for name in dir(tactical_view_converter) if not name.startswith("_")]

    # Raises comprehensive AttributeError showing expected signature along with all discovered public attributes.
    raise AttributeError(
        "\nCould not convert player tracks "
        "to tactical-view coordinates.\n\n"
        "Expected method:\n"
        "  - transform_players_to_tactical_view("
        "keypoints_list, player_tracks)\n\n"
        "Available public attributes/methods:\n"
        + "\n".join(
            f"  - {name}"
            for name in available_methods
        )
    )


# ================================================================
# SPEED AND DISTANCE : Section header for player speed and covered distance calculation pipeline stage.
# ================================================================
# Defines helper function to calculate real-world movement speed (km/h) and distance (m) per player across frames.
def calculate_player_speed_and_distance(tactical_view_converter,
    court_keypoints,player_tracks):
    """
    Convert player tracks to tactical coordinates and calculate player distance and speed.
    """

    # Displays visual banner initiating speed and distance calculation sequence.
    print("\n" + "=" * 70)
    print("SPEED AND DISTANCE CALCULATION")
    print("=" * 70)

    # ------------------------------------------------------------
    # Validate inputs : Comment block introducing initial parameter validation checks.
    # ------------------------------------------------------------
    # Raises ValueError if tactical view converter object reference is missing.
    if tactical_view_converter is None:
        raise ValueError("TacticalViewConverter is None.")

    # Raises ValueError if court keypoint data structure is missing.
    if court_keypoints is None:
        raise ValueError("court_keypoints is None.")

    # Raises ValueError if tracked player coordinates data structure is missing.
    if player_tracks is None:
        raise ValueError("player_tracks is None.")

    # ------------------------------------------------------------
    # Required converter attributes : Comment block introducing court metric attribute checks on converter instance.
    # ------------------------------------------------------------
    # Tuple detailing required image-space pixel dimensions and physical metric court dimension attributes.
    required_attributes = ("width", "height", "actual_width_in_meters", "actual_height_in_meters")

    # Filters list to identify any mandatory metadata attributes missing from the converter instance.
    missing_attributes = [attribute for attribute in required_attributes if not hasattr(
        tactical_view_converter, attribute)]

    # Raises AttributeError listing all missing dimension properties needed for scale conversion.
    if missing_attributes:
        raise AttributeError("TacticalViewConverter is missing required attributes: "  + "\n".join(f"  - {attribute}" for attribute in missing_attributes))

    # ------------------------------------------------------------
    # Initialize calculator : Comment block introducing calculation engine initialization step.
    # ------------------------------------------------------------
    # Instantiates SpeedAndDistanceCalculator passing pixel dimensions and real-world meter metrics for spatial scaling.
    calculator = (SpeedAndDistanceCalculator(tactical_view_converter.width,tactical_view_converter.height,
                                             tactical_view_converter.actual_width_in_meters,
                                             tactical_view_converter.actual_height_in_meters))

    # ------------------------------------------------------------
    # Transform players : Comment block introducing 2D tactical projection step.
    # ------------------------------------------------------------
    # Prints log notification indicating tactical perspective transformation start.
    print("\nTransforming player positions to tactical-view coordinates...")

    # Invokes get_tactical_player_positions to convert raw pixel tracking bounding boxes to 2D tactical court positions.
    tactical_player_positions = (get_tactical_player_positions(tactical_view_converter,court_keypoints, player_tracks))

    # Raises ValueError if coordinate transformation process returns None instead of position arrays.
    if tactical_player_positions is None:
        raise ValueError("Tactical player positions are None.")

    # ------------------------------------------------------------
    # Validate frame count : Comment block introducing frame count verification between tactical positions and raw tracks.
    # ------------------------------------------------------------
    # Evaluates whether total frame count of converted tactical positions matches original player tracks length.
    if len(tactical_player_positions) != len(player_tracks):
        # Raises detailed ValueError displaying player tracks count versus tactical positions count upon mismatch.
        raise ValueError(
            "\nTactical player position frame "
            "count mismatch.\n"
            f"Player tracks: "
            f"{len(player_tracks)}\n"
            f"Tactical positions: "
            f"{len(tactical_player_positions)}"
        )

    # Prints diagnostic status confirming success of tactical position transformation.
    print("Tactical player position ", "transformation completed.")

    # ------------------------------------------------------------
    # Distance : Comment block introducing real-world movement distance calculation phase.
    # ------------------------------------------------------------
    # Prints console status log indicating player distance calculations have started.
    print("\nCalculating player distance...")

    # Calls calculator method to measure incremental displacement in meters frame-by-frame for each player.
    try:
        player_distance_per_frame = (calculator.calculate_distance(tactical_player_positions))

    #  Catches runtime exceptions during calculation and wraps them inside an informative RuntimeError.
    except Exception as error:
        raise RuntimeError(
            "Player distance calculation failed:\n"
            f"{error}"
        ) from error

    # Raises ValueError if distance computation returned a None object instead of a data structure.
    if player_distance_per_frame is None:
        raise ValueError("calculate_distance() returned None.")

    # Prints log message confirming completed calculation of player distances.
    print("Player distance calculation completed.")

    # ------------------------------------------------------------
    # Speed : Comment block introducing player speed calculation phase.
    # ------------------------------------------------------------
    # Prints console status log indicating player speed calculations have started.
    print("\nCalculating player speed...")
    # Calculates player movement velocity (e.g. km/h) across frames based on incremental distance data.
    try:
        player_speed_per_frame = (calculator.calculate_speed(player_distance_per_frame))

    # Wraps speed calculation exceptions inside contextual RuntimeError detailing the failure cause.
    except Exception as error:
        raise RuntimeError(
            "Player speed calculation failed:\n"
            f"{error}"
        ) from error

    # Raises ValueError if speed calculation method returned None.
    if player_speed_per_frame is None:
        raise ValueError("calculate_speed() returned None.")

    # Prints status confirmation for completed player speed calculation.
    print("Player speed calculation completed.")

    # ------------------------------------------------------------
    # Validate distance : Comment block introducing distance result frame length validation.
    # ------------------------------------------------------------
    # Checks whether distance array length matches input player tracks frame length.
    if len(player_distance_per_frame) != len(player_tracks):
        # Raises ValueError indicating frame length divergence in calculated distance metrics.
        raise ValueError(
            "\nDistance result frame count "
            "mismatch.\n"
            f"Player tracks: "
            f"{len(player_tracks)}\n"
            f"Distance: "
            f"{len(player_distance_per_frame)}"
        )

    # ------------------------------------------------------------
    # Validate speed : Comment block introducing speed result frame length validation.
    # ------------------------------------------------------------
    # Checks whether speed array length matches input player tracks frame length.
    if len(player_speed_per_frame) != len(player_tracks):
        # Raises ValueError indicating frame length divergence in calculated speed metrics.
        raise ValueError(
            "\nSpeed result frame count "
            "mismatch.\n"
            f"Player tracks: "
            f"{len(player_tracks)}\n"
            f"Speed: "
            f"{len(player_speed_per_frame)}"
        )

    # Returns calculated frame-wise player distance and speed data tuples.
    return (player_distance_per_frame,player_speed_per_frame)


# ================================================================
# MAIN : Section header for primary script execution function.
# ================================================================
# Defines main function encapsulating project setup and complete execution pipeline.
def main():
    # ============================================================
    # 1. PROJECT PATHS : Section header for defining file paths used across models, stubs, and media files.
    # ============================================================
    # Obtains absolute directory path containing current python script file.
    BASE_DIR = (Path(__file__).resolve().parent)

    # Constructs relative path pointing to input video file "video_3.mp4".
    video_path = (BASE_DIR / "input_video" / "video_3.mp4")

    # Constructs relative path pointing to fine-tuned YOLO player detection model weights.
    player_model_path = (BASE_DIR / "model" / "basketball_player_detection_training.pt")

    # Constructs relative path pointing to fine-tuned YOLO ball detection model weights.
    ball_model_path = (BASE_DIR / "model" / "basketball_ball_training.pt")

    # Constructs relative path pointing to PyTorch vision model weights for basketball court keypoint detection.
    court_keypoint_model_path = (BASE_DIR / "model" / "court_keypoint_detector.pt")

    # Constructs relative path pointing to 2D tactical court background graphic template image.
    court_image_path = (BASE_DIR / "images" / "img.png")

    # Constructs relative path pointing to pickle stub file for caching player detection and tracking data.
    player_stub_path = (BASE_DIR / "stubs" / "player_tracks_stubs.pkl")

    # Constructs relative path pointing to pickle stub file for caching ball tracking data.
    ball_stub_path = (BASE_DIR / "stubs" / "ball_tracks_stubs.pkl")

    # Constructs relative path pointing to pickle stub file for caching team assignment classifications.
    player_assignment_stub_path = (BASE_DIR / "stubs" / "player_assignment_stubs.pkl")

    # Constructs relative path pointing to target destination path for final annotated output video "video_12.avi".
    output_video_path = (
        BASE_DIR
        / "output_video"
        / "video_12.avi"
    )

    # ============================================================
    # 2. CHECK PROJECT FILES : Section header for verifying the existence of required video, model weights, and asset files.
    # ============================================================
    # Prints visual decorative divider lines and header for the project files check section.
    print("=" * 70)
    print("CHECKING PROJECT FILES")
    print("=" * 70)

    # Logs resolved absolute file path of input video file to console.
    print(f"Video : "f"{video_path}")

    # Logs resolved absolute file path of player detection model weights to console.
    print(f"Player model : " f"{player_model_path}")

    # Logs resolved absolute file path of ball detection model weights to console.
    print(f"Ball model : " f"{ball_model_path}")

    # Logs resolved absolute file path of court keypoint detector weights to console.
    print(f"Court keypoint model : " f"{court_keypoint_model_path}")

    # Logs resolved absolute file path of 2D court background image template to console.
    print(f"Court image : "f"{court_image_path}")

    # ------------------------------------------------------------
    # Validate files : Comment block introducing validation assertions for required project resources.
    # ------------------------------------------------------------
    # Checks whether target input video file exists on filesystem.
    if not video_path.exists():
        raise FileNotFoundError(
            f"\nInput video not found:\n"
            f"{video_path}"
        )

    # Checks whether player detection PyTorch model file exists on disk.
    if not player_model_path.exists():
        raise FileNotFoundError(
            f"\nPlayer model not found:\n"
            f"{player_model_path}"
        )

    # Checks whether ball detection PyTorch model file exists on disk.
    if not ball_model_path.exists():
        raise FileNotFoundError(
            f"\nBall model not found:\n"
            f"{ball_model_path}"
        )

    # Checks whether court keypoint detection model file exists on disk.
    if not court_keypoint_model_path.exists():
        model_directory = (BASE_DIR / "model")
        print("\nAvailable .pt files:")

        # Prints status message preceding available PyTorch model list logging.
        pt_files = list(model_directory.glob("*.pt"))

        # Evaluates if array contains any discovered '.pt' model files.
        if pt_files:
            # Iterates over each found PyTorch model file path object.
            for file in pt_files:
                print(f"  - {file.name}")

        # Executes fallback block if no '.pt' files exist in model directory.
        else:
            print("  No .pt files found.")
        raise FileNotFoundError(
            f"\nCourt keypoint model not found:\n"
            f"{court_keypoint_model_path}"
        )

    # Checks whether 2D tactical court background graphic exists on disk.
    if not court_image_path.exists():
        raise FileNotFoundError(
            f"\nCourt image not found:\n"
            f"{court_image_path}"
        )

    # ============================================================
    # 3. READ VIDEO : Section header for loading and decoding input video frames into memory.
    # ============================================================
    # Prints visual header divider announcing start of video frame extraction stage.
    print("\n" + "=" * 70)
    print("READING VIDEO")
    print("=" * 70)

    # Decodes input video file at video_path into list of OpenCV BGR image frames.
    video_frames = read_video(str(video_path))

    # Checks if returned frame array is empty or evaluates to False.
    if not video_frames:
        raise ValueError("No frames were read from the video.")

    # Measures total frame count decoded from input video stream.
    total_frames = len(video_frames)
    print(f"Total video frames: " f"{total_frames}")

    # ============================================================
    # 4. INITIALIZE PLAYER TRACKER : Section header for initializing deep learning player tracking algorithm.
    # ============================================================
    # Prints visual header divider for player tracker initialization step.
    print("\n" + "=" * 70)
    print("INITIALIZING PLAYER TRACKER")
    print("=" * 70)

    # Instantiates PlayerTracker object by passing file path string of fine-tuned player detection weights.
    player_tracker = PlayerTracker(str(player_model_path))

    # Logs confirmation message indicating successful instantiation of player tracking engine.
    print("Player tracker initialized successfully.")

    # ============================================================
    # 5. INITIALIZE BALL TRACKER : Section header for initializing the deep learning ball tracking module.
    # ============================================================
    # Prints visual decorative divider lines and header for the ball tracker setup phase.
    print("\n" + "=" * 70)
    print("INITIALIZING BALL TRACKER")
    print("=" * 70)

    # Instantiates BallTracker by providing the string file path to the trained ball detection model weights.
    ball_tracker = BallTracker(str(ball_model_path))
    print("Ball tracker initialized successfully.")

    # ============================================================
    # 6. INITIALIZE COURT KEYPOINT DETECTOR : Section header for setting up the court keypoint extraction model.
    # ============================================================
    # Prints visual header divider announcing court keypoint detector initialization.
    print("\n" + "=" * 70)
    print("INITIALIZING COURT KEYPOINT DETECTOR")
    print("=" * 70)

    # Instantiates CourtKeypointDetector with the target PyTorch keypoint detection model path.
    court_keypoint_detector = (CourtKeypointDetector(str(court_keypoint_model_path)))
    print("Court keypoint detector initialized successfully.")

    # ============================================================
    # 7. PLAYER TRACKING : Section header for executing player tracking across all video frames.
    # ============================================================
    # Prints decorative header marking the start of the player tracking pipeline stage.
    print("\n" + "=" * 70)
    print("PLAYER TRACKING")
    print("=" * 70)

    # Computes player detection tracks across video frames, attempting to load cached results from pickle stub path first.
    player_tracks = (player_tracker.get_object_tracks(video_frames, read_from_stub=True,
                                                      stub_path=str(player_stub_path)))

    # Validates that total frame count of extracted player tracking data matches original video frame length.
    validate_frame_count(player_tracks, total_frames,"PlayerTracker",)
    print(f"Player tracking completed: " f"{len(player_tracks)} frames")

    # ============================================================
    # 8. BALL TRACKING : Section header for running ball tracking algorithms across video frames.
    # ============================================================
    # Prints visual banner for the ball tracking pipeline phase.
    print("\n" + "=" * 70)
    print("BALL TRACKING")
    print("=" * 70)

    # Computes ball position tracks frame-by-frame, leveraging cached stub file if read_from_stub is True.
    ball_tracks = (ball_tracker.get_object_tracks(video_frames,read_from_stub=True,
                                                  stub_path=str(ball_stub_path)))

    # Validates that output frame count from ball tracking matches target video frame length.
    validate_frame_count(ball_tracks, total_frames,"BallTracker",)

    # Logs completion of ball tracking along with processed frame count.
    print(f"Ball tracking completed: " f"{len(ball_tracks)} frames")

    # ============================================================
    # 9. COURT KEYPOINT DETECTION : Section header for extracting court reference keypoints across video frames.
    # ============================================================
    # Prints decorative text header for the court keypoint detection pipeline step.
    print("\n" + "=" * 70)
    print("COURT KEYPOINT DETECTION")
    print("=" * 70)

    # Calls detect_court_keypoints helper function to extract frame-by-frame anatomical court boundary points.
    court_keypoints = (detect_court_keypoints(court_keypoint_detector, video_frames))
    if court_keypoints is None:
        raise ValueError("Court keypoint detector returned None.")
    print("Court keypoint detection completed.")
    print(f"Court keypoint result type: " f"{type(court_keypoints)}")

    # ============================================================
    # 10. TEAM ASSIGNMENT : Section header for classifying players into opposing teams based on jersey features.
    # ============================================================
    # Prints decorative section banner for team assignment processing stage.
    print("\n" + "=" * 70)
    print("TEAM ASSIGNMENT")
    print("=" * 70)

    # Instantiates TeamAssigner specifying custom target uniform class descriptors for Team 1 and Team 2.
    team_assigner = TeamAssigner(team_1_class_name="white shirt",
                                 team_2_class_name="dark blue shirt")

    # Classifies detected players into teams across frames, attempting cache load from stub file if available.
    player_assignment = (team_assigner.get_player_teams_across_frames(video_frames,player_tracks,
                                                                      read_from_stub=True,
                                                                      stub_path=str(
                                                                          player_assignment_stub_path)))

    # Validates that output player team assignment list matches exact input video frame length.
    validate_frame_count(player_assignment, total_frames,"TeamAssigner",)
    print(f"Player team assignment completed: " f"{len(player_assignment)} frames")

    # ============================================================
    # 11. BALL POSSESSION : Section header for ball possession detection and acquisition tracking phase.
    # ============================================================
    # Prints visual decorative divider lines and header for the ball possession detection step.
    print("\n" + "=" * 70)
    print("BALL POSSESSION DETECTION")
    print("=" * 70)

    # Instantiates BallAquisitionDetector engine to determine frame-by-frame player proximity to the ball.
    ball_aquisition_detector = (BallAquisitionDetector())

    # Runs proximity and control detection logic using player and ball track histories across frames.
    ball_aquisition = (ball_aquisition_detector.detect_ball_possession(player_tracks,ball_tracks))

    # Confirms output array length matches target video frame count.
    validate_frame_count(ball_aquisition, total_frames,"BallAquisitionDetector")
    print("Ball possession detection completed.")

    # ============================================================
    # 12. PASS AND INTERCEPTION : Section header for evaluating team passes and defensive interceptions.
    # ============================================================
    print("\n" + "=" * 70)
    print("PASS AND INTERCEPTION DETECTION")
    print("=" * 70)

    # Instantiates PassAndInterceptionDetector processing engine.
    pass_and_interception_detector = (PassAndInterceptionDetector())

    # Analyzes player tracks, ball movement, team IDs, and possession data to identify successful pass events.
    passes = (pass_and_interception_detector.detect_passes(player_tracks,ball_tracks,
                                                           player_assignment,ball_aquisition))

    # Detects defensive turnover and interception events based on team possession transitions.
    interceptions = (pass_and_interception_detector.detect_interceptions(player_tracks,ball_tracks,
                                                                         player_assignment,
                                                                         ball_aquisition))

    # Fallback assignment: converts None output to empty list for safe list handling.
    if passes is None:
        passes = []

    # Fallback assignment: converts None output to empty list for safe list handling.
    if interceptions is None:
        interceptions = []
    print(f"Pass events detected: "f"{len(passes)}")
    print(f"Interception events detected: "f"{len(interceptions)}")

    # ============================================================
    # 13. CONVERT EVENTS : Section header for converting discrete event structures into frame-indexed lookup arrays.
    # ============================================================
    print("\n" + "=" * 70)
    print("CONVERTING PASS / INTERCEPTION EVENTS")
    print("=" * 70)

    # Converts list of pass event dictionaries into frame-wise team status array.
    pass_array = (convert_pass_events_to_frame_array(passes, total_frames))

    # Converts list of interception event dictionaries into frame-wise team status array.
    interception_array = (convert_interception_events_to_frame_array(interceptions,total_frames))

    # Validates that converted pass array frame length matches total video frames.
    validate_frame_count(pass_array, total_frames,"Pass event array")

    # Validates that converted interception array frame length matches total video frames.
    validate_frame_count(interception_array, total_frames,"Interception event array")
    print(f"Team 1 pass frames: "f"{pass_array.count(1)}")
    print(f"Team 2 pass frames: "f"{pass_array.count(2)}")
    print(f"Team 1 interception frames: "f"{interception_array.count(1)}")
    print(f"Team 2 interception frames: "f"{interception_array.count(2)}")

    # ============================================================
    # 14. INITIALIZE DRAWERS : Section header for instantiating video graphics rendering overlay classes.
    # ============================================================
    print("\n" + "=" * 70)
    print("INITIALIZING DRAWERS")
    print("=" * 70)

    # Instantiates drawer responsible for rendering bounding boxes/circles and player IDs on frames.
    player_tracks_drawer = (PlayerTracksDrawer())

    # Instantiates drawer responsible for rendering ball indicators and movement traces.
    ball_tracks_drawer = (BallTracksDrawer())

    # Instantiates drawer responsible for rendering overall possession percentage HUD overlays.
    team_ball_control_drawer = (TeamBallControlDrawer())

    # Instantiates drawer for rendering visual badges/banners on pass and interception events.
    pass_interception_drawer = (PassInterceptionDrawer())

    # Instantiates drawer for plotting keypoint landmarks directly onto the main court camera view.
    court_keypoint_drawer = (CourtKeypointDrawer())

    # Instantiates drawer responsible for rendering top-down 2D radar mini-map overlay.
    tactical_view_drawer = (TacticalViewDrawer())
    print("All drawers initialized successfully.")

    # ============================================================
    # 15. DRAW PLAYER TRACKS : Section header for drawing player annotations onto raw video frames.
    # ============================================================
    print("Drawing player tracks...")

    # Renders player graphics (team colors, player IDs, possession highlights) onto video frames.
    output_video_frames = (player_tracks_drawer.draw(video_frames,player_tracks,
                                                     player_assignment,ball_aquisition))

    # Restores frame array size if PlayerTracksDrawer added or dropped any frame elements.
    output_video_frames = (restore_frame_count(output_video_frames,video_frames,"PlayerTracksDrawer"))
    print(Player tracks drawn: {len(output_video_frames)} frames)

    # ============================================================
    # 16. DRAW BALL TRACKS : Section header for overlaying ball trajectory and position annotations onto video frames.
    # ============================================================
    print("Drawing ball tracks...")
    # Saves state reference of frames prior to drawing ball annotations for post-validation check.
    frames_before_ball = (output_video_frames)

    # Draws ball trajectories, bounding boxes, or indicators onto the output video frames.
    output_video_frames = (ball_tracks_drawer.draw(output_video_frames,ball_tracks))

    # Ensures frame count consistency between input and output of BallTracksDrawer.
    output_video_frames = (restore_frame_count(output_video_frames, frames_before_ball,"BallTracksDrawer"))
    print(Ball tracks drawn: {len(output_video_frames)} frames)

    # ============================================================
    # 17. DRAW TEAM BALL CONTROL : Section header for rendering team ball possession statistics and overlays.
    # ============================================================
    print("Drawing team ball control...")

    # Saves reference frame array state before team ball control drawer execution.
    frames_before_team = (output_video_frames)

    # Draws team possession status graphics (e.g., possession percentage bars/HUD) onto frames.
    output_video_frames = (team_ball_control_drawer.draw(output_video_frames,player_assignment,
                                                         ball_aquisition))

    # Restores original frame list size if TeamBallControlDrawer altered frame count.
    output_video_frames = (restore_frame_count(output_video_frames, frames_before_team,"TeamBallControlDrawer"))
    print(f"Team ball control drawn: f{len(output_video_frames)} frames")

    # ============================================================
    # 18. DRAW PASS / INTERCEPTION
    # ============================================================
    print("Drawing pass and interception statistics...")

    # Saves frame array snapshot prior to executing pass/interception overlay drawer.
    frames_before_pass = (output_video_frames)

    # Renders event notification badges and stats overlays for passes and interceptions on video frames.
    output_video_frames = (pass_interception_drawer.draw(output_video_frames, pass_array,
                                                         interception_array))

    # Validates and restores frame count consistency for PassInterceptionDrawer.
    output_video_frames = (restore_frame_count(output_video_frames, frames_before_pass,"PassInterceptionDrawer"))
    print(Pass/interception statistics drawn: {len(output_video_frames)} frames")

    # ============================================================
    # 19. DRAW COURT KEYPOINTS : Section header for rendering detected court keypoint lines/dots on video frames.
    # ============================================================
    print("Drawing court keypoints...")

    # Stores pre-drawing frame array snapshot for frame integrity checks.
    frames_before_court = (output_video_frames)

    # Draws mapped court reference keypoints and boundaries onto output video frames.
    output_video_frames = (court_keypoint_drawer.draw(output_video_frames,court_keypoints))

    # Verifies and restores frame count following court keypoints overlay step.
    output_video_frames = (restore_frame_count(output_video_frames,frames_before_court,
                                               "CourtKeypointDrawer"))
    print(f"Court keypoints drawn: " f"{len(output_video_frames)} frames")

    # ============================================================
    # 20. INITIALIZE TACTICAL VIEW : Section header for initializing perspective homography transformation for 2D tactical view.
    # ============================================================
    print("\n" + "=" * 70)
    print("INITIALIZING TACTICAL VIEW")
    print("=" * 70)

    # Instantiates TacticalViewConverter object using template court layout image path.
    tactical_view_converter = (TacticalViewConverter(court_image_path=str(court_image_path)))
    print("Tactical view converter initialized successfully.")
    print("Tactical view converter attributes:")
    # Defines list of configuration attribute names to inspect on tactical view converter object.
    attributes_to_print = ("actual_height_in_meters","actual_width_in_meters",
                           "court_image_path","height","width","key_points")

    # Iterates through target attribute string names.
    for attribute in attributes_to_print:
        # Checks whether specified attribute exists on tactical_view_converter instance.
        if hasattr(tactical_view_converter, attribute):
            print(f"  {attribute}: " f"{getattr(tactical_view_converter, attribute)}")

    # ============================================================
    # 21. SPEED AND DISTANCE : Section header for calculating physical player movement metrics (speed and distance).
    # ============================================================
    (player_distance_per_frame, player_speed_per_frame) = calculate_player_speed_and_distance(tactical_view_converter, court_keypoints, player_tracks)

    # ============================================================
    # 22. DRAW SPEED AND DISTANCE : Section header for drawing calculated speed and distance labels under players.
    # ============================================================
    print("Drawing speed and distance...")

    # Checks whether speed and distance metric arrays were successfully computed.
    if (player_distance_per_frame is not None and player_speed_per_frame is not None):
        # Instantiates SpeedAndDistanceDrawer object to handle text overlay rendering.
        speed_and_distance_drawer = (SpeedAndDistanceDrawer())
        # Saves reference snapshot of video frames array prior to rendering step.
        frames_before_speed_distance = (output_video_frames)
        # Begins safety try block to handle potential drawer rendering exceptions gracefully.
        try:
            # Draws speed (km/h or m/s) and cumulative distance values near corresponding players on each frame.
            output_video_frames = (speed_and_distance_drawer.draw(output_video_frames,
                                                                  player_tracks,
                                                                  player_distance_per_frame,
                                                                  player_speed_per_frame))

            # Validates and restores frame count consistency after speed/distance drawing.
            output_video_frames = (restore_frame_count(output_video_frames,
                                                       frames_before_speed_distance,
                                                       "SpeedAndDistanceDrawer"))

            print(f"Speed/distance drawn: " f"{len(output_video_frames)} frames")

        # Catches any error during drawing process without crashing the pipeline execution.
        except Exception as error:
            print("SpeedAndDistanceDrawer failed.")
            print("Reason: {error}")
            print("Continuing without speed/distance overlay.")
            output_video_frames = (frames_before_speed_distance)
    else:
        print("Speed/distance data unavailable.")
        print("SpeedAndDistanceDrawer skipped.")

    # ============================================================
    # 23. PREPARE TACTICAL COURT KEYPOINTS : Section header for extracting keypoint coordinate configuration for tactical minimap display.
    # ============================================================
    print("\n" + "=" * 70)
    print("PREPARING TACTICAL COURT KEYPOINTS")
    print("=" * 70)

    # Safely fetches key_points property from converter object, defaulting to None if missing.
    tactical_court_keypoints = getattr(tactical_view_converter, "key_points", None)

    # Checks whether tactical keypoints property retrieval failed.
    if tactical_court_keypoints is None:
        raise AttributeError(
            "\nTacticalViewConverter does not expose "
            "`key_points`."
        )
    print(f"Tactical court keypoints: " f"{len(tactical_court_keypoints)}")

    # ============================================================
    # 24. TACTICAL COURT SIZE : Section header for retrieving width and height parameters of 2D tactical court image.
    # ============================================================
    # retrieves width attribute from tactical view converter instance.
    tactical_width = getattr(tactical_view_converter, "width", None)
    # retrieves height attribute from tactical view converter instance.
    tactical_height = getattr(tactical_view_converter, "height", None)
    # Validates whether width attribute is missing.
    if tactical_width is None:
        raise AttributeError(
            "TacticalViewConverter does not expose "
            "`width`."
        )
    # Validates whether height attribute is missing.
    if tactical_height is None:
        raise AttributeError(
            "TacticalViewConverter does not expose "
            "`height`."
        )
    print(f"Tactical court width : "f"{tactical_width}")
    print(f"Tactical court height: "f"{tactical_height}")

    # ============================================================
    # 25. DRAW TACTICAL VIEW : Section header for rendering top-down 2D tactical radar minimap overlay.
    # ============================================================
    print("Drawing tactical view...")
    frames_before_tactical = (output_video_frames)
    try:

        # Renders top-down 2D tactical court radar graphic with player position dots onto main frames.
        output_video_frames = (tactical_view_drawer.draw(output_video_frames,
                                                         tactical_view_converter.court_image_path,
                                                         tactical_width,tactical_height,
                                                         tactical_court_keypoints))

        # Restores and verifies frame list size after tactical minimap drawing.
        output_video_frames = (restore_frame_count(output_video_frames,frames_before_tactical,
                                                   "TacticalViewDrawer"))
        print(Tactical view drawn: {len(output_video_frames)} frames)
    except TypeError as error:
        print("TacticalViewDrawer.draw() argument compatibility error:")
        print(error)
        print("Available TacticalViewDrawer methods:")
        #  Iterates over all attributes and methods of tactical_view_drawer.
        for name in dir(tactical_view_drawer):
            if not name.startswith("_"):
                print(f"  - {name}")
        raise

    # ============================================================
    # 26. FINAL VALIDATION : Section header for performing final sanity checks on fully annotated output video frames.
    # ============================================================
    print("\n" + "=" * 70)
    print("FINAL OUTPUT VALIDATION")
    print("=" * 70)

    # Checks if generated output video frame list is empty or invalid.
    if not output_video_frames:
        raise ValueError("Output video frames are empty.")

    # Confirms total annotated output frame count matches original input video frame count.
    validate_frame_count(output_video_frames, total_frames,"Final Pipeline")
    print(Input frames : f{total_frames})
    print(f"Output frames: "f"{len(output_video_frames)}")
    print("Frame count validation passed.")

    # ============================================================
    # 27. SAVE VIDEO : Section header for writing processed frames to output video file on disk.
    # ============================================================
    print("\n" + "=" * 70)
    print("SAVING OUTPUT VIDEO")
    print("=" * 70)

    # Creates parent directory folder structure for output video path if it does not already exist.
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    # Encodes processed frames and saves video file to output_video_path using OpenCV VideoWriter.
    save_video(output_video_frames, str(output_video_path))

    print("Output video saved successfully:")
    print(output_video_path)

    # ============================================================
    # 28. FINAL SUMMARY : Section header for logging overall pipeline execution metric summary.
    # ============================================================
    print("\n" + "=" * 70)
    print("BASKETBALL INTELLIGENCE PIPELINE COMPLETED")
    print("=" * 70)
    print(f"Frames processed    : " f"{total_frames}")
    print(f"Pass events         : "f"{len(passes)}")
    print(f"Interception events : "f"{len(interceptions)}")
    print(f"Output video        : "f"{output_video_path}")
    print("=" * 70)

# ================================================================
# ENTRY POINT : Main script function when run directly.
# ================================================================
# Evaluates if script is being executed as top-level program.
if __name__ == "__main__":
    # Calls main() entry point function to launch basketball analytics pipeline
    main()