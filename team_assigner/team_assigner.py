# Imports PIL Image class to format cropped player images for Hugging Face processors.
from PIL import Image

# Imports OpenCV for reading video frame array data
import cv2

# Imports PyTorch to manage hardware device allocation (CPU vs CUDA) and tensor evaluation.
import torch

# Imports Hugging Face Transformers components to load zero-shot image classification processor pipelines and models.
from transformers import (AutoProcessor, AutoModelForZeroShotImageClassification)

# Imports caching functions to save and load computed team assignments to/from disk.
from utils import read_stub, save_stub


class TeamAssigner:
    """
    Assign players to teams based on jersey appearance.
    Uses a zero-shot vision-language model to classify the player's jersey crop into one of two team descriptions.
    """

    # Class that classifies player jersey appearances into team categories using vision-language zero-shot models (like Fashion-CLIP).
    def __init__(self, team_1_class_name="white shirt", team_2_class_name="dark blue shirt",
                 model_name="patrickjohncyh/fashion-clip"):
        """
        Args:
            team_1_class_name: Description of Team 1 jersey.
            team_2_class_name: Description of Team 2 jersey.
            model_name: Hugging Face model name.
        """

        # Initializes an empty dictionary reserved for storing team color mappings.
        self.team_colors = {}

        # Initializes an in-memory cache dictionary mapping unique player track IDs to assigned team IDs.
        self.player_team_dict = {}

        # Sets the text prompt description for Team 1's jersey color/type.
        self.team_1_class_name = team_1_class_name
        # Sets the text prompt description for Team 2's jersey color/type.
        self.team_2_class_name = team_2_class_name

        # Specifies the Hugging Face pre-trained model repository name to download/load.
        self.model_name = model_name

        # Placeholder for lazy-loading the transformer model instance.
        self.model = None
        # Placeholder for lazy-loading the image processor instance.
        self.processor = None

        # Sets execution target to CUDA GPU if hardware is detected, otherwise defaults to CPU.
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        """
        Load the zero-shot image classification model.
        """

        # Skips reload if the model is already initialized in memory.
        if self.model is not None:
            return

        # Logs model loading initialization
        print(f"Loading team classification model: " f"{self.model_name}")

        # Downloads and loads the text/image feature processor from Hugging Face.
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        # Downloads and loads the text/image feature processor from Hugging Face.
        self.model = AutoModelForZeroShotImageClassification.from_pretrained(self.model_name)

        # Transfers model parameters to the designated compute device (GPU or CPU).
        self.model.to(self.device)

        # Puts model in evaluation mode to disable dropout layers and stabilize inference.
        self.model.eval()

        # Confirms model setup completion to console.
        print(f"Team classification model loaded on " f"{self.device}")

    def get_player_color(self, frame, bbox):
        """
        Classify the player's jersey appearance.

        Args:
            frame: OpenCV BGR frame.
            bbox: Bounding box [x1, y1, x2, y2].

        Returns:
            Predicted team jersey description.
        """

        # Ensures model and processor are loaded
        self.load_model()

        # Extracts pixel height and width from frame dimensions.
        height, width = frame.shape[:2]

        # Unpacks bounding box coordinate values.
        x1, y1, x2, y2 = bbox

        # Convert coordinates to integers bounding box coordinates to integer pixel indices.
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        # Clip bounding box to image boundaries bounding box values inside valid image frame array limits
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        # Validate bounding box returns default Team 2 class if bounding box width/height is zero or negative.
        if x2 <= x1 or y2 <= y1:
            return self.team_2_class_name

        # Crop player Slices frame matrix to extract crop patch containing only the targeted player.
        player_crop = frame[y1:y2, x1:x2]

        # returns default Team 2 class if the cropped image patch contains zero pixel elements.
        if player_crop.size == 0:
            return self.team_2_class_name

        # OpenCV BGR -> RGB Converts the RGB NumPy matrix into a PIL Image object required by the processor.
        rgb_image = cv2.cvtColor(player_crop, cv2.COLOR_BGR2RGB)

        # NumPy -> PIL
        image = Image.fromarray(rgb_image)

        # Defines target text classification labels for zero-shot text matching.
        classes = [self.team_1_class_name, self.team_2_class_name]

        # Prepare model inputs tokenizes text classes and normalizes image crop
        inputs = self.processor(text=classes, images=image, return_tensors="pt", padding=True)

        # Moves all created input tensors to the active compute hardware device (GPU or CPU).
        inputs = {key: value.to(self.device) for key, value in inputs.items() if hasattr(value, "to")}

        # Disables gradient computation for inference and passes preprocessed inputs through the model.
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extracts raw image-to-text class similarity score logits from model output.
        logits_per_image = outputs.logits_per_image

        # Applies Softmax activation across text classes to obtain probability distribution scores summing to 1.
        probabilities = logits_per_image.softmax(dim=1)

        # Get best class retrieves index corresponding to the highest predicted text class probability.
        class_index = probabilities.argmax(dim=1).item()

        # Returns the text class label string with the highest prediction score.
        return classes[class_index]


    def get_player_team(self, frame, player_bbox, player_id):
        """
        Assign a player to a team.
        Uses cached assignment if the player has already been classified.
        """
        # Return cached team assignment player ID in cache; if already assigned, returns cached team ID immediately to save compute.
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Classify jersey Calls zero-shot model to infer jersey appearance for an uncached player ID.
        player_color = self.get_player_color(frame, player_bbox)

        # Team assignment Maps predicted text class string to numerical team ID (1 or 2).
        if player_color == self.team_1_class_name:
            team_id = 1
        else:
            team_id = 2

        # Cache assignment Stores player ID team mapping in memory cache so future frames don't re-run neural inference.
        self.player_team_dict[player_id] = team_id

        # Returns assigned integer team ID.
        return team_id


    def get_player_teams_across_frames(self, video_frames, player_tracks, read_from_stub=False, stub_path=None):
        """
        Assign every tracked player to a team across all video frames.
        """

        # Attempts to load previously computed team assignments from disk cache.
        player_assignment = read_stub(read_from_stub, stub_path)

        # Verifies cached structure matches video frame count before returning disk cache.
        if player_assignment is not None:
            if len(player_assignment) == len(video_frames):
                print("Loaded player team assignments from stub.")
                return player_assignment

        # Load model
        self.load_model()

        # Initializes empty list to accumulate per-frame player-to-team assignments.
        player_assignment = []

        # Loops over every frame's tracked player bounding boxes.
        for frame_num, player_track in enumerate(player_tracks):
            # Appends empty dictionary for storing player team assignments in current frame.
            player_assignment.append({})
            # Retrieves corresponding video frame image matrix.
            frame = video_frames[frame_num]
            # Iterates through all tracked players present in current frame.
            for player_id, track in player_track.items():
                # Extracts bounding box coordinates for the current player track ID.
                bbox = track["bbox"]
                # Obtains team ID for player
                team = self.get_player_team(frame, bbox, player_id)
                # Maps player track ID to assigned team ID in frame dictionary.
                player_assignment[frame_num][player_id] = team

        # Save results computed team assignment dataset to disk for future execution runs.
        save_stub(stub_path, player_assignment)

        # Prints confirmation message indicating stub file export success.
        print("Player team assignments saved.")

        # Returns full list of per-frame player team mapping dictionaries.
        return player_assignment