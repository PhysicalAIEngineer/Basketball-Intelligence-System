# Imports the Supervision library for high-level computer vision annotations and visualizations.
import supervision as sv


class CourtKeypointDrawer:
    """
    A drawer class responsible for drawing court keypoints on a sequence of frames.

    Attributes:
        keypoint_color (str): Hex color value for the keypoints.
    """

    # Initializes default hex color string for court keypoint markers (bright red).
    def __init__(self):
        self.keypoint_color = '#ff2c2c'

    # Function docstring detailing input frame lists, keypoint coordinate structures, and output frames.
    def draw(self, frames, court_keypoints):
        """
        Draws court keypoints on a given list of frames.

        Args:
            frames (list): A list of frames (as NumPy arrays or image objects) on which to draw.
            court_keypoints (list): A corresponding list of lists where each sub-list contains
                the (x, y) coordinates of court keypoints for that frame.

        Returns:
            list: A list of frames with keypoints drawn on them.
        """

        # Configures Supervision VertexAnnotator to draw keypoint dots with a 8px radius in red color.
        vertex_annotator = sv.VertexAnnotator(color=sv.Color.from_hex(self.keypoint_color), radius=8)

        # Configures Supervision VertexLabelAnnotator to render white keypoint index label badges with red backgrounds.
        vertex_label_annotator = sv.VertexLabelAnnotator(color=sv.Color.from_hex(self.keypoint_color),
            text_color=sv.Color.WHITE, text_scale=0.5, text_thickness=1)

        # List to store video frames after keypoints and labels have been rendered.
        output_frames = []

        # Iterates through frames while keeping track of the current frame index (`index`).
        for index, frame in enumerate(frames):

            # Creates a duplicate copy of the frame array to prevent modifying original video data.
            annotated_frame = frame.copy()

            # Extracts the court keypoints tensor/array for the current frame index.
            keypoints = court_keypoints[index]

            # Annotates frame with keypoint circular vertex markers using Supervision.
            annotated_frame = vertex_annotator.annotate(scene=annotated_frame, key_points=keypoints)

            # Moves keypoints tensor from GPU to CPU (if necessary) and converts it to a NumPy array.
            keypoints_numpy = keypoints.cpu().numpy()

            # Renders numerical index labels next to each keypoint on the image.
            annotated_frame = vertex_label_annotator.annotate(scene=annotated_frame,key_points=keypoints_numpy)

            # Appends fully annotated frame containing keypoint dots and index badges to output list.
            output_frames.append(annotated_frame)

        # Returns list of video frames populated with court keypoint vertex annotations.
        return output_frames