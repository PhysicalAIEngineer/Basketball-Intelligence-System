# Imports OpenCV library for frame manipulations and drawing text.
import cv2

# Defines a class responsible for rendering frame index numbers on video frames.
class FrameNumberDrawer:

    # Default constructor initializing the FrameNumberDrawer class instance.
    def __init__(self):
        pass

    # Draws sequential frame numbers on the top-left corner of each provided video frame.
    def draw(self,frames):

        #  List to store processed video frames with drawn frame numbers.
        output_frames = []

        # Iterates through the total count of frames using an integer index `i`.
        for i in range(len(frames)):

            # Creates a duplicate copy of the current frame array to avoid modifying the original image.
            frame = frames[i].copy()

            # Renders frame index integer `i` as green text at coordinates (x=10, y=30) with scale 1 and thickness 2.
            cv2.putText(frame, str(i), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Appends annotated frame featuring top-left frame number to output list.
            output_frames.append(frame)

        # Returns list of video frames containing frame index overlay numbers.
        return output_frames