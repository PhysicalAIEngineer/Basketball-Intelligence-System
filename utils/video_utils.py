"""
module for reading and writing video files provides utility function to load video
frames into memory and save processed frames back to video files with support for
common video formats
"""

# Imports the standard Operating System library for handling file paths and directories.
import os

# Imports OpenCV for reading, writing, and manipulating video frames.
import cv2


def read_video(video_path):
    """
    read all frames from a video file into memory
    :param video_path: path to video file
    :return: list of video frames as numpy array
    """
    # open video
    cap = cv2.VideoCapture(video_path)

    # Creates a VideoCapture object to establish an open stream to the target video file.
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Initializes an empty list to store loaded video frames
    frames = []

    # Reads the next sequential frame.
    while True:
        # Reads the next sequential frame. 'ret' is a boolean flag (True if successful), and 'frame' is a NumPy array.
        ret, frame = cap.read()
        # Breaks the loop when the video reaches the end or fails to decode a frame.
        if not ret:
            break
        # Appends the decoded frame
        frames.append(frame)

    # Closes the video file handle to free system
    cap.release()

    # Ensures the file wasn't corrupt or empty before returning the output.
    if len(frames) == 0:
        raise ValueError(f"Video Was opened but no frames could be read:" f" {os.path.basename(video_path)}")

    # Returns the complete list of loaded frame
    return frames

def save_video(output_video_frames, output_video_path):
    """
    save processed frames to a video file
    creates necessary directory if they do not exist and writes frames using XVID codec
    :param output_video_frames: list of frames to save
    :param output_video_path:  path where the video should be saved
    """
    # Guards against passing an empty list, which would crash resolution checks below.
    if not output_video_frames:
        raise ValueError("output_video_frames is empty")

    # Extracts the target directory path from the full destination path.
    output_dir = os.path.dirname(output_video_path)

    # Automatically creates missing parent directories so file creation doesn't fail.
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Extracts pixel dimensions (Height, Width) from the shape tuple `(H, W, Channels)` of the first frame.
    height, width = output_video_frames[0].shape[:2]

    # Defines the 'XVID' MPEG-4 video codec specification using standard four-character code syntax.
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # Initializes the VideoWriter object with destination path, codec, frame rate (24 FPS), and target size tuple (Width, Height).
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))

    # Validates that OpenCV successfully initialized the video writer.
    if not out.isOpened():
        raise ValueError(f"Could not open output video file: {output_video_path}")
    # Iterates over each frame in the memory list and encodes it directly into the video file.
    for frame in output_video_frames:
        out.write(frame)

    # Finalizes and closes the video file writer, saving the file to disk.
    out.release()

    # Prints the absolute file path to confirm successful export.
    print(f"Video saved successfully: " f"{os.path.abspath(output_video_path)}")

