# Imports video reader ('read_video') and writer ('save_video') helper functions from local utils package.
from utils.video_utils import read_video, save_video

# Defines the main execution function of the script.
def main():

    # Defines relative path string to source video file.
    input_video_path = "input_video/video_1.mp4"

    # Defines target file path string where output processed video will be saved.
    output_video_path = "output_video/output_video_1.avi"

    # Prints status message to console indicating video loading has started.
    print("Reading video...")

    # Reads video file and extracts frames as a list of NumPy arrays.
    video_frames = read_video(input_video_path)

    # Prints formatted string displaying total number of loaded frames.
    print(f"Frames read: {len(video_frames)}")

    # Prints status message indicating video encoding and saving process has started.
    print("Saving video...")

    # Encodes frame list back into video file and writes it to target output directory path.
    save_video(video_frames, output_video_path)

# Script entry point guard: executes main() function only when run directly as main module.
if __name__ == "__main__":
    main()