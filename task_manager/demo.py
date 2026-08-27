# import the YOLO model from ultralytics library
from ultralytics import YOLO

# load the pre-trained YOLO model
model = YOLO("model/basketball_player_detection_training.pt")

# Perform Detection
# Using stream=True is a massive optimization for videos as it prevents
# your system memory (RAM) from filling up on long video clips.
results = model.track(source="input_video/video_1.mp4", save=True, stream=True)

print("===========")
print("Printing Bounding Box Information:")

# Print Bounding Box Information Frame by Frame
for frame_idx, result in enumerate(results):
    print(f"\n--- Frame {frame_idx + 1} ---")

    # Check if any objects were detected in this frame
    if len(result.boxes) == 0:
        print("No players detected in this frame.")
        continue

    # Loop through every single box found in the current frame
    for box in result.boxes:
        # Get coordinates in xyxy format (xmin, ymin, xmax, ymax)
        coords = box.xyxy[0].tolist()
        # Get confidence score (e.g., 0.85)
        conf = float(box.conf[0])
        # Get class ID integer (e.g., 0 for player)
        class_id = int(box.cls[0])
        # Get class name label string
        class_name = model.names[class_id]

        print(f"Label: {class_name} | Confidence: {conf:.2f} | Bounding Box: {coords}")
