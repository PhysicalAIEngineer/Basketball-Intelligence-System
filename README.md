# Basketball Intelligence System

End-to-End Computer Vision System for Basketball Video Analytics

An end-to-end Computer Vision and AI-based basketball intelligence system that converts broadcast basketball video into structured analytics on players, balls, teams, movements, possessions, events, and tactics.

The system processes basketball footage frame-by-frame and combines:
  1. Object Detection
  2. Multi-Object Tracking
  3. Ball Tracking
  4. Court Keypoint Detection
  5. Team Classification
  6. Ball Possession Analysis
  7. Pass Detection
  8. Interception Detection
  9. Perspective Transformation / Homography
  10. Tactical Court Visualization
  11. Player Speed Estimation
  12. Player Distance Estimation
  13. Automated Video Annotation

The final result is an annotated basketball video containing player IDs, team assignments, ball tracking, possession information, pass/interception events, court information, player movement metrics, and a tactical-view visualization.

## 🎯 Overview

Basketball contains a large amount of spatial and temporal information that is difficult to extract manually.

A single broadcast video contains information about:
  1. Where every player is located
  2. Which team each player belongs to
  3. Where the basketball is
  4. Which player controls the ball
  5. How far players move
  6. How fast players move
  7. When passes occur
  8. When interceptions occur
  9. How teams occupy the court
  10. How player positions evolve over time

## 🎯 Problem Statement

Traditional basketball analysis often requires:
  1. Manual video review
  2. Human tagging
  3. Expensive tracking hardware
  4. Specialized sports-analysis software
  5. Significant analyst time

This project explores how much of this process can be automated using standard video footage and computer vision.

The central problem is:

How can basketball video be transformed into structured player, team, movement, possession, and event analytics without requiring dedicated tracking hardware?

## 🎯 Objectives

The primary objectives are:
  1. Detect basketball players.
  2. Track players across frames.
  3. Detect and track the basketball.
  4. Detect court landmarks.
  5. Assign players to teams.
  6. Determine ball possession.
  7. Detect passes.
  8. Detect interceptions.
  9. Transform broadcast coordinates into tactical court coordinates.
  10. Estimate player speed.
  11. Estimate player distance traveled.
  12. Visualize player trajectories.
  13. Generate a tactical court view.
  14. Produce an annotated output video.
  15. Maintain frame-level consistency throughout the pipeline.

## 🏗️ End-to-End System Architecture
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/fe7cf3ed-4d9b-4d7f-a906-de85e0333fd2" />

## 📊 Visual Output


https://github.com/user-attachments/assets/e4a0c2cc-0712-4ae2-8c5c-810d00696c6c

## 📈 Evaluation Metrics
  <img width="638" height="660" alt="image" src="https://github.com/user-attachments/assets/87446890-8b6a-4610-a9cc-76b3cbc7085a" />

## 🏰 Project Structure
main.py
– Orchestrates the entire pipeline: reading video frames, running detection/tracking, team assignment, drawing results, and saving the output video.

trackers/
– Houses PlayerTracker and BallTracker, which use detection models to generate bounding boxes and track objects across frames.

utils/
– Contains helper functions like bbox_utils.py for geometric calculations, stubs_utils.py for reading and saving intermediate results, and video_utils.py for reading/saving videos.

drawers/
– Contains classes that overlay bounding boxes, court lines, passes, etc., onto frames.

ball_aquisition/
– Logic for identifying which player is in possession of the ball.

pass_and_interception_detector/
– Identifies passing events and interceptions.

court_keypoint_detector/
– Detects lines and keypoints on the court using the specified model.

team_assigner/
– Uses zero-shot classification (Hugging Face or similar) to assign players to teams based on jersey color.

configs/
– Holds default paths for models, stubs, and output video.





