# Basketball Vision Analytics Frontend

React + Vite frontend matching the uploaded basketball-analysis reference video.

## Features

- Upload MP4/MOV/WebM basketball video
- Drag-and-drop upload
- Local video preview
- Analyze button with processing pipeline
- Reference annotated output video
- Detection / tracking / team / court / movement / possession / event dashboard
- Player tracking table
- Tactical court visualization
- Event timeline
- JSON export
- Optional FastAPI integration

## Run

```bash
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

## Real backend integration

Set:

```bash
VITE_API_BASE_URL=http://localhost:8000
```

The frontend sends:

```http
POST /api/analyze
Content-Type: multipart/form-data
file=<uploaded video>
```

Expected JSON:

```json
{
  "output_video_url": "/results/annotated.mp4",
  "metrics": {
    "players": 10,
    "trackingIdf1": 88.7,
    "trackingHota": 76.4,
    "teamAccuracy": 95.2,
    "courtMae": 0.72,
    "speedMae": 1.34,
    "possessionF1": 91.1,
    "passPrecision": 87.5,
    "passRecall": 83.3,
    "interceptionPrecision": 85.7,
    "interceptionRecall": 75.0
  }
}
```

## Production architecture

```text
React Frontend
      |
      | POST /api/analyze
      v
FastAPI
      |
      v
Video Worker
      |
      +--> YOLO Player/Ball Detection
      +--> ByteTrack / BoT-SORT
      +--> Team Classifier
      +--> Court Keypoints + Homography
      +--> Ball Possession
      +--> Speed / Distance
      +--> Pass / Interception
      |
      v
Annotated MP4 + JSON metrics
      |
      v
React Dashboard
```

The bundled `public/reference-output.mp4` is the frontend demonstration/reference output and is not newly inferred ground-truth analysis.