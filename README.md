# RoadScan AI

AI-powered road defect detection and inspection system.

## Current milestone

The repository now contains the first working frontend foundation:

- Professional live-inspection dashboard
- Browser camera activation using `getUserMedia`
- GPS permission/status using browser geolocation
- Start/pause/stop inspection controls
- Live detection summary UI (initially zero; no fabricated detections)
- Evidence-integrity design for future camera-frame-backed detections
- Responsive dark field-operations interface

## Roadmap

1. Live camera + real pothole detector
2. Bounding boxes and confidence scores
3. Unique pothole tracking/counting
4. Evidence-frame capture
5. GPS geotagging
6. Interactive road map
7. Inspection database
8. Analytics
9. PDF inspection reports with maps and evidence

## Development

```bash
npm install
npm run dev
```

Open the local development server in a browser and grant camera/location permissions when prompted.

## Data integrity principle

RoadScan must never invent inspection results. A pothole count, map marker, or report entry should trace back to a real processed camera/video frame and its associated inspection record. When location data is unavailable, the application will explicitly show that it is unavailable.
