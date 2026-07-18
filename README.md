# Theater Seat Occupancy Detection

A computer vision pipeline that detects individual seats in a theater image and classifies each as **Occupied** or **Unoccupied**, producing an aggregate occupancy count.

---

## Overview

| | |
|---|---|
| **Input** | Theater seating image (100 seats) |
| **Output** | Total / Occupied / Unoccupied seat counts, annotated image, structured JSON results |
| **Tech Stack** | Python, OpenCV, NumPy, Streamlit |
| **Runtime** | < 1 second per image (CPU) |

---

## Problem Statement

Given an image of a theater containing 100 seats, detect each seat and classify its occupancy status, then report:

```
Total Seats      : 100
Occupied Seats   : 68
Unoccupied Seats : 32
```

---

## Architecture

```
                 ┌─────────────────────┐
   Input Image ─▶│   Seat Detection    │  Grayscale threshold + contour
                 │                     │  detection, filtered by seat-
                 └──────────┬──────────┘  sized bounding box area
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Occupancy           │  Per-seat crop analysis:
                 │ Classification      │  hue variance + skin/clothing
                 │                     │  tone ratio
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Aggregation &      │  Counts, annotated image,
                 │  Reporting          │  JSON export
                 └─────────────────────┘
```

### 1. Seat Detection
Seats are localized using classical image processing:
1. Convert image to grayscale.
2. Apply binary thresholding to separate seat-colored regions from the (darker) background.
3. Run contour detection on the thresholded mask.
4. Filter contours by bounding-box area to retain seat-sized regions and discard noise.

### 2. Occupancy Classification
Each detected seat region is cropped and evaluated on two signals:

| Signal | Rationale |
|---|---|
| Hue-channel standard deviation | An empty seat is a near-uniform fabric color; an occupied seat has higher color variance due to the presence of a person. |
| Skin/clothing-tone pixel ratio | Presence of skin or clothing tones within the seat crop indicates occupancy. |

A seat is classified **Occupied** if either signal exceeds its threshold.

### 3. Output
- Console summary (total / occupied / unoccupied)
- Annotated image — green box = unoccupied, red box = occupied
- `seat_results.json` — per-seat bounding box and classification, for downstream use

---

## Project Structure

```
theater-seat-detection/
├── app.py                     # Streamlit UI
├── seat_occupancy.py          # Core detection + classification pipeline
├── generate_test_image.py     # Synthetic test image generator (see Note on Data)
├── theater_input.png          # Sample input image
├── annotated_output.png       # Sample output image
├── ground_truth.json          # Ground-truth labels for the sample image
├── seat_results.json          # Pipeline output for the sample image
└── README.md
```

---

## Setup

**Requirements:** Python 3.9+

```bash
pip install opencv-python-headless numpy pillow streamlit
```

## Usage

### Run via CLI
```bash
python3 generate_test_image.py            # generates a sample input image
python3 seat_occupancy.py theater_input.png
```

Output:
```
Total Seats      : 100
Occupied Seats   : 65
Unoccupied Seats : 35
```

### Run via UI
```bash
streamlit run app.py
```
Upload a theater seating image and view detection results, side-by-side comparison, and live counts in the browser.

---

## Note on Test Data

No real theater photo was provided as part of the assessment. `generate_test_image.py` procedurally generates a representative 10x10 seat grid image (seat-colored regions on a dark background, with occupied seats showing a simplified person silhouette) so the pipeline can be developed and validated end-to-end. The same pipeline accepts any real theater image as input via the CLI or UI upload.

---

## Known Limitations

| Limitation | Impact |
|---|---|
| Threshold + contour detection assumes seats are visually distinct from the background | May not generalize to cluttered or low-contrast real-world photos |
| No perspective correction | Assumes a roughly frontal or top-down camera angle |
| Classification is heuristic-based, not learned | Less robust to lighting variation, occlusion, and unusual seat/clothing colors than a trained classifier |

## Recommended Next Steps (Production Hardening)

| Improvement | Approach | Est. Effort |
|---|---|---|
| Robust seat detection | Fine-tune a YOLOv8 object detector on labeled real seat images | 1-2 days (data labeling is the bottleneck) |
| Learned occupancy classifier | Fine-tune a lightweight CNN (e.g. MobileNetV2) on labeled seat crops | 4-8 hours, given labeled data |
| Perspective correction | Homography transform using reference corner points | 2-4 hours |
| Robustness to occlusion/lighting | Iterative improvement with more diverse training data | Ongoing |

---

## Author
Sowmiya
