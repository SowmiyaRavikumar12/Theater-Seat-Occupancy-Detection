# Theater Seat Occupancy Detection

## Approach
1. **Seat detection**: Since no real theater photo was provided, a synthetic 10x10
   (100-seat) test image is generated (`generate_test_image.py`) that mimics a
   top-down/frontal theater seating view — seat-colored regions on a dark background,
   with occupied seats showing a simple person-shaped overlay.
   Seats are detected using classical CV: grayscale threshold to separate
   seat-colored regions from background, then contour detection + bounding-box
   area filtering to isolate seat-sized blobs. This is a deliberate choice over
   a trained detector (e.g. YOLO) given the time constraint and lack of labeled
   real-world training data — it's explainable and works well for a fixed-layout
   seating grid, which is realistic for a theater.

2. **Occupancy classification**: Each detected seat region is classified using
   two signals: (a) hue-channel standard deviation (empty seats are near-uniform
   fabric color; occupied seats have more visual variance from a person), and
   (b) skin/clothing-tone pixel ratio within the region. This is a lightweight,
   trainable-free heuristic. In production with labeled data, this would be
   replaced by a small CNN classifier (e.g. MobileNetV2 fine-tuned on cropped
   seat patches) for robustness to lighting/occlusion variation.

3. **Output**: total/occupied/unoccupied counts, an annotated image
   (green = unoccupied, red = occupied), and a `seat_results.json` with
   per-seat bounding boxes and classifications.

## Files
- `generate_test_image.py` — creates the synthetic test input + ground truth
- `seat_occupancy.py` — core detection + classification pipeline (CLI runnable)
- `app.py` — Streamlit UI for uploading an image and viewing results
- `theater_input.png` — sample test image
- `annotated_output.png` — sample output

## How to run
```
python3 generate_test_image.py      # creates sample input
python3 seat_occupancy.py theater_input.png
# or, for the UI:
streamlit run app.py
```

## Limitations & what I'd do with more time
- Detection assumes seats are visually distinct from the background and roughly
  seat-sized; a real photo (angled, varied lighting, occlusion) would need a
  trained object detector (YOLOv8 fine-tuned on seat crops) rather than
  threshold + contour detection.
- Classification heuristic is color/variance-based; a trained binary classifier
  on real seat-crop images (occupied vs unoccupied) would generalize far better.
- No perspective correction — assumes a roughly frontal/top-down camera angle.
