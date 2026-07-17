"""
Theater Seat Occupancy Detection
---------------------------------
Pipeline:
 1. Detect seat regions in the image using contour detection on a color-thresholded mask.
 2. For each detected seat region, classify Occupied / Unoccupied using color-variance
    heuristics (an empty seat is a near-uniform fabric color; an occupied seat has a
    person breaking that uniformity with skin/clothing tones).
 3. Aggregate and report total / occupied / unoccupied counts.
 4. Save an annotated output image (green box = unoccupied, red box = occupied).

Usage:
    python3 seat_occupancy.py <input_image_path>
"""
import cv2
import numpy as np
import sys
import json

def detect_seats(image, expected_seats=100, expected_rows=10, expected_cols=10):
    """
    Detects seat bounding boxes.
    Approach: the seats sit on a much darker background, so we threshold to find
    seat-colored blobs, then find contours and filter by area to keep seat-sized
    regions. Boxes are then snapped into a row/col grid for consistent ordering.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Seats are brighter than the background -> simple binary threshold
    _, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        # filter out noise / non-seat-sized blobs
        if 1500 < area < 6000:
            boxes.append((x, y, w, h))

    # Sort into reading order: top-to-bottom, then left-to-right, using row clustering
    boxes.sort(key=lambda b: (round(b[1] / 40), b[0]))

    if len(boxes) != expected_seats:
        print(f"[warning] Detected {len(boxes)} seat regions, expected {expected_seats}. "
              f"Proceeding with detected count.")

    return boxes


def classify_seat(image, box):
    """
    Classifies a single seat region as Occupied or Unoccupied.

    Heuristic: crop the seat region and look at color variance / hue spread.
    - Unoccupied seats are near-uniform fabric color -> low std deviation.
    - Occupied seats have a person (head + shoulders) with different color/texture
      -> higher std deviation and presence of skin-tone pixels.
    """
    x, y, w, h = box
    crop = image[y:y + h, x:x + w]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Overall color variance in the crop (occupied seats are less uniform)
    std_dev = np.std(hsv[:, :, 0])  # hue channel std

    # Skin/clothing tone detection (broad range covering the synthetic person colors)
    lower_skin = np.array([0, 30, 60])
    upper_skin = np.array([30, 200, 255])
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_ratio = np.sum(skin_mask > 0) / (w * h)

    is_occupied = bool((std_dev > 8) or (skin_ratio > 0.12))
    return is_occupied


def run(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    boxes = detect_seats(image)
    annotated = image.copy()

    results = []
    occupied_count = 0

    for i, box in enumerate(boxes):
        x, y, w, h = box
        occupied = classify_seat(image, box)
        results.append({"seat_id": i + 1, "bbox": [int(x), int(y), int(w), int(h)], "occupied": occupied})

        color = (0, 0, 255) if occupied else (0, 200, 0)  # red=occupied, green=unoccupied (BGR)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        if occupied:
            occupied_count += 1

    total = len(results)
    unoccupied_count = total - occupied_count

    summary = {
        "Total Seats": total,
        "Occupied Seats": occupied_count,
        "Unoccupied Seats": unoccupied_count
    }

    cv2.imwrite("/home/claude/task2_seats/annotated_output.png", annotated)
    with open("/home/claude/task2_seats/seat_results.json", "w") as f:
        json.dump({"summary": summary, "seats": results}, f, indent=2)

    return summary


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/task2_seats/theater_input.png"
    summary = run(path)
    print(f"Total Seats      : {summary['Total Seats']}")
    print(f"Occupied Seats   : {summary['Occupied Seats']}")
    print(f"Unoccupied Seats : {summary['Unoccupied Seats']}")
