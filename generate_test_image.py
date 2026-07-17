"""
Generates a synthetic theater image: 10x10 grid of seats (100 total).
Some seats are drawn as "occupied" (darker seat + person-shaped blob on top),
others as "unoccupied" (plain seat color).
This simulates a real input image for the seat-detection pipeline.
"""
import numpy as np
from PIL import Image, ImageDraw
import random
import json

random.seed(42)

ROWS, COLS = 10, 10
SEAT_W, SEAT_H = 60, 50
GAP_X, GAP_Y = 15, 20
MARGIN_X, MARGIN_Y = 40, 40

img_w = MARGIN_X * 2 + COLS * (SEAT_W + GAP_X) - GAP_X
img_h = MARGIN_Y * 2 + ROWS * (SEAT_H + GAP_Y) - GAP_Y

img = Image.new("RGB", (img_w, img_h), (30, 30, 35))  # dark theater background
draw = ImageDraw.Draw(img)

UNOCCUPIED_COLOR = (70, 90, 160)   # blue seat fabric
OCCUPIED_SEAT_COLOR = (55, 70, 130)  # slightly darker seat (partially hidden)
PERSON_COLORS = [(194, 154, 108), (120, 90, 70), (210, 180, 140), (90, 60, 50)]  # skin/clothing tones

ground_truth = []  # store actual occupied/unoccupied for validation

for r in range(ROWS):
    for c in range(COLS):
        x0 = MARGIN_X + c * (SEAT_W + GAP_X)
        y0 = MARGIN_Y + r * (SEAT_H + GAP_Y)
        x1, y1 = x0 + SEAT_W, y0 + SEAT_H

        is_occupied = random.random() < 0.68  # ~68% occupied to mimic expected output

        if is_occupied:
            draw.rounded_rectangle([x0, y0, x1, y1], radius=8, fill=OCCUPIED_SEAT_COLOR)
            # draw a simple "person" blob (head + shoulders) on top of the seat
            person_color = random.choice(PERSON_COLORS)
            head_r = 12
            cx = (x0 + x1) // 2
            head_cy = y0 + 14
            draw.ellipse([cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r], fill=person_color)
            draw.rounded_rectangle([x0 + 8, head_cy + 6, x1 - 8, y1 - 4], radius=6, fill=person_color)
        else:
            draw.rounded_rectangle([x0, y0, x1, y1], radius=8, fill=UNOCCUPIED_COLOR)
            # subtle seat back/cushion lines to look like an empty seat
            draw.line([x0 + 6, y0 + 8, x1 - 6, y0 + 8], fill=(50, 65, 120), width=2)

        ground_truth.append({
            "row": r, "col": c,
            "bbox": [x0, y0, x1, y1],
            "occupied": is_occupied
        })

img.save("/home/claude/task2_seats/theater_input.png")
with open("/home/claude/task2_seats/ground_truth.json", "w") as f:
    json.dump(ground_truth, f, indent=2)

occ_count = sum(1 for s in ground_truth if s["occupied"])
print(f"Generated theater_input.png with {len(ground_truth)} seats")
print(f"Occupied: {occ_count}, Unoccupied: {len(ground_truth) - occ_count}")
