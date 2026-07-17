import streamlit as st
import cv2
import numpy as np
from PIL import Image
from seat_occupancy import detect_seats, classify_seat

st.set_page_config(page_title="Theater Seat Occupancy Detector", layout="wide")
st.title("🎭 Theater Seat Occupancy Detection")
st.write("Upload a theater seating image to detect and classify seat occupancy.")

uploaded = st.file_uploader("Upload theater image", type=["png", "jpg", "jpeg"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    boxes = detect_seats(image)
    annotated = image.copy()
    occupied_count = 0

    for box in boxes:
        x, y, w, h = box
        occupied = classify_seat(image, box)
        color = (0, 0, 255) if occupied else (0, 200, 0)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        if occupied:
            occupied_count += 1

    total = len(boxes)
    unoccupied_count = total - occupied_count

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
    with col2:
        st.subheader("Detected (Red = Occupied, Green = Unoccupied)")
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Seats", total)
    m2.metric("Occupied Seats", occupied_count)
    m3.metric("Unoccupied Seats", unoccupied_count)
else:
    st.info("Upload an image to begin.")
