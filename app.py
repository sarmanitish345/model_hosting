import streamlit as st  
import os
import cv2
from ultralytics import YOLO
import tempfile
from PIL import Image
from pathlib import Path


# Page title
st.set_page_config(page_title="Elephant Detector", layout="centered")
st.title("🐘 Elephant Detector using YOLOv8")
st.caption("Upload an image or video to detect elephants using your trained model.")

# Load model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# File uploader
file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4"])

# Handle input
if file is not None:
    suffix = Path(file.name).suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(file.read())
        temp_path = temp_file.name

    if suffix in [".jpg", ".jpeg", ".png"]:
        st.image(temp_path, caption="Uploaded Image", use_column_width=True)
        results = model.predict(source=temp_path, save=False, conf=0.25)
        for r in results:
            result_img = r.plot()
            st.image(result_img, caption="Prediction", use_column_width=True)

    elif suffix == ".mp4":
        st.video(temp_path)
        st.write("Running inference on video...")
        save_dir = "output"
        results = model.predict(source=temp_path, save=True, project=save_dir, name="streamlit", conf=0.25)
        output_video_path = f"{save_dir}/streamlit/{Path(temp_path).name}"
        st.success("Detection Complete!")
        st.video(output_video_path)
