import streamlit as st
import cv2
import os
from PIL import Image
import io
import time
import numpy as np
from fpdf import FPDF

# Create directories
os.makedirs('uploads', exist_ok=True)
os.makedirs('processed', exist_ok=True)

# Page config
st.set_page_config(page_title="Advanced Image Processor", layout="wide")
st.title("Advanced Image Color Converter")
st.write("Upload or capture an image and apply processing like grayscale, binary, blur, edge detection, invert, and AI enhancement")

# --- Select Input Method ---
input_method = st.selectbox("Choose input method:", ("Upload Image", "Use Camera"))
cancel_camera = False

uploaded_file = None
camera_image = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
elif input_method == "Use Camera":
    camera_image = st.camera_input("Take a photo")
    if st.button("Cancel Camera"):
        cancel_camera = True

# Get image from either source
input_image = None
file_name = ""
file_bytes = None

if uploaded_file and not cancel_camera:
    file_bytes = uploaded_file.getvalue()
    input_image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    file_name = uploaded_file.name
elif camera_image and not cancel_camera:
    file_bytes = camera_image.getvalue()
    input_image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    file_name = f"camera_{int(time.time())}.png"

# Conversion options
option = st.radio("Select conversion type:", ("Grayscale", "Binary", "Edge Detection", "Blur", "Invert", "AI Enhance"))
threshold = 127
if option == "Binary":
    threshold = st.slider("Select threshold value:", 0, 255, 127)

# Brightness & Contrast
apply_bc = st.checkbox("Adjust Brightness/Contrast")
if apply_bc:
    brightness = st.slider("Brightness", -100, 100, 0)
    contrast = st.slider("Contrast", -100, 100, 0)

# Start processing
if input_image is not None:
    timestr = time.strftime("%Y%m%d-%H%M%S")
    upload_path = os.path.join('uploads', f"{timestr}_{file_name}")
    with open(upload_path, "wb") as f:
        f.write(file_bytes)

    original_image = input_image.copy()

    # Brightness and contrast
    if apply_bc:
        input_image = cv2.convertScaleAbs(input_image, alpha=1 + contrast / 100, beta=brightness)

    # Process based on selected option
    if option == "Grayscale":
        processed_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    elif option == "Binary":
        gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        _, processed_image = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    elif option == "Edge Detection":
        gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        processed_image = cv2.Canny(gray, 100, 200)
    elif option == "Blur":
        processed_image = cv2.GaussianBlur(input_image, (15, 15), 0)
    elif option == "Invert":
        processed_image = cv2.bitwise_not(input_image)
    elif option == "AI Enhance":
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        processed_image = cv2.filter2D(src=input_image, ddepth=-1, kernel=kernel)

    # Layout columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("Original")
        st.image(original_image, channels="BGR")

    with col2:
        st.header("Processed")
        st.image(processed_image, use_column_width=True)

        processed_path = os.path.join('processed', f"{timestr}_{option.lower()}_{file_name}")
        if len(processed_image.shape) == 2:
            pil_img = Image.fromarray(processed_image)
        else:
            pil_img = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

        pil_img.save(processed_path)

        # PNG download
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button("Download PNG", data=byte_im, file_name=f"{option.lower()}_{file_name}.png", mime="image/png")

        # PDF download
        pdf_path = f"{processed_path}.pdf"
        pdf = FPDF()
        pdf.add_page()
        pil_img.save("temp_img.png")
        pdf.image("temp_img.png", x=10, y=10, w=180)
        pdf.output(pdf_path)
        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF", data=f, file_name=f"{option.lower()}_{file_name}.pdf", mime="application/pdf")

    with col3:
        st.header("Info")
        h, w = original_image.shape[:2]
        channels = original_image.shape[2] if len(original_image.shape) == 3 else 1
        st.write(f"Filename: `{file_name}`")
        st.write(f"Dimensions: `{w} x {h}`")
        st.write(f"Channels: `{channels}`")
        st.write(f"Uploaded at: `{timestr}`")
        # st.success("File successfully processed")

