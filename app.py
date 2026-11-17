import warnings
import os

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image, ImageDraw

# Multi-page PDF support via PyMuPDF
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

st.set_page_config(
    page_title="Deep BiLSTM OCR Scanner",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Deep BiLSTM Text & Number Plate Extractor")
st.write(
    "Extracts complete text from **dense documents, multi-page PDFs, low-contrast images, "
    "and road vehicle license plates** using multi-pass contrast enhancement + "
    "**CRNN-BiLSTM** sequence recognition."
)

@st.cache_resource
def load_bilstm_ocr():
    return easyocr.Reader(
        lang_list=['en'],
        gpu=False,
        recog_network='standard',  # CNN + BiLSTM + CTC decoder
        verbose=False
    )

with st.spinner("Loading Deep BiLSTM OCR Engine..."):
    reader = load_bilstm_ocr()


def enhance_image_for_deep_ocr(img_rgb, upscale_factor=2.0):
    if upscale_factor > 1.0:
        h, w = img_rgb.shape[:2]
        if max(h, w) < 2200:
            img_rgb = cv2.resize(
                img_rgb,
                None,
                fx=upscale_factor,
                fy=upscale_factor,
                interpolation=cv2.INTER_CUBIC
            )

    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((cl, a_channel, b_channel))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    gaussian = cv2.GaussianBlur(enhanced_rgb, (0, 0), 2.0)
    sharpened = cv2.addWeighted(enhanced_rgb, 1.5, gaussian, -0.5, 0)
    return sharpened


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return 0.0

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter_area / float(boxA_area + boxB_area - inter_area + 1e-6)


def deep_scan_single_image(pil_img, conf_thresh, deep_mode, draw_boxes):
    orig_rgb = np.array(pil_img.convert("RGB"))
    orig_h, orig_w = orig_rgb.shape[:2]

    ocr_kwargs = dict(
        detail=1,
        paragraph=False,
        text_threshold=0.5,
        low_text=0.3,
        link_threshold=0.3,
        canvas_size=2560,
        mag_ratio=1.5
    )

    # Pass 1: Original Image
    raw_detections = []
    for (bbox, text, prob) in reader.readtext(orig_rgb, **ocr_kwargs):
        if prob >= conf_thresh and text.strip():
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            rect = [min(xs), min(ys), max(xs), max(ys)]
            raw_detections.append((rect, text.strip(), prob))

    # Pass 2: Enhanced + Upscaled + Sharpened Image (for vehicle plates & faint text)
    if deep_mode:
        scale = 2.0 if max(orig_h, orig_w) < 2200 else 1.0
        enhanced_rgb = enhance_image_for_deep_ocr(orig_rgb, upscale_factor=scale)
        for (bbox, text, prob) in reader.readtext(enhanced_rgb, **ocr_kwargs):
            if prob >= conf_thresh and text.strip():
                xs = [p[0] / scale for p in bbox]
                ys = [p[1] / scale for p in bbox]
                rect = [min(xs), min(ys), max(xs), max(ys)]
                raw_detections.append((rect, text.strip(), prob))

    # Deduplicate overlapping boxes keeping highest confidence
    raw_detections.sort(key=lambda x: x[2], reverse=True)
    final_detections = []
    for rect, text, prob in raw_detections:
        is_duplicate = False
        for kept_rect, _, _ in final_detections:
            if compute_iou(rect, kept_rect) > 0.45:
                is_duplicate = True
                break
        if not is_duplicate:
            final_detections.append((rect, text, prob))

    # Sort top-to-bottom, left-to-right
    final_detections.sort(key=lambda item: (round(item[0][1] / 20), item[0][0]))

    annotated = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    lines = []

    for rect, text, _ in final_detections:
        lines.append(text)
        if draw_boxes:
            draw.rectangle(rect, outline="#00FF00", width=3)

    return (annotated if draw_boxes else pil_img), lines


# Sidebar Settings
st.sidebar.header("Deep Analysis Settings")
deep_mode = st.sidebar.checkbox(
    "Enable Deep Multi-Pass Analysis (2x Upscale + CLAHE for Number Plates & Faint Text)",
    value=True
)
draw_boxes = st.sidebar.checkbox("Highlight Detected Text Regions", value=True)
conf_thresh = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.15,
    step=0.05,
    help="Lower values capture faint or distant vehicle license plate characters."
)

# Input Mode Selector
input_mode = st.radio(
    "Choose Input Source:",
    (
        "📁 Upload Image or Multi-Page PDF",
        "📷 Live Camera (Mobile Front/Back or Webcam)"
    ),
    horizontal=True
)

pages_to_scan = []

if input_mode == "📁 Upload Image or Multi-Page PDF":
    uploaded_file = st.file_uploader(
        "Upload an Image (JPG, PNG, WEBP) or Full Document (PDF)",
        type=["jpg", "jpeg", "png", "webp", "bmp", "tiff", "pdf"]
    )
    if uploaded_file is not None:
        if uploaded_file.name.lower().endswith(".pdf"):
            if not PDF_SUPPORT:
                st.error("PDF support requires PyMuPDF (`pymupdf`) in requirements.txt.")
            else:
                pdf_bytes = uploaded_file.read()
                pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                for page_num in range(len(pdf_doc)):
                    page = pdf_doc.load_page(page_num)
                    pix = page.get_pixmap(dpi=250)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    pages_to_scan.append((f"Page {page_num + 1}", img))
        else:
            pages_to_scan.append(("Uploaded Image", Image.open(uploaded_file).convert("RGB")))
else:
    # Automatically supports mobile front & rear camera switching
    camera_photo = st.camera_input("Capture text or vehicle number plate with camera")
    if camera_photo is not None:
        pages_to_scan.append(("Captured Image", Image.open(camera_photo).convert("RGB")))

# Execute Extraction & Render Results
if pages_to_scan:
    all_extracted_text = []
    first_annotated_img = None

    with st.spinner("Running Deep BiLSTM Multi-Pass Text Extraction..."):
        for label, pil_img in pages_to_scan:
            annotated_img, extracted_lines = deep_scan_single_image(
                pil_img, conf_thresh, deep_mode, draw_boxes
            )
            if first_annotated_img is None:
                first_annotated_img = annotated_img

            if len(pages_to_scan) > 1:
                all_extracted_text.append(f"--- {label} ---")

            if extracted_lines:
                all_extracted_text.extend(extracted_lines)
            else:
                all_extracted_text.append("[No readable text detected on this page]")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Annotated Detection Map")
        st.image(first_annotated_img, use_container_width=True)

    with col2:
        st.subheader("Complete Extracted Text")
        final_output = "\n".join(all_extracted_text)
        st.text_area("Detected Content (Click inside & Ctrl+A to copy):", value=final_output, height=340)
        st.download_button(
            label="📥 Download Complete Extracted Text (.txt)",
            data=final_output,
            file_name="extracted_text.txt",
            mime="text/plain"
        )