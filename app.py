from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from PIL import Image, ImageDraw
import streamlit as st
import tensorflow as tf

# OCR
try:
    import easyocr
except Exception:
    easyocr = None

# PDF report
try:
    from fpdf import FPDF
except Exception:
    FPDF = None


# --------------------------------------------------
# Streamlit page setup
# --------------------------------------------------

st.set_page_config(
    page_title="DesignSense AI",
    layout="wide"
)

st.markdown(
    """
    <style>
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, #F8FAFC 0%, #EEF2FF 45%, #FDF2F8 100%);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #312E81 100%);
    }

    [data-testid="stSidebar"] * {
        color: #F9FAFB;
    }

    /* Sidebar status boxes */
    [data-testid="stSidebar"] .stAlert {
        border-radius: 14px;
        border: none;
    }

    /* Main title */
    h1 {
        color: #111827;
        font-size: 3.2rem !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }

    /* Subtitles and section headers */
    h2, h3 {
        color: #1F2937;
        font-weight: 750 !important;
    }

    /* Caption */
    [data-testid="stCaptionContainer"] {
        color: #4B5563;
        font-size: 1.05rem;
    }

    /* Upload box */
    [data-testid="stFileUploader"] section {
        background: #FFFFFF;
        border: 2px dashed #A78BFA;
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 8px 24px rgba(124, 58, 237, 0.10);
    }

    [data-testid="stFileUploader"] button {
        background: linear-gradient(90deg, #7C3AED, #DB2777);
        color: white;
        border-radius: 10px;
        border: none;
        font-weight: 700;
    }

    /* Image cards */
    [data-testid="stImage"] {
        border-radius: 18px;
        overflow: hidden;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        padding: 18px;
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(17, 24, 39, 0.08);
    }

    [data-testid="metric-container"] label {
        color: #6B7280 !important;
        font-weight: 700;
    }

    [data-testid="metric-container"] div {
        color: #111827 !important;
    }

    /* Tables */
    [data-testid="stTable"] {
        background: #FFFFFF;
        border-radius: 18px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(17, 24, 39, 0.08);
    }

    /* Text area */
    textarea {
        border-radius: 14px !important;
        border: 1px solid #DDD6FE !important;
        background: #FFFFFF !important;
    }

    /* Download button */
    .stDownloadButton button {
        background: linear-gradient(90deg, #7C3AED, #DB2777);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 0.75rem 1.2rem;
        font-weight: 700;
    }

    .stDownloadButton button:hover {
        background: linear-gradient(90deg, #6D28D9, #BE185D);
        color: white;
    }

    /* General button */
    .stButton button {
        border-radius: 12px;
        font-weight: 700;
    }

    /* Suggestions spacing */
    ul, li, p {
        color: #1F2937;
        font-size: 1rem;
    }

    /* Remove top empty spacing slightly */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="
        background: white;
        padding: 32px 36px;
        border-radius: 24px;
        box-shadow: 0 16px 40px rgba(17, 24, 39, 0.10);
        margin-bottom: 28px;
        border: 1px solid #E5E7EB;
    ">
        <h1 style="margin-bottom: 8px;">DesignSense AI</h1>
        <p style="
            font-size: 1.1rem;
            color: #4B5563;
            margin-bottom: 0;
        ">
            Deep Learning-Based Visual Design Quality and Accessibility Checker
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Main settings
# --------------------------------------------------

MODEL_PATH = Path("model/design_quality_model.keras")

# Important: this order must match your training code
CLASS_NAMES = ["average", "good", "poor"]

IMG_SIZE = (224, 224)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

@st.cache_resource
def load_design_model():
    if not MODEL_PATH.exists():
        return None

    try:
        # safe_mode=False helps if your model contains Lambda layers
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            safe_mode=False
        )
        return model
    except TypeError:
        # For older TensorFlow versions
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )
        return model
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None


# --------------------------------------------------
# Load OCR model
# --------------------------------------------------

@st.cache_resource
def load_ocr_reader():
    if easyocr is None:
        return None

    try:
        reader = easyocr.Reader(["en"], gpu=False)
        return reader
    except Exception as e:
        st.warning(f"OCR could not be loaded: {e}")
        return None


# --------------------------------------------------
# Deep learning prediction
# --------------------------------------------------

def predict_design_quality(image: Image.Image, model) -> Tuple[str, float, List[float]]:
    if model is None:
        return "model_not_found", 0.0, []

    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.array(image).astype("float32")
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    predicted_class = CLASS_NAMES[predicted_index]

    return predicted_class, confidence, prediction.tolist()


# --------------------------------------------------
# OCR text extraction
# --------------------------------------------------

def extract_text_from_image(image: Image.Image) -> List[Dict[str, Any]]:
    reader = load_ocr_reader()

    if reader is None:
        return []

    image_array = np.array(image.convert("RGB"))

    try:
        results = reader.readtext(image_array, detail=1)
    except Exception:
        return []

    text_blocks = []

    for bbox, text, confidence in results:
        if text and str(text).strip():
            text_blocks.append({
                "text": str(text).strip(),
                "confidence": float(confidence),
                "bbox": bbox
            })

    return text_blocks


# --------------------------------------------------
# Bounding box helper
# --------------------------------------------------

def bbox_to_rectangle(bbox):
    x_values = [int(point[0]) for point in bbox]
    y_values = [int(point[1]) for point in bbox]

    x1 = max(0, min(x_values))
    y1 = max(0, min(y_values))
    x2 = max(x_values)
    y2 = max(y_values)

    return x1, y1, x2, y2


# --------------------------------------------------
# Text density analysis
# --------------------------------------------------

def analyse_text_density(image: Image.Image, text_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    width, height = image.size
    image_area = width * height

    total_text_area = 0

    for block in text_blocks:
        x1, y1, x2, y2 = bbox_to_rectangle(block["bbox"])
        text_area = max(0, x2 - x1) * max(0, y2 - y1)
        total_text_area += text_area

    text_area_ratio = total_text_area / image_area if image_area > 0 else 0

    extracted_text = " ".join([block["text"] for block in text_blocks])
    word_count = len(extracted_text.split())
    text_block_count = len(text_blocks)

    if text_block_count <= 6 and text_area_ratio < 0.08:
        density_level = "Low"
    elif text_block_count <= 16 and text_area_ratio < 0.18:
        density_level = "Medium"
    else:
        density_level = "High"

    return {
        "density_level": density_level,
        "text_block_count": text_block_count,
        "word_count": word_count,
        "text_area_ratio": round(text_area_ratio, 3),
        "extracted_text": extracted_text
    }


# --------------------------------------------------
# Contrast and accessibility functions
# --------------------------------------------------

def srgb_to_linear(value):
    value = value / 255.0

    if value <= 0.03928:
        return value / 12.92

    return ((value + 0.055) / 1.055) ** 2.4


def relative_luminance(rgb):
    r, g, b = rgb

    return (
        0.2126 * srgb_to_linear(r)
        + 0.7152 * srgb_to_linear(g)
        + 0.0722 * srgb_to_linear(b)
    )


def calculate_contrast_ratio(rgb1, rgb2):
    lum1 = relative_luminance(rgb1)
    lum2 = relative_luminance(rgb2)

    lighter = max(lum1, lum2)
    darker = min(lum1, lum2)

    return (lighter + 0.05) / (darker + 0.05)


def estimate_text_contrast(image: Image.Image, bbox) -> float:
    image = image.convert("RGB")

    width, height = image.size

    x1, y1, x2, y2 = bbox_to_rectangle(bbox)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    if x2 <= x1 or y2 <= y1:
        return 1.0

    crop = image.crop((x1, y1, x2, y2))
    pixels = np.array(crop).reshape(-1, 3)

    if len(pixels) < 10:
        return 1.0

    luminance_values = np.array([
        relative_luminance(tuple(pixel))
        for pixel in pixels
    ])

    dark_count = max(1, len(pixels) // 10)
    light_count = max(1, len(pixels) // 10)

    dark_indices = np.argsort(luminance_values)[:dark_count]
    light_indices = np.argsort(luminance_values)[-light_count:]

    dark_colour = tuple(np.mean(pixels[dark_indices], axis=0))
    light_colour = tuple(np.mean(pixels[light_indices], axis=0))

    contrast_ratio = calculate_contrast_ratio(dark_colour, light_colour)

    return round(float(contrast_ratio), 2)


def analyse_contrast_and_accessibility(image: Image.Image, text_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    ratios = []

    for block in text_blocks:
        ratio = estimate_text_contrast(image, block["bbox"])
        ratios.append(ratio)

    if len(ratios) == 0:
        return {
            "contrast_level": "Unknown",
            "average_contrast_ratio": 0,
            "minimum_contrast_ratio": 0,
            "accessibility_score": 50
        }

    average_ratio = round(float(np.mean(ratios)), 2)
    minimum_ratio = round(float(np.min(ratios)), 2)

    if average_ratio >= 4.5:
        contrast_level = "Good"
        accessibility_score = 90
    elif average_ratio >= 3.0:
        contrast_level = "Moderate"
        accessibility_score = 70
    else:
        contrast_level = "Poor"
        accessibility_score = 45

    return {
        "contrast_level": contrast_level,
        "average_contrast_ratio": average_ratio,
        "minimum_contrast_ratio": minimum_ratio,
        "accessibility_score": accessibility_score
    }


# --------------------------------------------------
# Visual hierarchy estimate
# --------------------------------------------------

def analyse_visual_hierarchy(design_quality, text_density, contrast_result):
    score = 75

    if design_quality == "good":
        score += 10
    elif design_quality == "average":
        score -= 5
    elif design_quality == "poor":
        score -= 18

    if text_density["density_level"] == "High":
        score -= 20
    elif text_density["density_level"] == "Medium":
        score -= 8

    if contrast_result["contrast_level"] == "Poor":
        score -= 15
    elif contrast_result["contrast_level"] == "Moderate":
        score -= 5

    score = max(0, min(100, score))

    if score >= 80:
        hierarchy_level = "Strong"
    elif score >= 60:
        hierarchy_level = "Moderate"
    else:
        hierarchy_level = "Weak"

    return {
        "hierarchy_score": score,
        "hierarchy_level": hierarchy_level
    }


# --------------------------------------------------
# Overall score
# --------------------------------------------------

def calculate_overall_score(design_quality, text_density, contrast_result, hierarchy_result):
    design_score_map = {
        "good": 90,
        "average": 68,
        "poor": 42,
        "model_not_found": 55
    }

    density_score_map = {
        "Low": 90,
        "Medium": 72,
        "High": 45
    }

    design_score = design_score_map.get(design_quality, 55)
    density_score = density_score_map.get(text_density["density_level"], 60)
    contrast_score = contrast_result["accessibility_score"]
    hierarchy_score = hierarchy_result["hierarchy_score"]

    overall_score = (
        design_score * 0.35
        + density_score * 0.20
        + contrast_score * 0.25
        + hierarchy_score * 0.20
    )

    return int(round(overall_score))


# --------------------------------------------------
# Suggestion generator
# --------------------------------------------------

def generate_design_suggestions(
    design_quality,
    text_density,
    contrast_result,
    hierarchy_result,
    overall_score
):
    suggestions = []

    if design_quality == "model_not_found":
        suggestions.append(
            "Trained model not found. Place design_quality_model.keras inside the model folder."
        )
    elif design_quality == "good":
        suggestions.append(
            "The design has a good visual structure. Minor refinements can make it stronger."
        )
    elif design_quality == "average":
        suggestions.append(
            "The design is usable, but spacing, hierarchy and contrast can be improved."
        )
    elif design_quality == "poor":
        suggestions.append(
            "The design needs improvement in layout, readability and visual balance."
        )

    if text_density["density_level"] == "High":
        suggestions.append(
            "Reduce text content and keep only the most important information."
        )
        suggestions.append(
            "Move extra details to a caption, description, QR code or second page."
        )
    elif text_density["density_level"] == "Medium":
        suggestions.append(
            "Text density is moderate. Make sure the title and key details are clearly visible."
        )
    else:
        suggestions.append(
            "Text density is low, which usually helps quick readability."
        )

    if contrast_result["contrast_level"] == "Poor":
        suggestions.append(
            "Improve colour contrast between text and background."
        )
        suggestions.append(
            "Use darker text on light backgrounds or lighter text on dark backgrounds."
        )
    elif contrast_result["contrast_level"] == "Moderate":
        suggestions.append(
            "Contrast is acceptable, but important text can be made stronger."
        )
    elif contrast_result["contrast_level"] == "Good":
        suggestions.append(
            "Text contrast appears strong in detected areas."
        )
    else:
        suggestions.append(
            "OCR could not detect clear text areas. Use a sharper image for better analysis."
        )

    if hierarchy_result["hierarchy_level"] == "Weak":
        suggestions.append(
            "Make the main title larger and reduce the visual weight of less important elements."
        )
    elif hierarchy_result["hierarchy_level"] == "Moderate":
        suggestions.append(
            "Improve visual hierarchy by making the main message more prominent."
        )

    if overall_score >= 80:
        suggestions.append(
            "Overall, the design appears strong and presentation-ready."
        )
    elif overall_score >= 60:
        suggestions.append(
            "Overall, the design is acceptable but can be refined further."
        )
    else:
        suggestions.append(
            "Overall, the design needs improvement before publishing or sharing."
        )

    return suggestions


# --------------------------------------------------
# Draw OCR boxes
# --------------------------------------------------

def draw_text_boxes(image: Image.Image, text_blocks: List[Dict[str, Any]]):
    image_copy = image.convert("RGB").copy()
    draw = ImageDraw.Draw(image_copy)

    for block in text_blocks:
        x1, y1, x2, y2 = bbox_to_rectangle(block["bbox"])
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    return image_copy


# --------------------------------------------------
# PDF report
# --------------------------------------------------

def clean_pdf_text(text):
    return str(text).encode("latin-1", "ignore").decode("latin-1")


def create_pdf_report(report, suggestions):
    if FPDF is None:
        return None

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "DesignSense AI Report", ln=True)

    pdf.ln(5)

    pdf.set_font("Arial", "", 11)

    for key, value in report.items():
        pdf.set_font("Arial", "B", 11)
        pdf.cell(65, 8, clean_pdf_text(str(key) + ":"), border=0)

        pdf.set_font("Arial", "", 11)
        pdf.cell(0, 8, clean_pdf_text(str(value)), ln=True)

    pdf.ln(5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Suggestions", ln=True)

    pdf.set_font("Arial", "", 11)

    for i, suggestion in enumerate(suggestions, start=1):
        pdf.multi_cell(0, 7, clean_pdf_text(f"{i}. {suggestion}"))

    pdf_data = pdf.output(dest="S").encode("latin-1")

    return pdf_data


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.header("Project Status")

    if MODEL_PATH.exists():
        st.success("Model found")
    else:
        st.warning("Model not found")
        st.write("Expected path:")
        st.code("model/design_quality_model.keras")

    if easyocr is not None:
        st.success("EasyOCR available")
    else:
        st.warning("EasyOCR not installed")

    st.markdown("---")
    st.write("Upload a poster, flyer, brochure, certificate, web banner or social media creative.")


# --------------------------------------------------
# Image upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a design image",
    type=["png", "jpg", "jpeg", "webp"]
)

if uploaded_file is None:
    st.info("Upload a design image to start.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")

model = load_design_model()


# --------------------------------------------------
# Run full analysis
# --------------------------------------------------

with st.spinner("Analysing design..."):
    design_quality, confidence, probabilities = predict_design_quality(image, model)

    text_blocks = extract_text_from_image(image)

    text_density = analyse_text_density(image, text_blocks)

    contrast_result = analyse_contrast_and_accessibility(image, text_blocks)

    hierarchy_result = analyse_visual_hierarchy(
        design_quality,
        text_density,
        contrast_result
    )

    overall_score = calculate_overall_score(
        design_quality,
        text_density,
        contrast_result,
        hierarchy_result
    )

    suggestions = generate_design_suggestions(
        design_quality,
        text_density,
        contrast_result,
        hierarchy_result,
        overall_score
    )


# --------------------------------------------------
# Display results
# --------------------------------------------------

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Uploaded Design")
    st.image(image, use_container_width=True)

    if len(text_blocks) > 0:
        st.subheader("Detected Text Areas")
        boxed_image = draw_text_boxes(image, text_blocks)
        st.image(boxed_image, use_container_width=True)
    else:
        st.warning("No text detected by OCR.")

with right_col:
    st.subheader("Final DesignSense AI Report")

    metric_1, metric_2, metric_3 = st.columns(3)

    display_quality = design_quality.replace("_", " ").title()

    metric_1.metric("Design Quality", display_quality)
    metric_2.metric("Overall Score", f"{overall_score}/100")
    metric_3.metric("Accessibility", f"{contrast_result['accessibility_score']}/100")

    st.markdown("### Analysis Details")

    report = {
        "Design Quality": display_quality,
        "Model Confidence": f"{confidence * 100:.2f}%" if confidence else "N/A",
        "Text Density": text_density["density_level"],
        "Detected Text Blocks": text_density["text_block_count"],
        "Word Count": text_density["word_count"],
        "Text Area Ratio": text_density["text_area_ratio"],
        "Contrast Level": contrast_result["contrast_level"],
        "Average Contrast Ratio": contrast_result["average_contrast_ratio"],
        "Minimum Contrast Ratio": contrast_result["minimum_contrast_ratio"],
        "Accessibility Score": f"{contrast_result['accessibility_score']}/100",
        "Visual Hierarchy": hierarchy_result["hierarchy_level"],
        "Visual Hierarchy Score": f"{hierarchy_result['hierarchy_score']}/100",
        "Overall Score": f"{overall_score}/100"
    }

    st.table(report)

    if probabilities:
        st.markdown("### Model Class Probabilities")

        probability_report = {
            CLASS_NAMES[i].title(): f"{probabilities[i] * 100:.2f}%"
            for i in range(len(CLASS_NAMES))
        }

        st.table(probability_report)


# --------------------------------------------------
# Extracted text
# --------------------------------------------------

st.markdown("## Extracted Text")

if text_density["extracted_text"]:
    st.text_area(
        "OCR Output",
        text_density["extracted_text"],
        height=150
    )
else:
    st.write("No readable text detected.")


# --------------------------------------------------
# Suggestions
# --------------------------------------------------

st.markdown("## Design Suggestions")

for suggestion in suggestions:
    st.write(f"- {suggestion}")


# --------------------------------------------------
# PDF download
# --------------------------------------------------

pdf_report = create_pdf_report(report, suggestions)

if pdf_report is not None:
    st.download_button(
        label="Download PDF Report",
        data=pdf_report,
        file_name="designsense_ai_report.pdf",
        mime="application/pdf"
    )
else:
    st.info("PDF download is unavailable because fpdf is not installed.")