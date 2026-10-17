import os

import io

import json

import base64

import logging
import re

from pathlib import Path

from datetime import datetime

from xml.sax.saxutils import escape



import cv2

import numpy as np

import tensorflow as tf

from PIL import Image, ImageOps

from dotenv import load_dotenv

from flask import Flask, render_template, request, send_file

from openai import OpenAI



import matplotlib

matplotlib.use("Agg")

from matplotlib import colormaps



from reportlab.lib import colors

from reportlab.lib.enums import TA_CENTER

from reportlab.lib.pagesizes import A4

from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from reportlab.lib.units import mm

from reportlab.pdfbase import pdfmetrics

from reportlab.pdfbase.ttfonts import TTFont

from reportlab.platypus import (

    SimpleDocTemplate, Paragraph, Table, TableStyle,

    Image as PDFImage,

)



# =========================================================

# CONFIGURATION

# =========================================================



load_dotenv()



BASE_DIR = Path(__file__).resolve().parent



MODEL_PATH = Path(

    os.getenv(

        "MODEL_PATH",

        str(BASE_DIR / "models" / "efficientnet_finetuned.keras"),

    )

)



if not MODEL_PATH.is_absolute():

    MODEL_PATH = BASE_DIR / MODEL_PATH



IMAGE_SIZE = (224, 224)

MAX_UPLOAD_MB = 15



CLASS_NAMES = [

    "No DR",

    "Mild",

    "Moderate",

    "Severe",

    "Proliferative DR",

]



CLASS_HI = {

    "No DR": "डायबिटिक रेटिनोपैथी नहीं",

    "Mild": "हल्का",

    "Moderate": "मध्यम",

    "Severe": "गंभीर",

    "Proliferative DR": "प्रोलिफेरेटिव डायबिटिक रेटिनोपैथी",

}



ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}



FONT_PATH = (

    BASE_DIR

    / "fonts"

    / "NotoSansDevanagari-VariableFont_wdth,wght.ttf"

)



logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)



app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024



# =========================================================

# LOAD MODEL

# =========================================================



model = None

model_error = None



try:

    if not MODEL_PATH.exists():

        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")



    model = tf.keras.models.load_model(

        str(MODEL_PATH),

        compile=False,

        safe_mode=True,

    )



    logger.info("Model loaded: %s", MODEL_PATH)

    logger.info("Input shape: %s", model.input_shape)

    logger.info("Output shape: %s", model.output_shape)



except Exception as exc:

    model_error = str(exc)

    logger.exception("Failed to load model")



# =========================================================

# GROQ CLIENT

# =========================================================



groq_client = None

api_key = os.getenv("GROQ_API_KEY", "").strip()



if api_key:

    groq_client = OpenAI(

        api_key=api_key,

        base_url="https://api.groq.com/openai/v1",

        timeout=45.0,

        max_retries=1,

    )



GROQ_MODEL = os.getenv(

    "GROQ_MODEL",

    "meta-llama/llama-4-scout-17b-16e-instruct",

)



# =========================================================

# IMAGE HELPERS

# =========================================================



def allowed_file(filename):

    return (

        bool(filename)

        and "." in filename

        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

    )





def image_to_base64(image, format_name="PNG"):

    buffer = io.BytesIO()

    image.save(buffer, format=format_name)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")





def preprocess_image(image):

    """

    Model preprocessing:

    RGB, 224x224, float32, pixel range 0-255.

    Do not divide pixels by 255.

    """

    image = ImageOps.exif_transpose(image).convert("RGB")

    resized = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)

    array = np.asarray(resized, dtype=np.float32)

    batch = np.expand_dims(array, axis=0)

    return resized, batch





def get_probabilities(raw_output):

    if isinstance(raw_output, (list, tuple)):

        if len(raw_output) != 1:

            raise ValueError("Expected a single classification output.")

        raw_output = raw_output[0]



    values = np.asarray(raw_output, dtype=np.float64).reshape(-1)



    if values.size != len(CLASS_NAMES):

        raise ValueError(

            f"Expected {len(CLASS_NAMES)} outputs; got {values.size}."

        )



    if not np.all(np.isfinite(values)):

        raise ValueError("Model returned invalid scores.")



    # If output is already a probability distribution, preserve it.
    if (

        np.all(values >= 0)

        and np.all(values <= 1)

        and np.isclose(values.sum(), 1.0, atol=1e-3)

    ):

        return values / max(float(values.sum()), 1e-12)



    # Otherwise interpret outputs as logits.
    shifted = values - np.max(values)

    exp_values = np.exp(shifted)

    return exp_values / exp_values.sum()





# =========================================================

# =========================================================
# RETINAL FUNDUS IMAGE VALIDATION
# Groq vision pre-check; not a clinically validated detector.
# =========================================================


def validate_retinal_image(image, minimum_score=0.80):
    """Return (accepted, score, message) before DR prediction."""
    if groq_client is None:
        return False, 0.0, "Retinal-image validation is unavailable: configure GROQ_API_KEY."
    try:
        image_b64 = image_to_base64(image.convert("RGB"), format_name="JPEG")
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": (
                    "Determine only whether this is a genuine retinal fundus photograph showing "
                    "the retina inside the eye. Reject ordinary photos, screenshots, documents, "
                    "charts, other medical images, and ambiguous images. Do not assess disease. "
                    "Return ONLY JSON: is_fundus (boolean), confidence (number 0 to 1), reason (string). "
                    "Be conservative when uncertain."
                )},
                {"role": "user", "content": [
                    {"type": "text", "text": "Is this a retinal fundus photograph? Return JSON only."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ]},
            ],
            temperature=0,
            max_tokens=120,
        )
        content = response.choices[0].message.content or ""
        match = re.search(r"\{.*\}", content, flags=re.S)
        if not match:
            return False, 0.0, "Could not verify image type. Upload a retinal fundus photograph."
        data = json.loads(match.group(0))
        is_fundus = data.get("is_fundus") is True
        score = min(1.0, max(0.0, float(data.get("confidence", 0.0))))
        reason = str(data.get("reason", "Image type could not be confirmed."))[:200]
        if not is_fundus or score < minimum_score:
            return False, score, "This image could not be confirmed as retinal. Please upload a retinal fundus photograph. " + reason
        return True, score, "Retinal fundus image check passed."
    except Exception:
        logger.exception("Retinal-image validation failed")
        return False, 0.0, "Retinal-image validation failed. Check Groq settings; DR prediction was not run."


# EXPERIMENTAL IMAGE QUALITY CHECKS

# Advisory only; these checks do not block prediction.

# =========================================================



def detect_blur(image, threshold=30.0):

    gray = cv2.cvtColor(

        np.asarray(image.convert("RGB")),

        cv2.COLOR_RGB2GRAY,

    )



    score = float(cv2.Laplacian(gray, cv2.CV_64F).var())



    return {

        "is_blurry": score < threshold,

        "blur_score": round(score, 2),

        "message": (

            "Image may be blurry; consider a sharper image."

            if score < threshold

            else "No strong blur signal detected."

        ),

    }





def detect_crop(image, min_retina_ratio=0.35, edge_margin=0.02):

    """

    Experimental framing heuristic only.

    It does not prove that an image is cropped.

    """

    rgb = np.asarray(image.convert("RGB"))

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)



    mask = gray > 10

    retina_ratio = float(np.mean(mask))



    h, w = mask.shape

    margin_y = max(1, int(h * edge_margin))

    margin_x = max(1, int(w * edge_margin))



    touches_edge = (

        np.any(mask[:margin_y, :])

        or np.any(mask[-margin_y:, :])

        or np.any(mask[:, :margin_x])

        or np.any(mask[:, -margin_x:])

    )



    flagged = retina_ratio < min_retina_ratio or touches_edge



    return {

        "is_cropped": bool(flagged),

        "retina_ratio": round(retina_ratio, 3),

        "message": (

            "Possible crop or unusual framing; review manually."

            if flagged

            else "No obvious framing issue detected."

        ),

    }





# =========================================================

# GRAD-CAM

# =========================================================



def find_last_conv_layer(loaded_model):

    """Find a Conv2D layer inside a nested CNN backbone."""

    for layer in reversed(loaded_model.layers):

        if isinstance(layer, tf.keras.Model):

            try:

                nested_model, nested_layer = find_last_conv_layer(layer)

                return nested_model, nested_layer

            except ValueError:

                pass



        if isinstance(layer, tf.keras.layers.Conv2D):

            return loaded_model, layer



    raise ValueError("No Conv2D layer found for Grad-CAM.")





def generate_gradcam(image_batch, class_index):

    """

    Best-effort Grad-CAM for a model with a nested CNN backbone.

    If the architecture is incompatible, classification can still work.

    """

    if model is None:

        raise RuntimeError("Model is not loaded.")



    nested_candidates = [

        layer for layer in model.layers

        if isinstance(layer, tf.keras.Model)

    ]



    backbone = None

    conv_layer = None



    for candidate in reversed(nested_candidates):

        try:

            _, found_layer = find_last_conv_layer(candidate)

            backbone = candidate

            conv_layer = found_layer

            break

        except ValueError:

            continue



    if backbone is None:

        raise ValueError(

            "Grad-CAM requires a nested CNN backbone. "

            "Prediction is still available without the heatmap."

        )



    feature_model = tf.keras.Model(

        inputs=backbone.inputs,

        outputs=[conv_layer.output, backbone.output],

    )



    backbone_index = model.layers.index(backbone)

    head_layers = model.layers[backbone_index + 1:]



    with tf.GradientTape() as tape:

        conv_features, x = feature_model(

            image_batch,

            training=False,

        )



        for layer in head_layers:

            if isinstance(layer, tf.keras.layers.InputLayer):

                continue

            x = layer(x, training=False)



        if isinstance(x, (list, tuple)):

            x = x[0]



        class_score = x[:, class_index]



    gradients = tape.gradient(class_score, conv_features)



    if gradients is None:

        raise RuntimeError("Grad-CAM gradients are unavailable.")



    weights = tf.reduce_mean(

        gradients,

        axis=(1, 2),

        keepdims=True,

    )



    heatmap = tf.reduce_sum(

        weights * conv_features,

        axis=-1,

    )[0]



    heatmap = tf.maximum(heatmap, 0)

    heatmap = tf.math.divide_no_nan(

        heatmap,

        tf.reduce_max(heatmap),

    ).numpy()



    if not np.all(np.isfinite(heatmap)) or np.max(heatmap) <= 0:

        raise RuntimeError("Grad-CAM heatmap is empty.")



    heatmap_image = Image.fromarray(

        np.uint8(np.clip(heatmap, 0, 1) * 255)

    ).resize(IMAGE_SIZE, Image.Resampling.BILINEAR)



    heatmap_array = (

        np.asarray(heatmap_image, dtype=np.float32) / 255.0

    )



    colored = colormaps["jet"](heatmap_array)[..., :3]



    colored_image = Image.fromarray(

        np.uint8(np.clip(colored, 0, 1) * 255),

        mode="RGB",

    )



    original = Image.fromarray(

        image_batch[0].astype(np.uint8),

        mode="RGB",

    )



    return Image.blend(original, colored_image, alpha=0.40)





# =========================================================

# AI REPORT

# =========================================================



def generate_fallback_report(

    predicted_class,

    scores,

    gradcam_available=False,

):

    ordered = sorted(

        scores,

        key=lambda item: item["value"],

        reverse=True,

    )



    top_score = ordered[0]["value"] if ordered else 0.0

    second_score = ordered[1]["value"] if len(ordered) > 1 else 0.0

    gap = top_score - second_score



    next_step = {

        "Severe": (

            "Please arrange prompt assessment by an eye-care professional."

        ),

        "Proliferative DR": (

            "Please seek prompt assessment by an eye-care professional."

        ),

    }.get(

        predicted_class,

        "Have the retinal image assessed by a qualified eye-care professional.",

    )



    if gradcam_available:

        gradcam_text = (

            "A Grad-CAM overlay was generated, but image-based interpretation "

            "was unavailable. The heatmap shows regions influencing the model; "

            "it does not confirm lesions."

        )

    else:

        gradcam_text = (

            "Grad-CAM was unavailable, so no heatmap interpretation can be provided."

        )



    uncertainty = (

        f"The highest model score is {top_score:.2f}% and the next-highest "

        f"is {second_score:.2f}% (difference {gap:.2f} percentage points). "

        "These are model scores, not calibrated clinical confidence, and may be wrong."

    )



    return (

        f"Prediction\n{predicted_class}\n\n"

        "Why the model may have predicted this\n"

        f"The model assigned its highest score to {predicted_class}. Scores alone "

        "do not show which retinal features caused the result or prove that a "

        "specific lesion is present.\n\n"

        f"Grad-CAM interpretation\n{gradcam_text}\n\n"

        f"Uncertainty\n{uncertainty}\n\n"

        f"Next step\n{next_step}\n\n"

        "Disclaimer\nThis is an unvalidated research prototype and must not guide "

        "clinical decisions."

    )





def generate_genai_report(

    predicted_class,

    scores,

    original_image=None,

    gradcam_image=None,

    quality=None,

):

    """

    Sends the original fundus image and optional Grad-CAM overlay to a

    vision-capable Groq model. Falls back safely if the request fails.

    """

    fallback = generate_fallback_report(

        predicted_class,

        scores,

        gradcam_available=gradcam_image is not None,

    )



    if groq_client is None or original_image is None:

        return fallback



    scores_text = "\n".join(

        f"{item['name']}: {item['value']:.2f}%"

        for item in scores

    )



    quality = quality or {}

    blur = quality.get("blur", {})

    crop = quality.get("crop", {})



    system_prompt = """

You explain outputs from an experimental diabetic retinopathy classifier.

You are not a clinician and must not diagnose.



You receive the original retinal fundus image, an optional Grad-CAM overlay,

model class scores, and optional image-quality notes.



Use exactly these headings:

Prediction

Why the model may have predicted this

Grad-CAM interpretation

Uncertainty

Next step

Disclaimer



Instructions:

\- Explain the predicted class and compare the two highest model scores.

\- Scores are model outputs, not calibrated clinical confidence.

\- Describe only image features you can reasonably observe.

\- Never invent hemorrhages, exudates, microaneurysms, abnormal vessels,

  neovascularization, or other lesions.

\- If you cannot confidently identify a visible feature, say so.

\- The first image is the original fundus image.

\- The second image, if supplied, is the Grad-CAM overlay.

\- Describe the approximate highlighted region only if it is visually apparent.

\- Compare that region with the original image and explain whether a visible

  feature seems to correspond to it, or whether the reason remains unclear.

\- Grad-CAM shows regions influencing the model; it does not prove a lesion

  or establish causality.

\- Red or high-intensity areas do not automatically mean disease.

\- Do not claim that Grad-CAM proves why a class was predicted.

\- If no heatmap is supplied, explicitly say interpretation is unavailable.

\- Mention image quality only when supported by the provided notes or clearly visible.

\- Do not prescribe treatment or claim clinical certainty.

\- Recommend assessment by a qualified eye-care professional.

\- State that this is an unvalidated research prototype and must not guide

  clinical decisions.

\- Use simple, concise English.

"""



    user_content = [

        {

            "type": "text",

            "text": (

                f"Predicted class: {predicted_class}\n\n"

                f"Model scores:\n{scores_text}\n\n"

                f"Image quality notes:\n"

                f"Blur: {blur.get('message', 'Not provided')}\n"

                f"Framing: {crop.get('message', 'Not provided')}\n\n"

                "The first image is the original retinal fundus image. "

                "The second image, if supplied, is the Grad-CAM overlay "

                "for the predicted class. Compare them carefully."

            ),

        }

    ]



    try:

        original_b64 = image_to_base64(original_image)



        user_content.append({

            "type": "image_url",

            "image_url": {

                "url": f"data:image/png;base64,{original_b64}"

            },

        })



        if gradcam_image is not None:

            gradcam_b64 = image_to_base64(gradcam_image)



            user_content.append({

                "type": "image_url",

                "image_url": {

                    "url": f"data:image/png;base64,{gradcam_b64}"

                },

            })

        else:

            user_content[0]["text"] += (

                "\nGrad-CAM overlay was unavailable. Do not invent its interpretation."

            )



        response = groq_client.chat.completions.create(

            model=GROQ_MODEL,

            messages=[

                {"role": "system", "content": system_prompt},

                {"role": "user", "content": user_content},

            ],

            temperature=0.2,

            max_tokens=900,

        )



        report = response.choices[0].message.content

        return report.strip() if report and report.strip() else fallback



    except Exception:

        logger.exception("Vision-based Groq report failed; using fallback")

        return fallback





# =========================================================

# MAIN PAGE AND PREDICTION

# =========================================================



@app.route("/", methods=["GET", "POST"])

def index():

    result = None

    error = None



    if request.method == "POST":

        if model is None:

            error = (

                "The model could not be loaded. "

                "Check the model path and server terminal."

            )

        else:

            uploaded_file = request.files.get("fundus_image")



            if uploaded_file is None or not uploaded_file.filename:

                error = "Please choose an image first."



            elif not allowed_file(uploaded_file.filename):

                error = "Only JPG, JPEG, and PNG images are supported."



            else:

                try:

                    image = Image.open(uploaded_file.stream)

                    image = ImageOps.exif_transpose(image).convert("RGB")



                    # Reject non-retinal images and blurry images before DR prediction.
                    retinal_ok, retinal_score, retinal_message = validate_retinal_image(image)
                    if not retinal_ok:
                        error = retinal_message
                    else:
                        blur_result = detect_blur(image)
                        crop_result = detect_crop(image)
                        if blur_result["is_blurry"]:
                            error = (
                                "This retinal image is too blurry to analyze reliably. "
                                "Please upload a sharper retinal photograph."
                            )
                        else:
                            original_image = image.copy()
                            _, image_batch = preprocess_image(image)
                            raw_output = model(image_batch, training=False)



                            if isinstance(raw_output, (list, tuple)):

                                if len(raw_output) != 1:

                                    raise ValueError("Expected a single model output.")

                                raw_output = raw_output[0]



                            raw_values = (

                                raw_output.numpy()

                                if hasattr(raw_output, "numpy")

                                else raw_output

                            )



                            probabilities = get_probabilities(raw_values)



                            predicted_index = int(np.argmax(probabilities))

                            predicted_class = CLASS_NAMES[predicted_index]



                            scores = [

                                {

                                    "name": name,

                                    "value": float(probability * 100.0),

                                }

                                for name, probability in zip(

                                    CLASS_NAMES,

                                    probabilities,

                                )

                            ]



                            confidence = float(

                                probabilities[predicted_index] * 100.0

                            )



                            # Generate Grad-CAM; prediction remains available if it fails.
                            gradcam_base64 = None

                            gradcam_error = None

                            gradcam_image = None



                            try:

                                gradcam_image = generate_gradcam(

                                    image_batch,

                                    predicted_index,

                                )

                                gradcam_base64 = image_to_base64(gradcam_image)



                            except Exception as exc:

                                gradcam_error = str(exc)

                                logger.exception("Grad-CAM generation failed")



                            quality = {

                                "blur": blur_result,

                                "crop": crop_result,

                            }



                            # Send both images to Groq for a grounded explanation.
                            report = generate_genai_report(

                                predicted_class,

                                scores,

                                original_image=original_image,

                                gradcam_image=gradcam_image,

                                quality=quality,

                            )



                            result = {

                                "filename": Path(uploaded_file.filename).name,

                                "predicted_class": predicted_class,

                                "confidence": confidence,

                                "scores": scores,

                                "original_image": image_to_base64(original_image),

                                "gradcam": gradcam_base64,

                                "gradcam_error": gradcam_error,

                                "report": report,

                                "quality": quality,

                                "verification": {

                                    "verified": None,

                                    "reason": (

                                        "Reference matching is not used as a "

                                        "hard rejection check in this version."

                                    ),

                                    "best_match": None,

                                    "good_matches": 0,

                                },

                            }



                except Exception as exc:

                    logger.exception("Prediction failed")

                    error = (

                        "Could not process this image. "

                        f"Technical details: {exc}"

                    )



    return render_template(

        "index.html",

        result=result,

        error=error,

        model_ready=model is not None,

        model_error=model_error,

    )





# =========================================================

# ENGLISH / HINDI PDF REPORT

# =========================================================



def build_pdf_report(

    language,

    predicted_class,

    confidence,

    scores,

    original_image_b64,

):

    hindi = language == "hi"



    labels = {

        "title": (

            "डायबिटिक रेटिनोपैथी स्क्रीनिंग रिपोर्ट"

            if hindi else "Diabetic Retinopathy Screening Report"

        ),

        "date": "रिपोर्ट की तारीख" if hindi else "Report date",

        "image": "रेटिना की छवि" if hindi else "Retinal image",

        "prediction": "मॉडल का अनुमान" if hindi else "Model prediction",

        "confidence": "मॉडल स्कोर" if hindi else "Model score",

        "scores": "क्लास के स्कोर" if hindi else "Class scores",

        "details": "रिपोर्ट विवरण" if hindi else "Report details",

        "meaning": "अर्थ" if hindi else "Meaning",

        "uncertainty": "अनिश्चितता" if hindi else "Uncertainty",

        "gradcam": "Grad-CAM स्पष्टीकरण" if hindi else "Grad-CAM explanation",

        "next": "अगला कदम" if hindi else "Next step",

        "disclaimer": "अस्वीकरण" if hindi else "Disclaimer",

        "class": "क्लास" if hindi else "Class",

        "score": "स्कोर (%)" if hindi else "Score (%)",

    }



    shown_class = (

        CLASS_HI.get(predicted_class, predicted_class)

        if hindi else predicted_class

    )



    if hindi:

        paragraphs = {

            "meaning": (

                f"मॉडल का अनुमान '{shown_class}' श्रेणी की ओर है। "

                "यह चिकित्सकीय पुष्टि नहीं है।"

            ),

            "uncertainty": (

                "स्कोर मॉडल के आउटपुट हैं, चिकित्सकीय रूप से कैलिब्रेटेड "

                "विश्वास-स्तर नहीं। परिणाम गलत हो सकता है। अधिक स्कोर "

                "बीमारी की पुष्टि नहीं करता।"

            ),

            "gradcam": (

                "Grad-CAM उन छवि क्षेत्रों को दिखाता है जिन्होंने मॉडल के "

                "आउटपुट को प्रभावित किया। यह घाव या बीमारी की पुष्टि नहीं करता।"

            ),

            "next": (

                "रेटिना की छवि का मूल्यांकन योग्य नेत्र विशेषज्ञ से करवाएँ। "

                "इस ऐप के आधार पर उपचार का निर्णय न लें।"

            ),

            "disclaimer": (

                "यह एक शोध प्रोटोटाइप है, चिकित्सकीय रूप से मान्य उपकरण नहीं। "

                "यह निदान या उपचार का विकल्प नहीं है और इसे चिकित्सकीय "

                "निर्णय लेने के लिए उपयोग नहीं किया जाना चाहिए।"

            ),

        }

    else:

        paragraphs = {

            "meaning": (

                f"The model output favors the {predicted_class} category. "

                "This is not a confirmed clinical finding."

            ),

            "uncertainty": (

                "Scores are model outputs, not calibrated clinical confidence. "

                "They may be wrong; a high score does not confirm a diagnosis."

            ),

            "gradcam": (

                "Grad-CAM highlights image regions that influenced the model. "

                "It does not confirm lesions or disease."

            ),

            "next": (

                "Have the retinal image assessed by a qualified eye-care "

                "professional. Do not make treatment decisions based on this app."

            ),

            "disclaimer": (

                "This is an unvalidated research prototype, not a clinically "

                "validated tool. It must not guide clinical decisions."

            ),

        }



    font_name = "Helvetica"



    if hindi:

        if not FONT_PATH.exists():

            raise FileNotFoundError(

                f"Hindi font not found at {FONT_PATH}. "

                "Use a static NotoSansDevanagari-Regular.ttf if the variable "

                "font fails."

            )



        if "NotoHindi" not in pdfmetrics.getRegisteredFontNames():

            pdfmetrics.registerFont(TTFont("NotoHindi", str(FONT_PATH)))



        font_name = "NotoHindi"



    buffer = io.BytesIO()



    doc = SimpleDocTemplate(

        buffer,

        pagesize=A4,

        rightMargin=18 * mm,

        leftMargin=18 * mm,

        topMargin=16 * mm,

        bottomMargin=16 * mm,

    )



    styles = getSampleStyleSheet()



    title_style = ParagraphStyle(

        "ConsistentDRTitle",

        parent=styles["Title"],

        fontName=font_name,

        fontSize=16,

        leading=22,

        alignment=TA_CENTER,

        textColor=colors.HexColor("#17365D"),

        spaceAfter=5 * mm,

    )



    body_style = ParagraphStyle(

        "ConsistentDRBody",

        parent=styles["BodyText"],

        fontName=font_name,

        fontSize=9,

        leading=13,

        wordWrap="CJK",

        spaceAfter=3,

    )



    heading_style = ParagraphStyle(

        "ConsistentDRHeading",

        parent=styles["Heading2"],

        fontName=font_name,

        fontSize=11,

        leading=15,

        spaceBefore=7,

        spaceAfter=4,

        keepWithNext=True,

    )



    story = [

        Paragraph(escape(labels["title"]), title_style),

        Paragraph(

            escape(

                f"{labels['date']}: "

                f"{datetime.now().strftime('%Y-%m-%d %H:%M')}"

            ),

            body_style,

        ),

        Paragraph(escape(labels["image"]), heading_style),

    ]



    # Embed the original retinal image instead of printing its filename.
    try:

        raw_image = base64.b64decode(

            original_image_b64,

            validate=True,

        )



        pil_image = Image.open(io.BytesIO(raw_image)).convert("RGB")



        image_buffer = io.BytesIO()

        pil_image.save(image_buffer, format="PNG")

        image_buffer.seek(0)



        story.append(

            PDFImage(

                image_buffer,

                width=75 * mm,

                height=75 * mm,

                kind="proportional",

            )

        )



    except Exception:

        logger.exception("Could not embed retinal image in PDF")

        story.append(

            Paragraph(

                escape(

                    "छवि रिपोर्ट में शामिल नहीं हो सकी।"

                    if hindi

                    else "The retinal image could not be embedded in this report."

                ),

                body_style,

            )

        )



    story.extend([

        Paragraph(escape(labels["prediction"]), heading_style),

        Paragraph(escape(shown_class), body_style),

        Paragraph(

            escape(f"{labels['confidence']}: {float(confidence):.2f}%"),

            body_style,

        ),

        Paragraph(escape(labels["scores"]), heading_style),

    ])



    table_data = [[labels["class"], labels["score"]]]



    for item in scores:

        if not isinstance(item, dict):

            continue



        name = str(item.get("name", ""))



        if hindi:

            name = CLASS_HI.get(name, name)



        try:

            value = float(item.get("value", 0))

        except (TypeError, ValueError):

            value = 0.0



        table_data.append([escape(name), f"{value:.2f}%"])



    score_table = Table(

        table_data,

        colWidths=[100 * mm, 45 * mm],

        repeatRows=1,

    )



    score_table.setStyle(TableStyle([

        ("FONTNAME", (0, 0), (-1, -1), font_name),

        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF5")),

        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17365D")),

        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B8C4D1")),

        ("VALIGN", (0, 0), (-1, -1), "TOP"),

        ("PADDING", (0, 0), (-1, -1), 6),

    ]))



    story.extend([

        score_table,

        Paragraph(escape(labels["details"]), heading_style),

    ])



    # Keep section order consistent between English and Hindi.
    for key in (

        "meaning",

        "uncertainty",

        "gradcam",

        "next",

        "disclaimer",

    ):

        story.append(Paragraph(escape(labels[key]), heading_style))

        story.append(Paragraph(escape(paragraphs[key]), body_style))



    doc.build(story)

    buffer.seek(0)

    return buffer





# =========================================================

# DOWNLOAD PDF

# =========================================================



@app.route("/download-report", methods=["POST"])

def download_report():

    language = request.form.get("language", "en").strip()



    if language not in ("en", "hi"):

        return "Invalid language.", 400



    predicted_class = request.form.get(

        "predicted_class", "Unknown"

    ).strip()



    original_image_b64 = request.form.get(

        "original_image", ""

    ).strip()



    try:

        confidence_text = request.form.get(

            "confidence", "0"

        ).strip().replace("%", "")



        confidence = float(confidence_text)



        if not np.isfinite(confidence):

            raise ValueError("Confidence is not finite.")



    except (TypeError, ValueError):

        logger.exception(

            "Invalid confidence field: %r",

            request.form.get("confidence"),

        )

        return "Invalid confidence data. Please analyze the image again.", 400



    scores_text = request.form.get("scores", "").strip()



    try:

        scores = json.loads(scores_text)



        if not isinstance(scores, list):

            raise ValueError("Scores must be a JSON list.")



        clean_scores = []



        for item in scores:

            if not isinstance(item, dict):

                continue



            name = str(item.get("name", "")).strip()

            value = float(item.get("value", 0))



            if not np.isfinite(value):

                continue



            clean_scores.append({

                "name": name,

                "value": value,

            })



        scores = clean_scores



        if not scores:

            raise ValueError("No valid class scores were received.")



    except (ValueError, TypeError, json.JSONDecodeError):

        logger.exception(

            "Invalid scores field received by PDF route: %r",

            scores_text[:500],

        )

        return (

            "Invalid report scores. Please analyze the image again "

            "and ensure the scores hidden input uses "

            "{{ result.scores | tojson | forceescape }}.",

            400,

        )



    if not original_image_b64:

        logger.error("Original image was missing from the PDF form.")

        return (

            "The retinal image is missing. Please analyze the image again "

            "and check the original_image hidden input.",

            400,

        )



    try:

        image_bytes = base64.b64decode(

            original_image_b64,

            validate=True,

        )



        with Image.open(io.BytesIO(image_bytes)) as check_image:

            check_image.verify()



        pdf_buffer = build_pdf_report(

            language=language,

            predicted_class=predicted_class,

            confidence=confidence,

            scores=scores,

            original_image_b64=original_image_b64,

        )



        return send_file(

            pdf_buffer,

            mimetype="application/pdf",

            as_attachment=True,

            download_name=(

                "retinopathy_report_hi.pdf"

                if language == "hi"

                else "retinopathy_report_en.pdf"

            ),

        )



    except Exception:

        logger.exception("PDF report generation failed")

        return (

            "Could not generate the PDF. Check the terminal for "

            "the detailed error.",

            500,

        )





# =========================================================

# ERROR HANDLER

# =========================================================



@app.errorhandler(413)

def file_too_large(_error):

    return render_template(

        "index.html",

        result=None,

        error=f"Image is too large. Maximum size is {MAX_UPLOAD_MB} MB.",

        model_ready=model is not None,

        model_error=model_error,

    ), 413





# =========================================================

# RUN APP

# =========================================================



if __name__ == "__main__":

    app.run(

        host="127.0.0.1",

        port=int(os.getenv("PORT", "5000")),

        debug=False,

    )