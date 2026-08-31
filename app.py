# ============================================================
# SAHOOLAT AI
# Patient-Friendly Medical Information Literacy Assistant
# Founder: Mian Ali Shan s/o Saadat Ali
# ============================================================

import os
import re
import json
import base64
from io import BytesIO
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageStat
from pypdf import PdfReader
from groq import Groq

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="Sahoolat AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# MODELS
# ============================================================

TEXT_MODEL = os.getenv(
    "SAHOOLAT_TEXT_MODEL",
    "openai/gpt-oss-120b",
)

VISION_MODEL = os.getenv(
    "SAHOOLAT_VISION_MODEL",
    "qwen/qwen3.6-27b",
)

# ============================================================
# LANGUAGE
# ============================================================

LANGUAGE_INSTRUCTIONS = {
    "English": """
Use simple, clear professional English.
Explain medical information so an ordinary patient can understand it.
Keep important medical terminology when necessary, but explain it.
""",

    "اردو": """
Use simple Pakistani Urdu.
Keep important medical test names and clinical terms in English
parentheses when useful.
Avoid difficult literary Urdu.
""",

    "Roman Urdu": """
Use simple Roman Urdu that Pakistani users can easily understand.
Keep laboratory test names and important medical terms in English.
""",
}

# ============================================================
# REPORT CATEGORIES
# ============================================================

REPORT_CATEGORIES = [
    "Blood Test",
    "Lipid Profile",
    "Diabetes / Glucose",
    "Liver Function",
    "Kidney Function",
    "Thyroid",
    "CBC / Hematology",
    "Urine Test",
    "Imaging / Scan",
    "Prescription / Doctor Instructions",
    "General Medical Report",
    "Other",
]

# ============================================================
# EMERGENCY TERMS
# ============================================================

EMERGENCY_TERMS = [
    "severe chest pain",
    "chest pain",
    "difficulty breathing",
    "severe difficulty breathing",
    "severe shortness of breath",
    "unconscious",
    "unconsciousness",
    "seizure",
    "stroke symptoms",
    "face drooping",
    "arm weakness",
    "speech difficulty",
    "speech trouble",
    "severe bleeding",
    "uncontrolled bleeding",
    "suicidal",
    "suicide attempt",
    "self harm",
    "خودکشی",
    "سینے میں شدید درد",
    "سینے میں درد",
    "سانس لینے میں شدید دشواری",
    "سانس لینے میں مشکل",
    "بے ہوش",
    "دورہ",
    "فالج کی علامات",
    "شدید خون بہنا",
]

NEGATION_TERMS = [
    "no ",
    "not ",
    "denies ",
    "without ",
    "negative for ",
    "denied ",
    "no history of ",
    "does not have ",
    "doesn't have ",
    "نہیں",
    "نہ ہونے",
]

# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are Sahoolat AI.

Sahoolat AI is a health-information literacy assistant for
patients and families in Pakistan.

The goal is to make medical information easier to understand.

You are NOT a doctor.

============================================================
STRICT MEDICAL SAFETY
============================================================

1. Never diagnose a disease.
2. Never prescribe medicine.
3. Never provide medicine dosage instructions.
4. Never tell the patient to start, stop, or change medication.
5. Never invent a medical value.
6. Never invent a unit.
7. Never invent a reference range.
8. Never modify a number from the source.
9. Never guess unreadable information.
10. If information cannot be read, use "Cannot determine".
11. If the report does not provide a reference range,
    use "Not provided".
12. If there is no reference range, do not classify the
    laboratory value as normal or abnormal.
13. An abnormal laboratory value alone is NOT automatically
    an emergency.
14. Emergency classification should be based on explicit
    current emergency symptoms or clearly urgent information.
15. Do not diagnose based on laboratory results.
16. Do not provide treatment plans.
17. Encourage discussion with a qualified healthcare professional.
18. Only use information provided by the user or visible in
    the uploaded document.

============================================================
EXTRACTION
============================================================

Extract only clearly visible information.

For every laboratory value provide:

- test
- result
- unit
- reference_range
- status
- confidence
- evidence

Allowed status:

"Within provided range"
"Below provided range"
"Above provided range"
"Cannot determine"

Allowed confidence:

"High"
"Medium"
"Low"
"Cannot determine"

Evidence must be a short description of what is visible.

Do NOT invent evidence.

============================================================
REPORT CATEGORY
============================================================

Choose one of the supplied report categories.

============================================================
ATTENTION LEVEL
============================================================

GREEN:
General educational information.

YELLOW:
Information that should be discussed with a healthcare professional.

RED:
Possible emergency based on explicit current emergency symptoms
or clearly urgent information.

Do NOT make a result RED merely because a laboratory number
is high or low.

============================================================
IMPORTANT TERMS
============================================================

Explain important medical terms in simple patient-friendly language.

============================================================
DOCTOR QUESTIONS
============================================================

Generate useful questions based ONLY on the supplied report.

Do not turn questions into diagnoses or treatment instructions.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Use exactly:

{
  "report_category": "General Medical Report",
  "summary": "short patient-friendly summary",
  "values": [
    {
      "test": "test name",
      "result": "result",
      "unit": "unit",
      "reference_range": "reference range",
      "status": "Cannot determine",
      "confidence": "High",
      "evidence": "short visible evidence"
    }
  ],
  "attention_level": "GREEN",
  "attention_reason": "short explanation",
  "important_terms": [
    {
      "term": "medical term",
      "meaning": "simple explanation"
    }
  ],
  "doctor_questions": [
    "question 1",
    "question 2",
    "question 3"
  ],
  "next_steps": [
    "safe educational next step"
  ],
  "safety_note": "short safety message"
}
"""

# ============================================================
# GROQ CLIENT
# ============================================================

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key:
        return None

    return Groq(api_key=api_key)


# ============================================================
# JSON HELPERS
# ============================================================

def clean_json(text):
    if not text:
        return ""

    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]

    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


def parse_json(text):
    try:
        return json.loads(clean_json(text))
    except Exception:
        return None


# ============================================================
# PRIVACY DETECTION
# ============================================================

def detect_personal_information(text):
    if not text:
        return []

    findings = []

    # Pakistan CNIC
    if re.search(r"\b\d{5}-\d{7}-\d\b", text):
        findings.append("Possible CNIC number")

    # Pakistan phone
    if re.search(r"(?:\+92|0)3\d{9}\b", text):
        findings.append("Possible phone number")

    # Email
    if re.search(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        text,
    ):
        findings.append("Possible email address")

    # Address indicators
    address_words = [
        "address:",
        "home address:",
        "street:",
        "house no:",
        "پتہ:",
    ]

    lower = text.lower()

    for word in address_words:
        if word in lower:
            findings.append("Possible address")

    # Patient ID
    if re.search(
        r"(patient id|patient no|mrn|medical record number)\s*[:#-]?\s*\w+",
        lower,
    ):
        findings.append("Possible patient identifier")

    return list(dict.fromkeys(findings))


# ============================================================
# PRIVACY REDACTION
# ============================================================

def redact_personal_information(text):
    if not text:
        return text

    # CNIC
    text = re.sub(
        r"\b\d{5}-\d{7}-\d\b",
        "[REDACTED-CNIC]",
        text,
    )

    # Pakistan phone
    text = re.sub(
        r"(?:\+92|0)3\d{9}\b",
        "[REDACTED-PHONE]",
        text,
    )

    # Email
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "[REDACTED-EMAIL]",
        text,
    )

    return text


# ============================================================
# EMERGENCY SCREENING
# ============================================================

def contains_emergency_terms(text):
    if not text:
        return False

    lower = text.lower()

    for term in EMERGENCY_TERMS:
        term_lower = term.lower()

        if term_lower not in lower:
            continue

        index = lower.find(term_lower)

        start = max(0, index - 50)
        nearby = lower[start:index]

        if any(
            negation.lower() in nearby
            for negation in NEGATION_TERMS
        ):
            continue

        return True

    return False


# ============================================================
# IMAGE QUALITY
# ============================================================

def assess_image_quality(image):
    width, height = image.size

    problems = []
    score = 100

    if width < 900 or height < 600:
        problems.append("Image resolution is low.")
        score -= 25

    try:
        gray = image.convert("L")
        stat = ImageStat.Stat(gray)
        brightness = stat.mean[0]

        if brightness < 45:
            problems.append("Image appears very dark.")
            score -= 25

        elif brightness > 245:
            problems.append("Image appears overexposed.")
            score -= 20

    except Exception:
        pass

    if score >= 80:
        quality = "Good"

    elif score >= 55:
        quality = "Fair"

    else:
        quality = "Poor"

    return {
        "quality": quality,
        "score": max(0, score),
        "problems": problems,
        "width": width,
        "height": height,
    }


# ============================================================
# IMAGE BASE64
# ============================================================

def image_to_base64(image):
    image = image.convert("RGB")

    max_dimension = 2000

    width, height = image.size

    if max(width, height) > max_dimension:

        scale = max_dimension / max(width, height)

        image = image.resize(
            (
                int(width * scale),
                int(height * scale),
            )
        )

    buffer = BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=90,
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# ============================================================
# VALIDATION
# ============================================================

def validate_result(data):

    if not isinstance(data, dict):
        return demo_report("English")

    valid_statuses = {
        "Within provided range",
        "Below provided range",
        "Above provided range",
        "Cannot determine",
    }

    valid_confidence = {
        "High",
        "Medium",
        "Low",
        "Cannot determine",
    }

    # -------------------------
    # Values
    # -------------------------

    values = data.get("values", [])

    if not isinstance(values, list):
        values = []

    cleaned_values = []
    seen = set()

    for item in values:

        if not isinstance(item, dict):
            continue

        test = str(
            item.get("test", "Unknown")
        ).strip()

        result = str(
            item.get("result", "Cannot determine")
        ).strip()

        unit = str(
            item.get("unit", "")
        ).strip()

        reference = str(
            item.get("reference_range", "Not provided")
        ).strip()

        status = str(
            item.get("status", "Cannot determine")
        ).strip()

        confidence = str(
            item.get("confidence", "Cannot determine")
        ).strip()

        evidence = str(
            item.get("evidence", "Cannot determine")
        ).strip()

        if status not in valid_statuses:
            status = "Cannot determine"

        if confidence not in valid_confidence:
            confidence = "Cannot determine"

        if not reference or reference.lower() in {
            "not provided",
            "unknown",
            "cannot determine",
        }:

            reference = "Not provided"
            status = "Cannot determine"

        key = (
            test.lower(),
            result.lower(),
            unit.lower(),
        )

        if key in seen:
            continue

        seen.add(key)

        cleaned_values.append(
            {
                "test": test or "Unknown",
                "result": result or "Cannot determine",
                "unit": unit,
                "reference_range": reference,
                "status": status,
                "confidence": confidence,
                "evidence": evidence or "Cannot determine",
            }
        )

    data["values"] = cleaned_values

    # -------------------------
    # Category
    # -------------------------

    category = str(
        data.get(
            "report_category",
            "General Medical Report",
        )
    )

    if category not in REPORT_CATEGORIES:
        category = "General Medical Report"

    data["report_category"] = category

    # -------------------------
    # Attention
    # -------------------------

    attention = str(
        data.get(
            "attention_level",
            "GREEN",
        )
    ).upper()

    if attention not in {
        "GREEN",
        "YELLOW",
        "RED",
    }:
        attention = "GREEN"

    data["attention_level"] = attention

    # -------------------------
    # Summary
    # -------------------------

    data["summary"] = str(
        data.get(
            "summary",
            "No summary available.",
        )
    )

    data["attention_reason"] = str(
        data.get(
            "attention_reason",
            "No additional explanation available.",
        )
    )

    # -------------------------
    # Terms
    # -------------------------

    terms = data.get(
        "important_terms",
        [],
    )

    if not isinstance(terms, list):
        terms = []

    cleaned_terms = []

    for term in terms:

        if isinstance(term, dict):

            cleaned_terms.append(
                {
                    "term": str(
                        term.get(
                            "term",
                            "Medical term",
                        )
                    ),

                    "meaning": str(
                        term.get(
                            "meaning",
                            "",
                        )
                    ),
                }
            )

    data["important_terms"] = cleaned_terms

    # -------------------------
    # Questions
    # -------------------------

    questions = data.get(
        "doctor_questions",
        [],
    )

    if not isinstance(questions, list):
        questions = []

    data["doctor_questions"] = [
        str(q)
        for q in questions
        if str(q).strip()
    ]

    # -------------------------
    # Next steps
    # -------------------------

    next_steps = data.get(
        "next_steps",
        [],
    )

    if not isinstance(next_steps, list):
        next_steps = []

    data["next_steps"] = [
        str(x)
        for x in next_steps
        if str(x).strip()
    ]

    # -------------------------
    # Safety note
    # -------------------------

    data["safety_note"] = str(
        data.get(
            "safety_note",
            (
                "This information is educational and does not "
                "replace professional medical advice."
            ),
        )
    )

    return data


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(uploaded_file):

    reader = PdfReader(
        BytesIO(
            uploaded_file.getvalue()
        )
    )

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):

        try:
            text = page.extract_text()

        except Exception:
            text = None

        if text:
            pages.append(
                f"[Page {page_number}]\n{text}"
            )

    return "\n\n".join(pages).strip()


# ============================================================
# API ERROR HANDLING
# ============================================================

def friendly_api_error(error):

    message = str(error)

    if "429" in message or "RESOURCE_EXHAUSTED" in message:

        return (
            "Groq API quota or rate limit was reached. "
            "Please wait and try again."
        )

    if "401" in message or "authentication" in message.lower():

        return (
            "Groq API key is missing or invalid. "
            "Check GROQ_API_KEY in your .env file."
        )

    if "403" in message:

        return (
            "Groq API access was denied. "
            "Check your API key and account permissions."
        )

    if "404" in message or "model_not_found" in message:

        return (
            "The configured Groq model was not found or is "
            "not available. Check the model name."
        )

    return (
        "The AI service returned an error. "
        "Check your API configuration and try again."
    )


# ============================================================
# TEXT ANALYSIS
# ============================================================

def analyze_text(text, language):

    client = get_groq_client()

    if client is None:

        raise RuntimeError(
            "GROQ_API_KEY is not configured."
        )

    privacy_findings = detect_personal_information(text)

    safe_text = redact_personal_information(text)

    prompt = f"""
Analyze the following medical information.

OUTPUT LANGUAGE:
{language}

LANGUAGE INSTRUCTIONS:
{LANGUAGE_INSTRUCTIONS[language]}

PRIVACY:
Some personal identifiers may have been automatically redacted.
Do not attempt to reconstruct them.

EXTRACTION:
Extract only information actually present.
Do not guess.

If a reference range is missing:
reference_range = "Not provided"
status = "Cannot determine"

If information is unreadable or unclear:
use "Cannot determine".

IMPORTANT:
Preserve numbers exactly.
Preserve decimal places exactly.
Preserve units exactly.
Preserve reference ranges exactly.

Generate patient-friendly explanations.

Generate doctor questions based only on the supplied information.

MEDICAL INFORMATION:
{safe_text}
"""

    response = client.chat.completions.create(
        model=TEXT_MODEL,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],

        temperature=0.1,

        max_tokens=4000,

        response_format={
            "type": "json_object"
        },
    )

    content = (
        response
        .choices[0]
        .message
        .content
    )

    data = parse_json(content)

    if data is None:
        raise RuntimeError(
            "Groq returned invalid JSON."
        )

    data = validate_result(data)

    data["_privacy_findings"] = privacy_findings

    return data


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(image, language):

    client = get_groq_client()

    if client is None:

        raise RuntimeError(
            "GROQ_API_KEY is not configured."
        )

    encoded = image_to_base64(image)

    prompt = f"""
Analyze this uploaded medical document image.

OUTPUT LANGUAGE:
{language}

LANGUAGE INSTRUCTIONS:
{LANGUAGE_INSTRUCTIONS[language]}

TASK:

Read the document carefully.

Extract clearly visible:

- test names
- results
- units
- reference ranges
- important medical terms
- doctor instructions
- report category

ACCURACY IS MORE IMPORTANT THAN COMPLETENESS.

Never guess.
Never invent.
Never change numbers.
Never change units.
Never create a reference range.

If something is unclear:
use "Cannot determine".

For every extracted laboratory value provide:

confidence:
High / Medium / Low / Cannot determine

evidence:
brief description of where the information is visible.

If the image contains a patient name or other personal identifier,
do not reproduce unnecessary identifiers in the summary.

Do not diagnose.
Do not prescribe.
Do not recommend medication changes.

Do not classify an abnormal laboratory result as RED
merely because it is abnormal.

Return ONLY valid JSON.
"""

    response = client.chat.completions.create(

        model=VISION_MODEL,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },

            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },

                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                "data:image/jpeg;base64,"
                                + encoded
                            )
                        },
                    },
                ],
            },
        ],

        temperature=0.1,

        max_tokens=4000,

        response_format={
            "type": "json_object"
        },
    )

    content = (
        response
        .choices[0]
        .message
        .content
    )

    data = parse_json(content)

    if data is None:
        raise RuntimeError(
            "Groq vision model returned invalid JSON."
        )

    return validate_result(data)


# ============================================================
# EXTRACTION VERIFICATION
# ============================================================

def verify_extraction(data):

    values = data.get(
        "values",
        [],
    )

    if not values:

        return {
            "score": 0,
            "label": "No values to verify",
            "checks": [],
        }

    checks = []

    for item in values:

        confidence = item.get(
            "confidence",
            "Cannot determine",
        )

        evidence = item.get(
            "evidence",
            "Cannot determine",
        )

        test = item.get(
            "test",
            "",
        )

        result = item.get(
            "result",
            "",
        )

        if (
            confidence == "High"
            and evidence != "Cannot determine"
            and result != "Cannot determine"
        ):

            status = "PASS"

            reason = (
                "Clear extraction evidence reported."
            )

        elif confidence == "Medium":

            status = "REVIEW"

            reason = (
                "Extraction should be checked against "
                "the original report."
            )

        else:

            status = "REVIEW"

            reason = (
                "Value may require manual verification."
            )

        checks.append(
            {
                "test": test,
                "result": result,
                "status": status,
                "reason": reason,
            }
        )

    passed = sum(
        1
        for item in checks
        if item["status"] == "PASS"
    )

    score = round(
        passed / len(checks) * 100
    )

    if score >= 80:

        label = "High extraction confidence"

    elif score >= 50:

        label = "Review recommended"

    else:

        label = (
            "Manual verification strongly recommended"
        )

    return {
        "score": score,
        "label": label,
        "checks": checks,
    }


# ============================================================
# STATISTICS
# ============================================================

def calculate_stats(values):

    within = 0
    outside = 0
    unknown = 0

    for item in values:

        status = item.get(
            "status",
            "",
        )

        if status == "Within provided range":

            within += 1

        elif status in {
            "Below provided range",
            "Above provided range",
        }:

            outside += 1

        else:

            unknown += 1

    return (
        len(values),
        within,
        outside,
        unknown,
    )


# ============================================================
# DEMO REPORT
# ============================================================

def demo_report(language):

    if language == "اردو":

        return {
            "report_category": "Lipid Profile",

            "summary":
                "یہ ایک ڈیمو رپورٹ ہے۔ Total Cholesterol اور "
                "Triglycerides فراہم کردہ reference range سے اوپر ہیں۔ "
                "یہ نتائج خود کسی بیماری کی تشخیص نہیں کرتے۔",

            "values": [
                {
                    "test": "Total Cholesterol",
                    "result": "265",
                    "unit": "mg/dL",
                    "reference_range": "< 200 mg/dL",
                    "status": "Above provided range",
                    "confidence": "High",
                    "evidence":
                        "Demo value shown in the sample report.",
                },
                {
                    "test": "Triglycerides",
                    "result": "215",
                    "unit": "mg/dL",
                    "reference_range": "< 150 mg/dL",
                    "status": "Above provided range",
                    "confidence": "High",
                    "evidence":
                        "Demo value shown in the sample report.",
                },
                {
                    "test": "HDL",
                    "result": "52",
                    "unit": "mg/dL",
                    "reference_range": ">= 40 mg/dL",
                    "status": "Within provided range",
                    "confidence": "High",
                    "evidence":
                        "Demo value shown in the sample report.",
                },
            ],

            "attention_level": "YELLOW",

            "attention_reason":
                "کچھ values فراہم کردہ range سے باہر ہیں۔ "
                "نتائج کو healthcare professional کے ساتھ discuss کریں۔",

            "important_terms": [
                {
                    "term": "Cholesterol",
                    "meaning":
                        "خون میں موجود ایک مادہ جس کے مختلف حصے جسم "
                        "کے لیے مختلف کردار رکھتے ہیں۔",
                },
                {
                    "term": "Triglycerides",
                    "meaning":
                        "خون میں موجود ایک قسم کی چربی۔",
                },
            ],

            "doctor_questions": [
                "ان نتائج کا میرے لیے کیا مطلب ہے؟",
                "کیا مجھے مزید evaluation کی ضرورت ہے؟",
                "مجھے follow-up کب کرنا چاہیے؟",
            ],

            "next_steps": [
                "اصل رپورٹ کو qualified healthcare professional "
                "کے ساتھ discuss کریں۔",
                "نتائج کو original report کے ساتھ verify کریں۔",
            ],

            "safety_note":
                "یہ تعلیمی معلومات ہیں، diagnosis یا treatment نہیں۔",
        }

    if language == "Roman Urdu":

        return {
            "report_category": "Lipid Profile",

            "summary":
                "Yeh demo report hai. Total Cholesterol aur "
                "Triglycerides provided reference range se upar hain. "
                "Yeh results khud kisi disease ki diagnosis nahi karte.",

            "values": [
                {
                    "test": "Total Cholesterol",
                    "result": "265",
                    "unit": "mg/dL",
                    "reference_range": "< 200 mg/dL",
                    "status": "Above provided range",
                    "confidence": "High",
                    "evidence":
                        "Demo value shown in the sample report.",
                },
                {
                    "test": "Triglycerides",
                    "result": "215",
                    "unit": "mg/dL",
                    "reference_range": "< 150 mg/dL",
                    "status": "Above provided range",
                    "confidence": "High",
                    "evidence":
                        "Demo value shown in the sample report.",
                },
                {
                    "test": "HDL",
                    "result": "52",
                    "unit": "mg/dL",
                    "reference_range": ">= 40 mg/dL",
                    "status": "Within provided range",
                    "confidence": "High",
                    "evidence":
                        "Demo value shown in the sample report.",
                },
            ],

            "attention_level": "YELLOW",

            "attention_reason":
                "Kuch values provided range se bahar hain. "
                "Results ko healthcare professional ke sath discuss karein.",

            "important_terms": [
                {
                    "term": "Cholesterol",
                    "meaning":
                        "Khoon mein mojood ek substance jo body mein "
                        "different roles rakhta hai.",
                },
                {
                    "term": "Triglycerides",
                    "meaning":
                        "Khoon mein mojood ek qisam ki charbi.",
                },
            ],

            "doctor_questions": [
                "In results ka mere liye kya matlab hai?",
                "Kya mujhe mazeed evaluation ki zaroorat hai?",
                "Mujhe follow-up kab karna chahiye?",
            ],

            "next_steps": [
                "Original report ko qualified healthcare "
                "professional ke sath discuss karein.",
                "Results ko original report ke sath verify karein.",
            ],

            "safety_note":
                "Yeh educational information hai, diagnosis ya treatment nahi.",
        }

    return {
        "report_category": "Lipid Profile",

        "summary":
            "This is a demonstration report. Total Cholesterol "
            "and Triglycerides are above the provided reference ranges. "
            "These results alone do not diagnose a disease.",

        "values": [
            {
                "test": "Total Cholesterol",
                "result": "265",
                "unit": "mg/dL",
                "reference_range": "< 200 mg/dL",
                "status": "Above provided range",
                "confidence": "High",
                "evidence":
                    "Demo value shown in the sample report.",
            },
            {
                "test": "Triglycerides",
                "result": "215",
                "unit": "mg/dL",
                "reference_range": "< 150 mg/dL",
                "status": "Above provided range",
                "confidence": "High",
                "evidence":
                    "Demo value shown in the sample report.",
            },
            {
                "test": "HDL",
                "result": "52",
                "unit": "mg/dL",
                "reference_range": ">= 40 mg/dL",
                "status": "Within provided range",
                "confidence": "High",
                "evidence":
                    "Demo value shown in the sample report.",
            },
        ],

        "attention_level": "YELLOW",

        "attention_reason":
            "Some values are outside the provided ranges. "
            "Discuss the results with a qualified healthcare professional.",

        "important_terms": [
            {
                "term": "Cholesterol",
                "meaning":
                    "A substance in the blood that has different "
                    "roles in the body.",
            },
            {
                "term": "Triglycerides",
                "meaning":
                    "A type of fat found in the blood.",
            },
        ],

        "doctor_questions": [
            "What do these results mean for me?",
            "Do I need further evaluation?",
            "When should I follow up?",
        ],

        "next_steps": [
            "Discuss the original report with a qualified healthcare professional.",
            "Verify extracted values against the original report.",
        ],

        "safety_note":
            "This is educational information, not a diagnosis or treatment plan.",
    }


# ============================================================
# DOWNLOAD TEXT
# ============================================================

def make_download_text(data):

    lines = [
        "SAHOOLAT AI",
        "Patient-Friendly Medical Information Summary",
        "=" * 65,
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Report category: {data.get('report_category', '')}",
        "",
        "WHAT THE REPORT SAYS",
        "-" * 65,
        data.get("summary", ""),
        "",
        "MEDICAL VALUES",
        "-" * 65,
    ]

    for item in data.get("values", []):

        lines.extend(
            [
                f"Test: {item.get('test', '')}",
                f"Result: {item.get('result', '')}",
                f"Unit: {item.get('unit', '')}",
                f"Reference: {item.get('reference_range', '')}",
                f"Status: {item.get('status', '')}",
                f"Confidence: {item.get('confidence', '')}",
                f"Evidence: {item.get('evidence', '')}",
                "",
            ]
        )

    lines.extend(
        [
            "ATTENTION LEVEL",
            "-" * 65,
            data.get("attention_level", "GREEN"),
            data.get("attention_reason", ""),
            "",
            "IMPORTANT MEDICAL TERMS",
            "-" * 65,
        ]
    )

    for term in data.get(
        "important_terms",
        [],
    ):

        lines.append(
            f"- {term.get('term', '')}: "
            f"{term.get('meaning', '')}"
        )

    lines.extend(
        [
            "",
            "SAFE NEXT STEPS",
            "-" * 65,
        ]
    )

    for step in data.get(
        "next_steps",
        [],
    ):

        lines.append(
            f"- {step}"
        )

    lines.extend(
        [
            "",
            "QUESTIONS FOR DOCTOR",
            "-" * 65,
        ]
    )

    for number, question in enumerate(
        data.get(
            "doctor_questions",
            [],
        ),
        start=1,
    ):

        lines.append(
            f"{number}. {question}"
        )

    lines.extend(
        [
            "",
            "SAFETY NOTE",
            "-" * 65,
            data.get(
                "safety_note",
                "",
            ),
            "",
            "Sahoolat AI provides educational health information only.",
            "It does not diagnose diseases or prescribe treatment.",
        ]
    )

    return "\n".join(lines)


# ============================================================
# PDF SUMMARY
# ============================================================

def create_pdf(data):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=22,
        spaceAfter=10,
    )

    heading_style = ParagraphStyle(
        "HeadingCustom",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=10,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=13,
        spaceAfter=6,
    )

    small_style = ParagraphStyle(
        "SmallCustom",
        parent=styles["BodyText"],
        fontSize=8,
        leading=11,
    )

    story = []

    story.append(
        Paragraph(
            "SAHOOLAT AI",
            title_style,
        )
    )

    story.append(
        Paragraph(
            "Patient-Friendly Medical Information Summary",
            body_style,
        )
    )

    story.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            small_style,
        )
    )

    story.append(
        Paragraph(
            f"Report Category: "
            f"{data.get('report_category', '')}",
            small_style,
        )
    )

    story.append(
        Spacer(1, 8)
    )

    story.append(
        Paragraph(
            "What the Report Says",
            heading_style,
        )
    )

    story.append(
        Paragraph(
            data.get("summary", ""),
            body_style,
        )
    )

    story.append(
        Paragraph(
            "Medical Values",
            heading_style,
        )
    )

    table_data = [
        [
            "Test",
            "Result",
            "Unit",
            "Reference",
            "Status",
        ]
    ]

    for item in data.get(
        "values",
        [],
    ):

        table_data.append(
            [
                item.get("test", ""),
                item.get("result", ""),
                item.get("unit", ""),
                item.get("reference_range", ""),
                item.get("status", ""),
            ]
        )

    if len(table_data) == 1:

        table_data.append(
            [
                "No values detected",
                "",
                "",
                "",
                "",
            ]
        )

    table = Table(
        table_data,
        repeatRows=1,
        colWidths=[
            30 * mm,
            22 * mm,
            20 * mm,
            45 * mm,
            45 * mm,
        ],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#E8F0FE"),
                ),

                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.black,
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),

                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    7,
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
            ]
        )
    )

    story.append(table)

    story.append(
        Paragraph(
            "Attention Level",
            heading_style,
        )
    )

    story.append(
        Paragraph(
            f"<b>{data.get('attention_level', 'GREEN')}</b>",
            body_style,
        )
    )

    story.append(
        Paragraph(
            data.get(
                "attention_reason",
                "",
            ),
            body_style,
        )
    )

    story.append(
        Paragraph(
            "Important Medical Terms",
            heading_style,
        )
    )

    for term in data.get(
        "important_terms",
        [],
    ):

        story.append(
            Paragraph(
                f"<b>{term.get('term', '')}</b>: "
                f"{term.get('meaning', '')}",
                body_style,
            )
        )

    story.append(
        Paragraph(
            "Safe Next Steps",
            heading_style,
        )
    )

    for step in data.get(
        "next_steps",
        [],
    ):

        story.append(
            Paragraph(
                f"- {step}",
                body_style,
            )
        )

    story.append(
        Paragraph(
            "Questions for Your Doctor",
            heading_style,
        )
    )

    for number, question in enumerate(
        data.get(
            "doctor_questions",
            [],
        ),
        start=1,
    ):

        story.append(
            Paragraph(
                f"{number}. {question}",
                body_style,
            )
        )

    story.append(
        Paragraph(
            "Safety",
            heading_style,
        )
    )

    story.append(
        Paragraph(
            data.get(
                "safety_note",
                "",
            ),
            body_style,
        )
    )

    story.append(
        Spacer(1, 10)
    )

    story.append(
        Paragraph(
            "Educational information only — not a diagnosis, "
            "prescription, or replacement for professional medical care.",
            small_style,
        )
    )

    doc.build(story)

    return buffer.getvalue()


# ============================================================
# VOICE SUPPORT
# ============================================================

def voice_output(text, language):

    """
    Uses the browser's built-in speech synthesis.

    This avoids adding another paid API and works directly
    inside the user's browser.

    Urdu quality depends on the Urdu voice installed/available
    in the browser/operating system.
    """

    if not text:
        return

    safe_text = json.dumps(text)

    if language == "اردو":
        lang_code = "ur-PK"

    elif language == "Roman Urdu":
        lang_code = "ur-PK"

    else:
        lang_code = "en-US"

    html = f"""
    <script>
    const text = {safe_text};
    const lang = "{lang_code}";

    function speakSahoolat() {{
        if (!window.speechSynthesis) {{
            alert("Speech synthesis is not supported by this browser.");
            return;
        }}

        window.speechSynthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = lang;
        utterance.rate = 0.85;
        utterance.pitch = 1.0;
        utterance.volume = 1.0;

        const voices = window.speechSynthesis.getVoices();

        const preferred = voices.find(
            v => v.lang.toLowerCase().startsWith(lang.toLowerCase())
        );

        if (preferred) {{
            utterance.voice = preferred;
        }}

        window.speechSynthesis.speak(utterance);
    }}

    speakSahoolat();
    </script>
    """

    st.components.v1.html(
        html,
        height=0,
    )


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

.main-title {
    font-size: 46px;
    font-weight: 800;
    margin-bottom: 0;
}

.subtitle {
    font-size: 19px;
    opacity: 0.7;
    margin-bottom: 20px;
}

.feature-card {
    padding: 20px;
    border-radius: 16px;
    border: 1px solid rgba(128,128,128,0.25);
    min-height: 120px;
}

.feature-title {
    font-size: 19px;
    font-weight: 700;
}

.feature-description {
    opacity: 0.72;
    margin-top: 8px;
}

.safety-box {
    padding: 15px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.25);
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "history" not in st.session_state:
    st.session_state.history = []

if "last_quality" not in st.session_state:
    st.session_state.last_quality = None

if "last_source_text" not in st.session_state:
    st.session_state.last_source_text = ""


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🩺 Sahoolat AI")

    st.caption(
        "AI-powered health information literacy"
    )

    st.divider()

    # Founder
    st.markdown("### 👨‍💻 Founder")

    st.write(
        "Mian Ali Shan"
    )

    st.caption(
        "s/o Saadat Ali"
    )

    st.divider()

    language = st.selectbox(
        "🌐 Output language",
        [
            "English",
            "اردو",
            "Roman Urdu",
        ],
    )

    st.divider()

    st.markdown("### 🔐 Privacy")

    st.caption(
        "Avoid uploading unnecessary CNIC numbers, "
        "phone numbers, addresses and personal identifiers."
    )

    st.divider()

    st.markdown("### 🛡️ Safety")

    st.caption(
        "Sahoolat AI explains medical information. "
        "It does not diagnose or prescribe."
    )

    st.divider()

    st.markdown("### Workflow")

    st.markdown(
        """
**1. Upload**

PDF / Image / Text

**2. Extract**

Medical information

**3. Validate**

Values + confidence + evidence

**4. Explain**

Simple patient language

**5. Prioritize**

Green / Yellow / Red

**6. Prepare**

Doctor questions

**7. Export**

Patient-friendly PDF
"""
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🩺 Sahoolat AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Making complex medical information easier to understand."
    "</div>",
    unsafe_allow_html=True,
)

st.warning(
    "⚠️ Educational health information only. "
    "Sahoolat AI does not diagnose diseases, prescribe medicines, "
    "or replace professional medical care."
)

# ============================================================
# FEATURE CARDS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

cards = [
    (
        c1,
        "📄 Extract",
        "Read medical documents.",
    ),

    (
        c2,
        "🖼️ Vision",
        "Read clear report images.",
    ),

    (
        c3,
        "🔎 Verify",
        "Show confidence and evidence.",
    ),

    (
        c4,
        "🧠 Explain",
        "Make information easier to understand.",
    ),
]

for column, title, description in cards:

    with column:

        st.markdown(
            f"""
<div class="feature-card">
<div class="feature-title">{title}</div>
<div class="feature-description">
{description}
</div>
</div>
""",
            unsafe_allow_html=True,
        )

st.write("")

st.divider()


# ============================================================
# QUICK DEMO
# ============================================================

demo_col1, demo_col2 = st.columns([3, 1])

with demo_col1:

    st.markdown(
        "### 🧪 Try Demo Report"
    )

    st.caption(
        "Use this for a fast hackathon demonstration without uploading a file."
    )

with demo_col2:

    if st.button(
        "🚀 Load Demo",
        use_container_width=True,
    ):

        st.session_state.last_result = demo_report(
            language
        )

        st.session_state.last_source_text = (
            "Demo medical report"
        )

        st.session_state.history.append(
            {
                "source": "Demo Report",
                "attention": "YELLOW",
                "tests": 3,
            }
        )

        st.rerun()


st.divider()


# ============================================================
# INPUT TYPE
# ============================================================

input_type = st.radio(
    "Choose input method",
    [
        "📄 PDF",
        "🖼️ Medical Image",
        "✍️ Text",
    ],
    horizontal=True,
)


# ============================================================
# PDF INPUT
# ============================================================

if input_type == "📄 PDF":

    pdf = st.file_uploader(
        "Upload medical PDF",
        type=["pdf"],
    )

    if pdf:

        st.success(
            f"✓ {pdf.name} uploaded."
        )

        if st.button(
            "✨ Analyze PDF",
            type="primary",
            use_container_width=True,
        ):

            try:

                with st.spinner(
                    "📄 Extracting information from PDF..."
                ):

                    extracted = extract_pdf_text(pdf)

                if not extracted:

                    st.warning(
                        "No readable text was found. "
                        "If this is a scanned PDF, upload a clear image."
                    )

                else:

                    privacy_findings = (
                        detect_personal_information(
                            extracted
                        )
                    )

                    if privacy_findings:

                        st.warning(
                            "🔐 Possible personal information detected: "
                            + ", ".join(
                                privacy_findings
                            )
                        )

                    st.session_state.last_source_text = extracted

                    if contains_emergency_terms(
                        extracted
                    ):

                        st.error(
                            "🚨 Possible current emergency-related "
                            "information detected. If severe symptoms "
                            "are happening now, seek immediate emergency care."
                        )

                    with st.spinner(
                        "🧠 AI is analyzing the report..."
                    ):

                        result = analyze_text(
                            extracted,
                            language,
                        )

                    st.session_state.last_result = result

                    st.session_state.last_quality = {
                        "quality": "Text PDF",
                        "score": 100,
                        "problems": [],
                    }

                    st.session_state.history.append(
                        {
                            "source": pdf.name,
                            "attention":
                                result.get(
                                    "attention_level",
                                    "GREEN",
                                ),
                            "tests":
                                len(
                                    result.get(
                                        "values",
                                        [],
                                    )
                                ),
                        }
                    )

                    st.success(
                        "✓ PDF analysis completed."
                    )

            except Exception as error:

                st.error(
                    friendly_api_error(error)
                )

                with st.expander(
                    "Technical details"
                ):

                    st.code(
                        str(error)
                    )


# ============================================================
# IMAGE INPUT
# ============================================================

elif input_type == "🖼️ Medical Image":

    image_file = st.file_uploader(
        "Upload medical report image",
        type=[
            "png",
            "jpg",
            "jpeg",
            "webp",
        ],
    )

    if image_file:

        try:

            image = Image.open(
                image_file
            ).convert("RGB")

            st.image(
                image,
                caption="Uploaded medical document",
                use_container_width=True,
            )

            quality = assess_image_quality(
                image
            )

            st.session_state.last_quality = quality

            st.markdown(
                "### 📸 Report Quality"
            )

            if quality["quality"] == "Good":

                st.success(
                    f"✓ Good image quality "
                    f"({quality['score']}/100)"
                )

            elif quality["quality"] == "Fair":

                st.warning(
                    f"⚠️ Fair image quality "
                    f"({quality['score']}/100)"
                )

            else:

                st.error(
                    f"❌ Poor image quality "
                    f"({quality['score']}/100)"
                )

            if quality["problems"]:

                for problem in quality["problems"]:

                    st.caption(
                        f"• {problem}"
                    )

            st.caption(
                f"Image size: "
                f"{quality['width']} × {quality['height']}"
            )

            if st.button(
                "✨ Analyze Medical Image",
                type="primary",
                use_container_width=True,
            ):

                with st.spinner(
                    "🧠 Vision model is reading the document..."
                ):

                    result = analyze_image(
                        image,
                        language,
                    )

                st.session_state.last_result = result

                st.session_state.last_source_text = ""

                st.session_state.history.append(
                    {
                        "source": image_file.name,
                        "attention":
                            result.get(
                                "attention_level",
                                "GREEN",
                            ),
                        "tests":
                            len(
                                result.get(
                                    "values",
                                    [],
                                )
                            ),
                    }
                )

                st.success(
                    "✓ Image analysis completed."
                )

        except Exception as error:

            st.error(
                friendly_api_error(error)
            )

            with st.expander(
                "Technical details"
            ):

                st.code(
                    str(error)
                )


# ============================================================
# TEXT INPUT
# ============================================================

else:

    text = st.text_area(
        "Paste medical report or doctor instructions",
        height=250,
        placeholder=(
            "Example:\n\n"
            "Total Cholesterol: 265 mg/dL\n"
            "Reference Range: < 200 mg/dL\n"
            "Triglycerides: 215 mg/dL\n"
            "Reference Range: < 150 mg/dL"
        ),
    )

    if text.strip():

        privacy_findings = (
            detect_personal_information(
                text
            )
        )

        if privacy_findings:

            st.warning(
                "🔐 Possible personal information detected: "
                + ", ".join(
                    privacy_findings
                )
            )

    if st.button(
        "✨ Analyze Medical Text",
        type="primary",
        use_container_width=True,
    ):

        if not text.strip():

            st.warning(
                "Please enter medical information first."
            )

        else:

            st.session_state.last_source_text = text

            if contains_emergency_terms(
                text
            ):

                st.error(
                    "🚨 Possible current emergency-related symptoms "
                    "detected. If severe symptoms are happening now, "
                    "seek immediate emergency medical care."
                )

            try:

                with st.spinner(
                    "🧠 Groq is analyzing..."
                ):

                    result = analyze_text(
                        text,
                        language,
                    )

                st.session_state.last_result = result

                st.session_state.history.append(
                    {
                        "source": "Pasted text",
                        "attention":
                            result.get(
                                "attention_level",
                                "GREEN",
                            ),
                        "tests":
                            len(
                                result.get(
                                    "values",
                                    [],
                                )
                            ),
                    }
                )

                st.success(
                    "✓ Text analysis completed."
                )

            except Exception as error:

                st.error(
                    friendly_api_error(error)
                )

                with st.expander(
                    "Technical details"
                ):

                    st.code(
                        str(error)
                    )


# ============================================================
# RESULTS
# ============================================================

if st.session_state.last_result:

    data = st.session_state.last_result

    st.divider()

    st.markdown(
        "## 📊 Analysis Dashboard"
    )

    # ========================================================
    # REPORT CATEGORY
    # ========================================================

    st.markdown(
        "### 📄 Report Category"
    )

    st.info(
        data.get(
            "report_category",
            "General Medical Report",
        )
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    values = data.get(
        "values",
        [],
    )

    total, within, outside, unknown = (
        calculate_stats(values)
    )

    a, b, c, d = st.columns(4)

    with a:

        st.metric(
            "Tests detected",
            total,
        )

    with b:

        st.metric(
            "🟢 Within range",
            within,
        )

    with c:

        st.metric(
            "🟡 Outside range",
            outside,
        )

    with d:

        st.metric(
            "⚪ Unknown",
            unknown,
        )

    # ========================================================
    # REPORT SUMMARY
    # ========================================================

    st.markdown(
        "### 📄 What the Report Says"
    )

    st.info(
        data.get(
            "summary",
            "",
        )
    )

    # ========================================================
    # VOICE OUTPUT
    # ========================================================

    st.markdown(
        "### 🎤 Listen to Explanation"
    )

    voice_text = data.get(
        "summary",
        "",
    )

    if st.button(
        "🔊 Read Summary Aloud",
        use_container_width=True,
    ):

        voice_output(
            voice_text,
            language,
        )

    st.caption(
        "Voice quality depends on the Urdu voice available "
        "in your browser and operating system."
    )

    # ========================================================
    # EXTRACTION VERIFICATION
    # ========================================================

    verification = verify_extraction(
        data
    )

    st.markdown(
        "### 🔎 Extraction Verification"
    )

    v1, v2 = st.columns(2)

    with v1:

        st.metric(
            "Verification score",
            f"{verification['score']}%",
        )

    with v2:

        st.write(
            verification["label"]
        )

    if verification["checks"]:

        verification_table = []

        for item in verification["checks"]:

            verification_table.append(
                {
                    "Test": item["test"],
                    "Result": item["result"],
                    "Check": item["status"],
                    "Reason": item["reason"],
                }
            )

        st.dataframe(
            verification_table,
            use_container_width=True,
            hide_index=True,
        )

        st.warning(
            "Always compare extracted values with the original report."
        )

    # ========================================================
    # MEDICAL VALUES
    # ========================================================

    st.markdown(
        "### 📊 Medical Values"
    )

    if values:

        table = []

        for item in values:

            table.append(
                {
                    "Test":
                        item.get(
                            "test",
                            "",
                        ),

                    "Result":
                        item.get(
                            "result",
                            "",
                        ),

                    "Unit":
                        item.get(
                            "unit",
                            "",
                        ),

                    "Reference":
                        item.get(
                            "reference_range",
                            "",
                        ),

                    "Status":
                        item.get(
                            "status",
                            "",
                        ),

                    "Confidence":
                        item.get(
                            "confidence",
                            "",
                        ),
                }
            )

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
        )

        with st.expander(
            "🔎 View extraction evidence"
        ):

            for item in values:

                st.markdown(
                    f"**{item.get('test', '')}**"
                )

                st.write(
                    f"Result: {item.get('result', '')} "
                    f"{item.get('unit', '')}"
                )

                st.write(
                    f"Reference: "
                    f"{item.get('reference_range', '')}"
                )

                st.write(
                    f"Confidence: "
                    f"{item.get('confidence', '')}"
                )

                st.write(
                    f"Evidence: "
                    f"{item.get('evidence', '')}"
                )

                st.divider()

    else:

        st.info(
            "No laboratory values were detected."
        )

    # ========================================================
    # ATTENTION
    # ========================================================

    st.markdown(
        "### 🚦 Attention Level"
    )

    level = data.get(
        "attention_level",
        "GREEN",
    )

    if level == "RED":

        st.error(
            "🔴 RED — Possible emergency"
        )

    elif level == "YELLOW":

        st.warning(
            "🟡 YELLOW — Discuss with a healthcare professional"
        )

    else:

        st.success(
            "🟢 GREEN — General educational information"
        )

    st.write(
        data.get(
            "attention_reason",
            "",
        )
    )

    # ========================================================
    # IMPORTANT TERMS
    # ========================================================

    st.markdown(
        "### 📚 Important Medical Terms"
    )

    terms = data.get(
        "important_terms",
        [],
    )

    if terms:

        for term in terms:

            with st.expander(
                term.get(
                    "term",
                    "Medical term",
                )
            ):

                st.write(
                    term.get(
                        "meaning",
                        "",
                    )
                )

    else:

        st.info(
            "No additional terms detected."
        )

    # ========================================================
    # SAFE NEXT STEPS
    # ========================================================

    st.markdown(
        "### ✅ Safe Next Steps"
    )

    next_steps = data.get(
        "next_steps",
        [],
    )

    if next_steps:

        for step in next_steps:

            st.markdown(
                f"• {step}"
            )

    else:

        st.write(
            "Discuss the report with a qualified healthcare professional."
        )

    # ========================================================
    # DOCTOR QUESTIONS
    # ========================================================

    st.markdown(
        "### ❓ Questions to Ask Your Doctor"
    )

    questions = data.get(
        "doctor_questions",
        [],
    )

    if questions:

        for number, question in enumerate(
            questions,
            start=1,
        ):

            st.markdown(
                f"**{number}.** {question}"
            )

    else:

        st.info(
            "No doctor questions generated."
        )

    # ========================================================
    # PRIVACY
    # ========================================================

    privacy_findings = data.get(
        "_privacy_findings",
        [],
    )

    if privacy_findings:

        st.markdown(
            "### 🔐 Privacy Protection"
        )

        st.success(
            "Personal identifiers were detected and redacted "
            "before text analysis."
        )

        for finding in privacy_findings:

            st.caption(
                f"• {finding}"
            )

    # ========================================================
    # SAFETY
    # ========================================================

    st.markdown(
        "### 🛡️ Safety Information"
    )

    st.caption(
        data.get(
            "safety_note",
            "",
        )
    )

    # ========================================================
    # DOWNLOAD
    # ========================================================

    st.markdown(
        "### 📥 Patient-Friendly Summary"
    )

    download_text = make_download_text(
        data
    )

    col_txt, col_pdf = st.columns(2)

    with col_txt:

        st.download_button(
            "📥 Download TXT",
            data=download_text,
            file_name="sahoolat_ai_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with col_pdf:

        try:

            pdf_data = create_pdf(
                data
            )

            st.download_button(
                "📄 Download PDF",
                data=pdf_data,
                file_name="sahoolat_ai_summary.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        except Exception as error:

            st.error(
                "Could not generate PDF."
            )

            st.caption(
                str(error)
            )


# ============================================================
# HISTORY
# ============================================================

if st.session_state.history:

    st.divider()

    st.markdown(
        "## 📈 Recent Analyses"
    )

    for index, item in enumerate(
        reversed(
            st.session_state.history
        ),
        start=1,
    ):

        level = item.get(
            "attention",
            "GREEN",
        )

        icon = {
            "GREEN": "🟢",
            "YELLOW": "🟡",
            "RED": "🔴",
        }.get(
            level,
            "⚪",
        )

        st.write(
            f"{index}. {icon} "
            f"{item.get('source', 'Report')} — "
            f"{item.get('tests', 0)} values detected"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🩺 Sahoolat AI • AI for Health Information Literacy"
)

st.caption(
    "Founder: Mian Ali Shan s/o Saadat Ali"
)

st.caption(
    "Educational information only — not a diagnosis, "
    "prescription, or replacement for professional medical care."
)