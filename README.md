# Sahoolat AI

## AI-Powered Medical Information Simplifier for Urdu

Sahoolat AI is an AI-powered application designed to help people understand complex medical information in simple Urdu.

Medical reports and health information can contain difficult English terminology that is hard for patients and their families to understand. Sahoolat AI helps simplify this information while keeping important medical details and safety warnings.

## Problem

Many patients in Pakistan receive medical reports and health information in English. Medical terminology can be difficult to understand, especially for people who are more comfortable with Urdu.

This can create confusion about what a report or medical information means.

## Solution

Sahoolat AI uses artificial intelligence to transform complex medical information into easy-to-understand Urdu.

Users can provide medical information and receive:

- Simple Urdu explanations
- Important medical terms explained
- Key information from the provided text
- Possible warning signs
- Questions they can ask a healthcare professional
- Safety guidance when professional medical interpretation is required

## Main Features

### 1. Medical Text Explanation
Users can enter complex medical information and receive a simple Urdu explanation.

### 2. PDF Support
Users can upload medical PDF documents and extract their text for explanation.

### 3. Image Support
Users can upload an image containing medical information and use OCR to extract the text.

### 4. Simple Urdu
The application focuses on making medical information easier to understand for Urdu-speaking users.

### 5. Safety Guidance
Sahoolat AI does not replace a doctor. It provides educational explanations and encourages users to consult qualified healthcare professionals when necessary.

### 6. Emergency Awareness
The system can highlight information that may require urgent professional attention.

## Technology

- Python
- Streamlit
- Groq API
- Large Language Models
- PyPDF
- Pillow
- Tesseract OCR
- python-dotenv

## Architecture

User Input
   |
   v
Text / PDF / Image
   |
   v
Text Extraction
   |
   v
AI Processing
   |
   v
Medical Information Simplification
   |
   v
Simple Urdu Explanation
   |
   v
Safety Guidance

## How to Run

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Sahoolat-AI