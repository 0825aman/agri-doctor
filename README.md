# 🌿 Agri Doctor — AI Crop Disease Diagnosis & Advisory

A beginner-friendly **Streamlit** web application that uses a **two-agent AI workflow** powered by **IBM watsonx.ai (Granite 4 H Small)** to diagnose crop diseases from plant images and generate farmer-friendly treatment advisories.

---

## Architecture

```
Farmer uploads image
        │
        ▼
┌─────────────────────────┐
│  Agent 1                │
│  PATHOLOGIST AGENT      │  ← IBM Granite 4 H Small (multimodal)
│  Analyses crop image    │
│  Returns diagnostic     │
│  report                 │
└──────────┬──────────────┘
           │ passes diagnosis
           ▼
┌─────────────────────────┐
│  Agent 2                │
│  EXTENSION OFFICER      │  ← IBM Granite 4 H Small (text)
│  Agent                  │
│  Generates farmer-      │
│  friendly advisory      │
└──────────┬──────────────┘
           │
           ▼
  Combined Final Report
  (downloadable .txt)
```

---

## Features

| Feature | Details |
|---------|---------|
| 🔬 Pathologist Agent | Analyses uploaded JPG/JPEG/PNG crop images; identifies diseases, symptoms, confidence level, and alternative diagnoses |
| 👨‍🌾 Extension Officer Agent | Receives diagnostic report; generates farmer-friendly advice in plain language |
| 🔗 Multi-Agent Collaboration | Agent 1's output is explicitly passed to Agent 2 — genuine pipeline |
| 🔑 Secure API key input | Password-type sidebar field; key never logged or displayed |
| ⬇️ Downloadable report | Full combined advisory exportable as a `.txt` file |
| ⚠️ Safety labels | AI-assessment disclaimers, pesticide safety warnings |
| ❌ Graceful error handling | Auth errors, model not found, rate limits, timeouts |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the application

```bash
streamlit run app.py
```

### 3. In the sidebar

1. Enter your **IBM Cloud API Key** (password field).
2. Enter your **IBM watsonx Project ID**.
3. Confirm the model names (defaults to `ibm/granite-4-0-tiny-preview`).

### 4. Upload and diagnose

1. Upload a clear JPG/JPEG/PNG photo of the affected crop leaf.
2. Click **🔍 Diagnose Crop**.
3. Watch both agents work in sequence.
4. Download the combined advisory report.

---

## Project Structure

```
agri_doctor/
├── app.py            # Main Streamlit application
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## IBM watsonx Credentials

| Item | Value |
|------|-------|
| Public Endpoint | `https://us-south.ml.cloud.ibm.com` |
| LLM | `ibm/granite-4-0-tiny-preview` (Granite 4 H Small) |
| Project ID | Enter in sidebar |
| API Key | Enter in sidebar (never hardcoded) |

> **Note:** If `ibm/granite-4-0-tiny-preview` is not yet available in your region, update the model ID in the sidebar settings. Check [IBM watsonx.ai supported models](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models.html).

---

## Safety & Disclaimers

- All diagnoses are **AI-generated preliminary assessments**, not confirmed laboratory results.
- Chemical treatment recommendations are general guidance only — always follow product labels and local agricultural regulations.
- Always consult a certified plant pathologist or local agricultural extension officer for official confirmation.

---

## License

For educational and demonstration purposes only.
