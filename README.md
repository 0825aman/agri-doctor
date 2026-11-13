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

---

## Deploying to IBM Cloud Code Engine

This section explains how to containerise and deploy Agri Doctor to **IBM Cloud Code Engine** — IBM's fully managed, serverless container platform.

### Prerequisites

| Requirement | Details |
|-------------|---------|
| IBM Cloud account | Pay-as-you-Go or higher (free tier does not include Code Engine) |
| IBM Cloud CLI | [Install guide](https://cloud.ibm.com/docs/cli?topic=cli-getting-started) |
| Code Engine plugin | Installed automatically by the deploy script |
| IBM watsonx.ai access | Project ID and API key from [cloud.ibm.com](https://cloud.ibm.com) |
| PowerShell 5.1+ | Pre-installed on Windows 10/11; available on macOS/Linux via `pwsh` |

---

### Files Added for Deployment

| File | Purpose |
|------|---------|
| `Dockerfile` | Builds the container image from `python:3.11-slim` |
| `.ceignore` | Excludes secrets, `__pycache__`, `.git`, `.venv`, etc. from the upload |
| `deploy-code-engine.ps1` | End-to-end PowerShell deploy script |

---

### How the Dockerfile Works

```dockerfile
FROM python:3.11-slim          # lightweight base
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py guardrails.py metrics.py dashboard.py .
EXPOSE 8501
CMD ["streamlit", "run", "app.py",
     "--server.port=8501",
     "--server.address=0.0.0.0",
     "--server.headless=true"]
```

Sensitive values (`WATSONX_API_KEY`, `WATSONX_PROJECT_ID`) are **never baked into the image**. They are injected at runtime via a Code Engine secret.

---

### Running the Deploy Script

#### On Windows (PowerShell)

```powershell
# From the repository root:
.\agri_doctor\deploy-code-engine.ps1
```

#### On macOS / Linux (PowerShell Core)

```bash
pwsh agri_doctor/deploy-code-engine.ps1
```

The script will:

1. **Log in** to IBM Cloud (`ibmcloud login --sso`)
2. **Install** the Code Engine plugin if not already present
3. **Create** a Code Engine project named `agri-doctor-project` (skipped if it exists)
4. **Select** the project
5. **Create a secret** named `agri-doctor-secrets` containing:
   - `WATSONX_API_KEY` ← prompted securely (not echoed)
   - `WATSONX_PROJECT_ID` ← prompted
6. **Deploy** the app from the local `agri_doctor/` source directory:
   - Builds the container using the `Dockerfile`
   - Sets non-sensitive env vars inline
   - References the secret for sensitive vars
   - Port `8501`, min-scale `0` (scales to zero when idle), max-scale `3`
7. **Print** the public HTTPS URL

---

### Code Engine Deployment Parameters

| Parameter | Value |
|-----------|-------|
| App name | `agri-doctor` |
| Project | `agri-doctor-project` |
| Region | `us-south` |
| Port | `8501` |
| Min scale | `0` (serverless — no cost when idle) |
| Max scale | `3` |
| CPU | `1 vCPU` |
| Memory | `2 GB` |
| Secret | `agri-doctor-secrets` |

---

### Environment Variables

| Variable | Source | Sensitive |
|----------|--------|-----------|
| `WATSONX_API_KEY` | Code Engine secret | ✅ Yes |
| `WATSONX_PROJECT_ID` | Code Engine secret | ✅ Yes |
| `WATSONX_URL` | Inline env var | No |
| `WATSONX_VISION_MODEL` | Inline env var | No |
| `WATSONX_TEXT_MODEL` | Inline env var | No |

---

### Useful Commands After Deployment

```bash
# View app status and public URL
ibmcloud ce application get --name agri-doctor

# Stream live logs
ibmcloud ce application logs --name agri-doctor --follow

# Update after a code change
ibmcloud ce application update --name agri-doctor --build-source ./agri_doctor

# Delete the app
ibmcloud ce application delete --name agri-doctor

# Delete the project (removes everything)
ibmcloud ce project delete --name agri-doctor-project
```

---

### Troubleshooting

| Problem | Solution |
|---------|----------|
| `FAILED — build error` | Check `ibmcloud ce buildrun list` and inspect logs |
| App returns 502 | Streamlit may still be starting — wait 30s and refresh |
| `unauthorized` on watsonx calls | Re-run the script to update the secret with a fresh API key |
| Port not reachable | Confirm `--port 8501` and `EXPOSE 8501` in Dockerfile match |

