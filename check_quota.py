"""
Agri Doctor — IBM watsonx.ai quota & connectivity diagnostic.
Run with:  python check_quota.py
"""
import os, sys
from dotenv import load_dotenv
load_dotenv()

api_key    = os.getenv("WATSONX_API_KEY", "")
project_id = os.getenv("WATSONX_PROJECT_ID", "")
url        = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
text_model = os.getenv("WATSONX_TEXT_MODEL", "ibm/granite-4-h-small")

def mask(v):
    return v[:4] + "****" + v[-4:] if len(v) > 8 else ("SET" if v else "NOT SET")

print("=" * 60)
print("  Agri Doctor — IBM watsonx.ai Diagnostic")
print("=" * 60)
print(f"  API Key    : {mask(api_key)}")
print(f"  Project ID : {mask(project_id)}")
print(f"  Endpoint   : {url}")
print(f"  Text model : {text_model}")
print()

if not api_key or "PASTE" in api_key:
    print("ERROR: WATSONX_API_KEY not set in .env")
    sys.exit(1)
if not project_id or "PASTE" in project_id:
    print("ERROR: WATSONX_PROJECT_ID not set in .env")
    sys.exit(1)

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters

creds = Credentials(url=url, api_key=api_key)

# --- Step 1: Try the primary text model ----------------------------------
print(f"[1/3] Testing text model: {text_model}")
model = ModelInference(model_id=text_model, credentials=creds, project_id=project_id)
params = TextChatParameters(max_tokens=20, temperature=0.1)
try:
    resp  = model.chat(
        messages=[{"role": "user", "content": "Reply with: OK"}],
        params=params,
    )
    reply = resp["choices"][0]["message"]["content"].strip()
    print(f"      PASS  —  model replied: {reply!r}")
except Exception as e:
    code = str(e)
    if "token_quota_reached" in code:
        print("      FAIL  —  token_quota_reached")
        print("             This account's Lite-tier token quota is exhausted.")
        print("             Fix options:")
        print("               A) Upgrade to Pay-As-You-Go at cloud.ibm.com/billing")
        print("               B) Use a different IBM Cloud account / API key")
    elif "401" in code or "unauthorized" in code.lower():
        print("      FAIL  —  401 Unauthorized — check API key")
    elif "404" in code or "not found" in code.lower():
        print(f"      FAIL  —  model not found: {text_model}")
    else:
        print(f"      FAIL  —  {e}")

# --- Step 2: Try a fallback model ----------------------------------------
fallback = "meta-llama/llama-3-3-70b-instruct"
print(f"\n[2/3] Testing fallback model: {fallback}")
model2 = ModelInference(model_id=fallback, credentials=creds, project_id=project_id)
try:
    resp2  = model2.chat(
        messages=[{"role": "user", "content": "Reply with: OK"}],
        params=params,
    )
    reply2 = resp2["choices"][0]["message"]["content"].strip()
    print(f"      PASS  —  model replied: {reply2!r}")
except Exception as e2:
    code2 = str(e2)
    if "token_quota_reached" in code2:
        print("      FAIL  —  token_quota_reached (same account quota applies)")
    else:
        print(f"      FAIL  —  {e2}")

# --- Step 3: Confirm IAM token generation (auth only) --------------------
print(f"\n[3/3] Confirming IAM authentication (no model call)...")
try:
    from ibm_watsonx_ai import APIClient
    client = APIClient(credentials=creds, project_id=project_id)
    # Lightweight: just list foundation model specs (no token consumption)
    specs = client.foundation_models.get_model_specs(limit=1)
    total = specs.get("total_count", "?")
    print(f"      PASS  —  IAM auth OK, {total} models available on endpoint")
except Exception as e3:
    print(f"      FAIL  —  {e3}")

print()
print("=" * 60)
print("  Diagnostic complete.")
print("=" * 60)
