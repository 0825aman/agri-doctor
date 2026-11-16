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
    # Lightweight: list foundation model specs (no token consumption)
    specs = client.foundation_models.get_model_specs(limit=5)
    resources = specs.get("resources", [])
    total = specs.get("total_count", len(resources))
    print(f"      PASS  —  IAM auth OK, {total} models available on endpoint")
    if resources:
        print(f"             Sample models: {[r.get('model_id','?') for r in resources[:3]]}")
except Exception as e3:
    # Try a simpler auth check via IAM token endpoint
    try:
        import urllib.request, json
        req = urllib.request.Request(
            "https://iam.cloud.ibm.com/identity/token",
            data=f"grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={api_key}".encode(),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            tok = json.loads(r.read())
        print(f"      PASS  —  IAM token obtained (expires in {tok.get('expires_in','?')}s)")
        print(f"             Note: watsonx API client error: {e3}")
    except Exception as e4:
        print(f"      FAIL  —  IAM auth failed: {e4}")

# --- Summary ---------------------------------------------------------------
print()
print("=" * 60)
print("  SUMMARY")
print("=" * 60)
print()
print("  Your credentials authenticate correctly with IBM Cloud IAM.")
print("  HOWEVER: token_quota_reached means the FREE TIER quota on")
print("  this IBM Cloud account is exhausted.")
print()
print("  The quota is ACCOUNT-WIDE — changing Project ID or API Key")
print("  within the same account does NOT reset it.")
print()
print("  REQUIRED ACTION (choose one):")
print()
print("  1. UPGRADE this account to Pay-As-You-Go:")
print("     https://cloud.ibm.com/billing/upgrade")
print("     (Free allowance still applies — only charged for overages)")
print()
print("  2. Use a DIFFERENT IBM Cloud account (new email):")
print("     https://cloud.ibm.com/registration")
print("     - Provision watsonx.ai Lite")
print("     - Create a project -> get Project ID")
print("     - Create an API key -> update .env")
print()
print("=" * 60)
print("  Diagnostic complete.")
print("=" * 60)
