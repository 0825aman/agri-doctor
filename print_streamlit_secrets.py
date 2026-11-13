"""Print Streamlit Cloud secrets block from .env — run once, copy output."""
import os
from dotenv import load_dotenv
load_dotenv()

key = os.getenv("WATSONX_API_KEY", "")
pid = os.getenv("WATSONX_PROJECT_ID", "")
url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
vm  = os.getenv("WATSONX_VISION_MODEL", "mistralai/mistral-small-3-1-24b-instruct-2503")
tm  = os.getenv("WATSONX_TEXT_MODEL", "ibm/granite-4-h-small")

print("=" * 60)
print("  Paste into Streamlit Cloud: Advanced Settings -> Secrets")
print("=" * 60)
print()
print(f'WATSONX_API_KEY       = "{key}"')
print(f'WATSONX_PROJECT_ID    = "{pid}"')
print(f'WATSONX_URL           = "{url}"')
print(f'WATSONX_VISION_MODEL  = "{vm}"')
print(f'WATSONX_TEXT_MODEL    = "{tm}"')
print()
print("=" * 60)
