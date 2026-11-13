<#
.SYNOPSIS
    Deploy Agri Doctor to IBM Cloud Code Engine.

.DESCRIPTION
    This script automates the full deployment pipeline:
      1. IBM Cloud CLI login
      2. Code Engine plugin installation
      3. Project creation / selection
      4. Secret creation for sensitive env vars
      5. Application create or update
      6. Print the public URL

.NOTES
    Prerequisites:
      - IBM Cloud CLI installed  : https://cloud.ibm.com/docs/cli
      - A Pay-as-you-Go IBM Cloud account
      - Your IBM Cloud API key (you will be prompted)

    Run from the REPO ROOT (parent of agri_doctor/):
        .\agri_doctor\deploy-code-engine.ps1

    Or from inside agri_doctor/:
        .\deploy-code-engine.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Configuration ─────────────────────────────────────────────────────────────
$APP_NAME        = "agri-doctor"
$PROJECT_NAME    = "agri-doctor-project"
$SECRET_NAME     = "agri-doctor-secrets"
$REGION          = "us-south"
$PORT            = 8501
$MIN_SCALE       = 0
$MAX_SCALE       = 3
$CPU             = "1"
$MEMORY          = "2G"

# Non-sensitive env vars (set inline, not as secrets)
$WATSONX_URL           = "https://us-south.ml.cloud.ibm.com"
$WATSONX_VISION_MODEL  = "mistralai/mistral-small-3-1-24b-instruct-2503"
$WATSONX_TEXT_MODEL    = "ibm/granite-4-h-small"

# Resolve the agri_doctor/ source directory (works whether run from repo root
# or from inside agri_doctor/)
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$SOURCE_DIR = $SCRIPT_DIR   # Dockerfile lives in agri_doctor/

# ── Helper ────────────────────────────────────────────────────────────────────
function Write-Step([string]$msg) {
    Write-Host ""
    Write-Host "====================================================================" -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host "====================================================================" -ForegroundColor Cyan
}

function Invoke-CE([string[]]$args) {
    # Wrapper that runs ibmcloud and surfaces exit codes
    & ibmcloud @args
    if ($LASTEXITCODE -ne 0) {
        throw "ibmcloud command failed with exit code $LASTEXITCODE"
    }
}

# ── Step 1: IBM Cloud CLI login ───────────────────────────────────────────────
Write-Step "STEP 1 — IBM Cloud Login"
Write-Host "You will be prompted for your IBM Cloud credentials." -ForegroundColor Yellow
ibmcloud login --sso -r $REGION
if ($LASTEXITCODE -ne 0) {
    # Fall back to API-key login if SSO not available
    $apiKey = Read-Host "SSO not available. Enter your IBM Cloud API key" -AsSecureString
    $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
                    [Runtime.InteropServices.Marshal]::SecureStringToBSTR($apiKey))
    Invoke-CE @("login", "--apikey", $plainKey, "-r", $REGION)
    Remove-Variable plainKey
}

# ── Step 2: Install / update Code Engine plugin ────────────────────────────────
Write-Step "STEP 2 — Install / Update Code Engine Plugin"
ibmcloud plugin install code-engine --force
# Ignore exit code — plugin may already be up to date

# ── Step 3: Create Code Engine project (idempotent) ───────────────────────────
Write-Step "STEP 3 — Create Code Engine Project: $PROJECT_NAME"
$existingProject = ibmcloud ce project list --output json 2>$null |
    ConvertFrom-Json |
    Where-Object { $_.name -eq $PROJECT_NAME }

if ($existingProject) {
    Write-Host "Project '$PROJECT_NAME' already exists — skipping creation." -ForegroundColor Green
} else {
    Invoke-CE @("ce", "project", "create", "--name", $PROJECT_NAME)
}

# ── Step 4: Select the project ────────────────────────────────────────────────
Write-Step "STEP 4 — Select Project: $PROJECT_NAME"
Invoke-CE @("ce", "project", "select", "--name", $PROJECT_NAME)

# ── Step 5: Create / update secret for sensitive env vars ─────────────────────
Write-Step "STEP 5 — Create Secret: $SECRET_NAME"

# Prompt for sensitive values (never echoed to console)
$watsonxApiKey   = Read-Host "Enter WATSONX_API_KEY (input hidden)" -AsSecureString
$watsonxProjId   = Read-Host "Enter WATSONX_PROJECT_ID"

$plainApiKey = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
                   [Runtime.InteropServices.Marshal]::SecureStringToBSTR($watsonxApiKey))

# Check if secret already exists
$secretExists = $false
try {
    ibmcloud ce secret get --name $SECRET_NAME 2>$null | Out-Null
    $secretExists = ($LASTEXITCODE -eq 0)
} catch { }

if ($secretExists) {
    Write-Host "Secret '$SECRET_NAME' exists — updating..." -ForegroundColor Yellow
    Invoke-CE @("ce", "secret", "update", "--name", $SECRET_NAME,
                "--from-literal", "WATSONX_API_KEY=$plainApiKey",
                "--from-literal", "WATSONX_PROJECT_ID=$watsonxProjId")
} else {
    Invoke-CE @("ce", "secret", "create", "--name", $SECRET_NAME,
                "--from-literal", "WATSONX_API_KEY=$plainApiKey",
                "--from-literal", "WATSONX_PROJECT_ID=$watsonxProjId")
}

# Scrub sensitive value from memory immediately
Remove-Variable plainApiKey
[GC]::Collect()

# ── Step 6: Deploy application (create or update) ─────────────────────────────
Write-Step "STEP 6 — Deploy Application: $APP_NAME"

# Common arguments used by both create and update
$commonArgs = @(
    "--name",          $APP_NAME,
    "--port",          "$PORT",
    "--min-scale",     "$MIN_SCALE",
    "--max-scale",     "$MAX_SCALE",
    "--cpu",           $CPU,
    "--memory",        $MEMORY,
    # Reference the secret — each key becomes an env var inside the container
    "--env-from-secret", $SECRET_NAME,
    # Non-sensitive env vars set inline
    "--env", "WATSONX_URL=$WATSONX_URL",
    "--env", "WATSONX_VISION_MODEL=$WATSONX_VISION_MODEL",
    "--env", "WATSONX_TEXT_MODEL=$WATSONX_TEXT_MODEL",
    "--env", "STREAMLIT_SERVER_HEADLESS=true",
    "--env", "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
)

# Check if the application already exists
$appExists = $false
try {
    ibmcloud ce application get --name $APP_NAME 2>$null | Out-Null
    $appExists = ($LASTEXITCODE -eq 0)
} catch { }

if ($appExists) {
    Write-Host "Application '$APP_NAME' exists — updating (re-build from source)..." -ForegroundColor Yellow
    Invoke-CE (@("ce", "application", "update",
                 "--build-source", $SOURCE_DIR) + $commonArgs)
} else {
    Write-Host "Creating application '$APP_NAME' from source..." -ForegroundColor Green
    Invoke-CE (@("ce", "application", "create",
                 "--build-source", $SOURCE_DIR) + $commonArgs)
}

# ── Step 7: Get public URL ────────────────────────────────────────────────────
Write-Step "STEP 7 — Deployment Complete"
Write-Host ""
ibmcloud ce application get --name $APP_NAME

$appJson = ibmcloud ce application get --name $APP_NAME --output json 2>$null | ConvertFrom-Json
$publicUrl = $appJson.status.url

Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Green
Write-Host "  Agri Doctor is LIVE!" -ForegroundColor Green
if ($publicUrl) {
    Write-Host "  Public URL : $publicUrl" -ForegroundColor Green
} else {
    Write-Host "  Run: ibmcloud ce application get --name $APP_NAME" -ForegroundColor Yellow
    Write-Host "  and look for the 'URL' field." -ForegroundColor Yellow
}
Write-Host "=====================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Tip: Stream logs with:" -ForegroundColor Cyan
Write-Host "  ibmcloud ce application logs --name $APP_NAME --follow" -ForegroundColor Cyan
Write-Host ""
