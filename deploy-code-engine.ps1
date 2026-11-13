<#
.SYNOPSIS
    Deploy Agri Doctor to IBM Cloud Code Engine.

.DESCRIPTION
    Full automated deployment pipeline — runs end-to-end unattended:
      1. Verify IBM Cloud CLI + Code Engine plugin
      2. Log in with API key from .env
      3. Target Default resource group
      4. Create Code Engine project (idempotent)
      5. Select project
      6. Create / update secret for sensitive env vars
      7. Build & deploy application from local source
      8. Wait for Ready status
      9. Print the public URL

.NOTES
    Prerequisites:
      - IBM Cloud CLI binary at C:\ibmcli\IBM_Cloud_CLI\ibmcloud.exe
        (already installed — run this script from the agri_doctor/ directory)
      - Pay-As-You-Go IBM Cloud account
        Upgrade at: https://cloud.ibm.com/billing/upgrade
      - .env file with WATSONX_API_KEY and WATSONX_PROJECT_ID populated

    Usage:
        cd agri_doctor
        .\deploy-code-engine.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Path: use portable binary installed earlier ────────────────────────────────
$IBMCLOUD = "C:\ibmcli\IBM_Cloud_CLI\ibmcloud.exe"
# Also accept system-installed ibmcloud if the portable one isn't there
if (-not (Test-Path $IBMCLOUD)) {
    $IBMCLOUD = "ibmcloud"
}

# ── Configuration ─────────────────────────────────────────────────────────────
$APP_NAME      = "agri-doctor"
$PROJECT_NAME  = "agri-doctor-project"
$SECRET_NAME   = "agri-doctor-secrets"
$REGION        = "us-south"
$RG            = "Default"
$PORT          = 8501
$MIN_SCALE     = 0
$MAX_SCALE     = 3
$CPU           = "1"
$MEMORY        = "2G"

$WATSONX_URL          = "https://us-south.ml.cloud.ibm.com"
$WATSONX_VISION_MODEL = "mistralai/mistral-small-3-1-24b-instruct-2503"
$WATSONX_TEXT_MODEL   = "ibm/granite-4-h-small"

$SOURCE_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path

# ── Helper: coloured step header ──────────────────────────────────────────────
function Write-Step([string]$n, [string]$msg) {
    Write-Host ""
    Write-Host ("=" * 68) -ForegroundColor Cyan
    Write-Host "  STEP $n — $msg" -ForegroundColor Cyan
    Write-Host ("=" * 68) -ForegroundColor Cyan
}

function Invoke-IBM([string[]]$a) {
    & $IBMCLOUD @a
    if ($LASTEXITCODE -ne 0) { throw "ibmcloud $($a[0..2] -join ' ') failed (exit $LASTEXITCODE)" }
}

# ── Load .env ────────────────────────────────────────────────────────────────
function Get-EnvValue([string]$key) {
    $line = Get-Content "$SOURCE_DIR\.env" -ErrorAction Stop |
            Where-Object { $_ -match "^$key\s*=" } |
            Select-Object -First 1
    if (-not $line) { throw ".env is missing key: $key" }
    return ($line -split "=", 2)[1].Trim()
}

Write-Host ""
Write-Host ("*" * 68) -ForegroundColor Green
Write-Host "  Agri Doctor — IBM Cloud Code Engine Deployment" -ForegroundColor Green
Write-Host ("*" * 68) -ForegroundColor Green

# ── Step 1: Verify CLI ────────────────────────────────────────────────────────
Write-Step "1" "Verify IBM Cloud CLI"
$ver = & $IBMCLOUD version 2>&1
Write-Host $ver
Write-Host "CLI OK." -ForegroundColor Green

# ── Step 2: Install / confirm Code Engine plugin ──────────────────────────────
Write-Step "2" "Code Engine Plugin"
$plugins = & $IBMCLOUD plugin list 2>&1
if ($plugins -notmatch "code-engine") {
    Write-Host "Installing Code Engine plugin..."
    Invoke-IBM @("plugin", "install", "code-engine", "-f")
} else {
    Write-Host "Code Engine plugin already installed." -ForegroundColor Green
}

# ── Step 3: Login ─────────────────────────────────────────────────────────────
Write-Step "3" "IBM Cloud Login (API key from .env)"
$apiKey = Get-EnvValue "WATSONX_API_KEY"
Invoke-IBM @("login", "--apikey", $apiKey, "-r", $REGION, "-g", $RG)
Remove-Variable apiKey
[GC]::Collect()
Write-Host "Logged in." -ForegroundColor Green

# ── Step 4: Create Code Engine project (idempotent) ───────────────────────────
Write-Step "4" "Create Code Engine Project: $PROJECT_NAME"
$existing = & $IBMCLOUD ce project list 2>&1
if ($existing -match $PROJECT_NAME) {
    Write-Host "Project '$PROJECT_NAME' already exists — skipping." -ForegroundColor Yellow
} else {
    Invoke-IBM @("ce", "project", "create", "--name", $PROJECT_NAME)
    Write-Host "Project created." -ForegroundColor Green
}

# ── Step 5: Select project ────────────────────────────────────────────────────
Write-Step "5" "Select Project"
Invoke-IBM @("ce", "project", "select", "--name", $PROJECT_NAME)
Write-Host "Project selected." -ForegroundColor Green

# ── Step 6: Create / update secret ────────────────────────────────────────────
Write-Step "6" "Create / Update Secret: $SECRET_NAME"
$watsonxApiKey = Get-EnvValue "WATSONX_API_KEY"
$watsonxProjId = Get-EnvValue "WATSONX_PROJECT_ID"

$secretCheck = & $IBMCLOUD ce secret get --name $SECRET_NAME 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "Secret exists — updating..." -ForegroundColor Yellow
    Invoke-IBM @("ce", "secret", "update", "--name", $SECRET_NAME,
                 "--from-literal", "WATSONX_API_KEY=$watsonxApiKey",
                 "--from-literal", "WATSONX_PROJECT_ID=$watsonxProjId")
} else {
    Write-Host "Creating secret..."
    Invoke-IBM @("ce", "secret", "create", "--name", $SECRET_NAME,
                 "--from-literal", "WATSONX_API_KEY=$watsonxApiKey",
                 "--from-literal", "WATSONX_PROJECT_ID=$watsonxProjId")
}
Remove-Variable watsonxApiKey, watsonxProjId
[GC]::Collect()
Write-Host "Secret stored." -ForegroundColor Green

# ── Step 7: Build & deploy ────────────────────────────────────────────────────
Write-Step "7" "Build & Deploy Application: $APP_NAME"

$deployArgs = @(
    "--name",            $APP_NAME,
    "--build-source",    $SOURCE_DIR,
    "--port",            "$PORT",
    "--min-scale",       "$MIN_SCALE",
    "--max-scale",       "$MAX_SCALE",
    "--cpu",             $CPU,
    "--memory",          $MEMORY,
    "--env-from-secret", $SECRET_NAME,
    "--env", "WATSONX_URL=$WATSONX_URL",
    "--env", "WATSONX_VISION_MODEL=$WATSONX_VISION_MODEL",
    "--env", "WATSONX_TEXT_MODEL=$WATSONX_TEXT_MODEL",
    "--env", "STREAMLIT_SERVER_HEADLESS=true",
    "--env", "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
)

$appCheck = & $IBMCLOUD ce application get --name $APP_NAME 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "Application exists — updating (re-deploy from source)..." -ForegroundColor Yellow
    Invoke-IBM (@("ce", "application", "update") + $deployArgs)
} else {
    Write-Host "Creating new application from source..." -ForegroundColor Green
    Invoke-IBM (@("ce", "application", "create") + $deployArgs)
}

# ── Step 8: Wait for Ready ────────────────────────────────────────────────────
Write-Step "8" "Waiting for Application to Become Ready"
$tries = 0; $maxTries = 24   # 24 × 15s = 6 minutes
do {
    Start-Sleep -Seconds 15
    $tries++
    $status = & $IBMCLOUD ce application get --name $APP_NAME --output json 2>$null | ConvertFrom-Json
    $ready  = $status.status.conditions | Where-Object { $_.type -eq "Ready" -and $_.status -eq "True" }
    $pct    = [math]::Min(100, $tries * 4)
    Write-Host "  [$pct%] Waiting... (attempt $tries/$maxTries)"
} while (-not $ready -and $tries -lt $maxTries)

if ($ready) {
    Write-Host "  Application is READY!" -ForegroundColor Green
} else {
    Write-Host "  Timed out waiting — check logs below." -ForegroundColor Yellow
}

# ── Step 9: Show URL & logs hint ──────────────────────────────────────────────
Write-Step "9" "Deployment Complete"
$appJson   = & $IBMCLOUD ce application get --name $APP_NAME --output json 2>$null | ConvertFrom-Json
$publicUrl = $appJson.status.url

Write-Host ""
Write-Host ("*" * 68) -ForegroundColor Green
Write-Host "  Agri Doctor is LIVE on IBM Cloud Code Engine!" -ForegroundColor Green
if ($publicUrl) {
    Write-Host "  Public URL : $publicUrl" -ForegroundColor Yellow
} else {
    Write-Host "  Run: ibmcloud ce application get --name $APP_NAME" -ForegroundColor Yellow
    Write-Host "  and look for the 'URL' field." -ForegroundColor Yellow
}
Write-Host ("*" * 68) -ForegroundColor Green
Write-Host ""
Write-Host "Useful commands:" -ForegroundColor Cyan
Write-Host "  Stream logs : $IBMCLOUD ce application logs --name $APP_NAME --follow"
Write-Host "  Get status  : $IBMCLOUD ce application get  --name $APP_NAME"
Write-Host "  List apps   : $IBMCLOUD ce application list"
Write-Host "  Delete app  : $IBMCLOUD ce application delete --name $APP_NAME --force"
Write-Host ""
