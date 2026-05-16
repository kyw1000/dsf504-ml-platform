# DSF504 — Git + GitHub setup script
# Run once from PowerShell in C:\DSF504:
#   cd C:\DSF504
#   .\git_setup.ps1 -GitHubUser YOUR_USERNAME -RepoName dsf504-ml-platform
#
# Prerequisites:
#   1. Git for Windows installed  (https://git-scm.com/download/win)
#   2. A GitHub Personal Access Token with "repo" scope
#      → GitHub.com → Settings → Developer settings → Personal access tokens → Tokens (classic)
#      → Generate new token (classic) → check "repo" → copy the token
#   3. (Optional) GitHub CLI: winget install GitHub.cli  then  gh auth login

param(
    [Parameter(Mandatory)] [string] $GitHubUser,
    [string] $RepoName    = "dsf504-ml-platform",
    [string] $Description = "DSF504 Financial AI Analytics Platform — multi-use-case ML dashboard",
    [switch] $Private
)

Set-Location C:\DSF504

# ── 1. Configure git identity ──────────────────────────────────────────────────
git config user.name  $GitHubUser
git config user.email "kyw@fusions360.com"
git config --global init.defaultBranch main

# ── 2. Init repo (skip if already done) ───────────────────────────────────────
if (-not (Test-Path ".git")) {
    git init
    Write-Host "✅ git init done" -ForegroundColor Green
} else {
    Write-Host "ℹ️  Already a git repo" -ForegroundColor Cyan
}

# ── 3. First commit ────────────────────────────────────────────────────────────
git add .
$status = git status --short
if ($status) {
    git commit -m "feat: initial commit — DSF504 ML platform

- dashboard/app.py  (2591 lines, 8-page Streamlit dashboard)
- dashboard/viz_library.py  (market analytics chart library)
- use_case_A_fraud/  01–06 pipeline scripts (IEEE-CIS Fraud)
- use_case_B_credit/ 01–06 pipeline scripts (Give Me Some Credit)
- use_case_C_nlp/    01–06 pipeline scripts (Financial Phrasebank)
- use_case_C_market/ 01–06 pipeline scripts (Optiver Volatility)
- use_case_E_insurance/ 01–06 pipeline scripts (Porto Seguro)
- config.py, utils/, requirements.txt
- .gitignore excludes data/, models/, reports/, logs/"
    Write-Host "✅ Initial commit created" -ForegroundColor Green
} else {
    Write-Host "ℹ️  Nothing to commit" -ForegroundColor Cyan
}

# ── 4. Create GitHub repo (requires gh CLI or PAT) ────────────────────────────
$ghAvailable = Get-Command gh -ErrorAction SilentlyContinue

if ($ghAvailable) {
    # Use GitHub CLI (easiest — run 'gh auth login' first)
    $visibility = if ($Private) { "--private" } else { "--public" }
    gh repo create $RepoName --description $Description $visibility --source=. --remote=origin --push
    Write-Host "✅ GitHub repo created and pushed via gh CLI" -ForegroundColor Green
} else {
    # Fallback: manual PAT approach
    Write-Host ""
    Write-Host "GitHub CLI not found. To create the remote repo:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "OPTION A — GitHub CLI (recommended):" -ForegroundColor White
    Write-Host "  winget install GitHub.cli"
    Write-Host "  gh auth login"
    Write-Host "  gh repo create $RepoName --public --source=. --remote=origin --push"
    Write-Host ""
    Write-Host "OPTION B — Create manually then add remote:" -ForegroundColor White
    Write-Host "  1. Go to https://github.com/new"
    Write-Host "  2. Name: $RepoName   (do NOT initialise with README)"
    Write-Host "  3. Then run:"
    Write-Host "     git remote add origin https://github.com/$GitHubUser/$RepoName.git"
    Write-Host "     git push -u origin main"
    Write-Host ""
    Write-Host "OPTION C — PAT in one command:" -ForegroundColor White
    Write-Host "  `$token = Read-Host 'Paste your GitHub PAT' -AsSecureString"
    Write-Host "  `$plain = [Runtime.InteropServices.Marshal]::PtrToStringAuto("
    Write-Host "               [Runtime.InteropServices.Marshal]::SecureStringToBSTR(`$token))"
    Write-Host "  git remote add origin https://`$plain@github.com/$GitHubUser/$RepoName.git"
    Write-Host "  git push -u origin main"
}

Write-Host ""
Write-Host "── Suggested workflow going forward ──────────────────────────────" -ForegroundColor Cyan
Write-Host "  After each session's changes:"
Write-Host "    git add -A"
Write-Host "    git commit -m 'fix: describe what changed'"
Write-Host "    git push"
Write-Host ""
Write-Host "  Before a big change (e.g. editing app.py):"
Write-Host "    git stash          # save current state"
Write-Host "    # ... make changes ..."
Write-Host "    git stash pop      # restore if needed"
Write-Host ""
Write-Host "  Tag a stable release:"
Write-Host "    git tag -a v1.0 -m 'Stable: all 5 use cases complete'"
Write-Host "    git push --tags"
