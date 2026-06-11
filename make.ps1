<#
.SYNOPSIS
    One-command setup and task runner for the Cosine Similarity Deep Dive.

.DESCRIPTION
    Creates a local virtual environment (.venv), installs every dependency, and
    gives you shortcuts to run the lessons and the tests. Windows-first, since
    that is where this repo is developed.

.EXAMPLE
    .\make.ps1 setup
    Create .venv and install everything.

.EXAMPLE
    .\make.ps1 test
    Run the pytest suite (the cross-implementation equality check).

.EXAMPLE
    .\make.ps1 run 01
    Run lesson 01. Accepts 01..10 (matches the NN_*.py file prefix).

.EXAMPLE
    .\make.ps1 all
    Run every lesson 01-10 in order.
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet("setup", "test", "run", "all", "clean", "help")]
    [string]$Task = "help",

    [Parameter(Position = 1)]
    [string]$Arg
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"

function Invoke-Setup {
    if (-not (Test-Path $VenvPy)) {
        Write-Host "Creating virtual environment in .venv ..." -ForegroundColor Cyan
        python -m venv (Join-Path $Root ".venv")
    }
    Write-Host "Upgrading pip ..." -ForegroundColor Cyan
    & $VenvPy -m pip install --upgrade pip
    Write-Host "Installing requirements ..." -ForegroundColor Cyan
    & $VenvPy -m pip install -r (Join-Path $Root "requirements.txt")
    Write-Host "Setup complete. Try:  .\make.ps1 run 01" -ForegroundColor Green
}

function Assert-Venv {
    if (-not (Test-Path $VenvPy)) {
        throw "No .venv found. Run '.\make.ps1 setup' first."
    }
}

switch ($Task) {
    "setup" { Invoke-Setup }
    "test" {
        Assert-Venv
        & $VenvPy -m pytest -q
    }
    "run" {
        Assert-Venv
        if (-not $Arg) { throw "Usage: .\make.ps1 run <NN>  e.g.  .\make.ps1 run 01" }
        $file = Get-ChildItem $Root -Filter "$Arg`_*.py" | Select-Object -First 1
        if (-not $file) { throw "No lesson file matching '$Arg`_*.py'." }
        Write-Host "=== Running $($file.Name) ===" -ForegroundColor Cyan
        & $VenvPy $file.FullName
    }
    "all" {
        Assert-Venv
        Get-ChildItem $Root -Filter "[0-9][0-9]_*.py" | Sort-Object Name | ForEach-Object {
            Write-Host "`n=== $($_.Name) ===" -ForegroundColor Cyan
            & $VenvPy $_.FullName
        }
    }
    "clean" {
        if (Test-Path (Join-Path $Root ".venv")) {
            Remove-Item -Recurse -Force (Join-Path $Root ".venv")
            Write-Host "Removed .venv" -ForegroundColor Green
        }
    }
    default {
        Write-Host "Cosine Similarity Deep Dive - task runner" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "  .\make.ps1 setup      Create .venv and install dependencies"
        Write-Host "  .\make.ps1 test       Run the pytest equality suite"
        Write-Host "  .\make.ps1 run 01     Run a single lesson (01..10)"
        Write-Host "  .\make.ps1 all        Run every lesson in order"
        Write-Host "  .\make.ps1 clean      Delete the virtual environment"
    }
}
