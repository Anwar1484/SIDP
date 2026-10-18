param([string]$PythonPath, [switch]$Cpu)

# Repair or create the environment using a project-local or user-installed Python.
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$localPython = Join-Path $projectRoot '.runtime\python311\python.exe'
$metadataCode = "import json,sys; print(json.dumps({'executable':sys.executable,'major':sys.version_info.major,'minor':sys.version_info.minor}))"

if ($PythonPath) {
    $pythonCandidate = (Resolve-Path -LiteralPath $PythonPath).Path
    $pythonMetadata = & $pythonCandidate -c $metadataCode
} elseif (Test-Path -LiteralPath $localPython) {
    $pythonCandidate = $localPython
    $pythonMetadata = & $pythonCandidate -c $metadataCode
} else {
    $pythonLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($pythonLauncher) {
        $pythonMetadata = & $pythonLauncher.Source -3 -c $metadataCode
    } else {
        $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
        if (-not $pythonCommand) {
            throw 'Install Python 3.11 or newer, then rerun setup_windows.ps1.'
        }
        $pythonMetadata = & $pythonCommand.Source -c $metadataCode
    }
}
if ($LASTEXITCODE -ne 0) { throw 'The selected Python could not start.' }
$pythonInfo = $pythonMetadata | ConvertFrom-Json
if ($pythonInfo.major -lt 3 -or ($pythonInfo.major -eq 3 -and $pythonInfo.minor -lt 11)) {
    throw 'This project needs Python 3.11 or newer.'
}
$pythonCandidate = $pythonInfo.executable
if ($pythonCandidate -match '\\(?:AppData|LocalCache)\\Roaming\\uv\\|\\\.cache\\codex-runtimes\\') {
    throw 'Select a user-installed Python with -PythonPath; app-managed runtimes may be invisible outside the app.'
}

$environmentPath = [IO.Path]::GetFullPath((Join-Path $projectRoot '.venv'))
$configPath = Join-Path $environmentPath 'pyvenv.cfg'
if (Test-Path -LiteralPath $configPath) {
    $versionLine = Get-Content -LiteralPath $configPath | Where-Object { $_ -match '^version\s*=' }
    if ($versionLine) {
        $previousVersion = (($versionLine -split '=', 2)[1].Trim() -split '\.')[0..1] -join '.'
        $selectedVersion = "$($pythonInfo.major).$($pythonInfo.minor)"
        if ($previousVersion -ne $selectedVersion) {
            $backupPath = [IO.Path]::GetFullPath((Join-Path $projectRoot (
                '.venv-backup-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N').Substring(0, 8)
            )))
            # Confirm both targets remain directly inside this checkout before moving.
            if ((Split-Path $environmentPath -Parent) -ne $projectRoot -or
                (Split-Path $backupPath -Parent) -ne $projectRoot) {
                throw 'Environment backup escaped the project directory.'
            }
            Move-Item -LiteralPath $environmentPath -Destination $backupPath
            Write-Host "Preserved the previous environment at $backupPath"
        }
    }
}

Write-Host "Using Python $($pythonInfo.major).$($pythonInfo.minor): $pythonCandidate"
& $pythonCandidate -m venv --upgrade $environmentPath
if ($LASTEXITCODE -ne 0) { throw 'Creating or repairing .venv failed.' }
$environmentPython = Join-Path $environmentPath 'Scripts\python.exe'
$nvidiaTool = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if (-not $Cpu -and $nvidiaTool) {
    $gpuNames = & $nvidiaTool.Source --query-gpu=name --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0 -and $gpuNames) {
        & $environmentPython -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host 'Installing NVIDIA-enabled PyTorch for the detected GPU.'
            & $environmentPython -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
            if ($LASTEXITCODE -ne 0) { throw 'NVIDIA PyTorch installation failed. Retry setup, or use -Cpu.' }
        }
    }
}
& $environmentPython -m pip install -r (Join-Path $projectRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed; see the output above.' }
& $environmentPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency verification failed.' }
& $environmentPython (Join-Path $projectRoot 'scripts\download_models.py')
if ($LASTEXITCODE -ne 0) { throw 'Model download failed. Rerun setup to resume the missing files.' }
Write-Host 'Setup complete. Launch with .\run_demo.cmd --open-browser. Press Ctrl+C to exit.'
