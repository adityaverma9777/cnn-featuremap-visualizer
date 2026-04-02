Set-Location $PSScriptRoot

function Quote-Single([string]$value) {
    return "'" + $value.Replace("'", "''") + "'"
}

function Get-FreeTcpPort([int]$startPort = 8000, [int]$endPort = 8100) {
    for ($port = $startPort; $port -le $endPort; $port++) {
        try {
            $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $port)
            $listener.Start()
            $listener.Stop()
            return $port
        } catch {
            # Try next port.
        }
    }

    throw "No free TCP port found between $startPort and $endPort."
}

$repoRoot = $PSScriptRoot
$frontendDir = Join-Path $repoRoot "frontend"
$requirementsPath = Join-Path $repoRoot "requirements.txt"

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Error "npm is not installed or not available in PATH. Install Node.js first."
    exit 1
}

if (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonCmd = "py"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonCmd = "python"
} else {
    Write-Error "Python is not installed or not available in PATH."
    exit 1
}

# Install backend dependencies only when they are missing.
& $pythonCmd -c "import fastapi, uvicorn, torch, torchvision, PIL, multipart" *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing backend dependencies..."
    & $pythonCmd -m pip install -r $requirementsPath
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

# Install frontend dependencies only if node_modules does not exist.
if (-not (Test-Path (Join-Path $frontendDir "node_modules"))) {
    Write-Host "Installing frontend dependencies..."
    Push-Location $frontendDir
    npm install
    $npmInstallExitCode = $LASTEXITCODE
    Pop-Location
    if ($npmInstallExitCode -ne 0) {
        exit $npmInstallExitCode
    }
}

$repoQuoted = Quote-Single $repoRoot
$frontendQuoted = Quote-Single $frontendDir
$backendPort = Get-FreeTcpPort
$apiBaseUrl = "http://127.0.0.1:$backendPort"
$apiBaseUrlQuoted = Quote-Single $apiBaseUrl

$backendCommand = "Set-Location $repoQuoted; & $pythonCmd -m uvicorn app.main:app --host 127.0.0.1 --port $backendPort --reload"
$frontendCommand = "Set-Location $frontendQuoted; `$env:VITE_API_BASE_URL=$apiBaseUrlQuoted; npm run dev"

Start-Process powershell -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $backendCommand) | Out-Null
Start-Process powershell -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $frontendCommand) | Out-Null

Write-Host "Backend starting at $apiBaseUrl"
Write-Host "Frontend starting via Vite (usually http://127.0.0.1:5173)"
Write-Host "Two PowerShell windows were opened for backend and frontend."
