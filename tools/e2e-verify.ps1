#Requires -Version 5.1
<#
.SYNOPSIS
  Opaque-box E2E verification suite for Neuromorph Nano HTTP UI API and MCP stdio server.
.DESCRIPTION
  Launches an isolated neuromorph-nano instance against a temporary NANO_HOME directory and
  ephemeral localhost port, exercises Tier 1 (feature coverage), Tier 2 (boundary/error cases),
  Tier 3 (cross-feature combinations), and Tier 4 (sequential MCP stdio + budget scenarios).
  Default mode (without -BaselineSmoke) enforces strict full M5 mode and exits 1 if any check
  fails or skips.
#>
param(
  [string]$Root = (Split-Path $PSScriptRoot -Parent),
  [string]$Exe = '',
  [int]$Port = 0,
  [switch]$BaselineSmoke,
  [int]$McpTimeoutSeconds = 180
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Net.Http

$StrictMode = -not $BaselineSmoke
$Script:Results = New-Object System.Collections.Generic.List[object]

function Record-Check {
  param(
    [int]$Tier,
    [string]$Area,
    [string]$Name,
    [ValidateSet('PASS', 'FAIL', 'SKIP')]
    [string]$Status,
    [string]$Detail = ''
  )
  $Script:Results.Add([PSCustomObject]@{
    Tier   = $Tier
    Area   = $Area
    Name   = $Name
    Status = $Status
    Detail = $Detail
  })
  $tag = "[$Status][T$Tier][$Area] $Name"
  if ($Detail) { Write-Host "$tag -- $Detail" } else { Write-Host $tag }
}

function Assert-Check {
  param(
    [int]$Tier,
    [string]$Area,
    [string]$Name,
    [bool]$Condition,
    [string]$Detail = ''
  )
  if ($Condition) {
    Record-Check -Tier $Tier -Area $Area -Name $Name -Status 'PASS' -Detail $Detail
  } else {
    Record-Check -Tier $Tier -Area $Area -Name $Name -Status 'FAIL' -Detail $Detail
  }
}

function Get-FreeTcpPort {
  $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
  $listener.Start()
  $freePort = ([System.Net.IPEndPoint]$listener.LocalEndpoint).Port
  $listener.Stop()
  return $freePort
}

function Resolve-NanoExe {
  param([string]$RootDir, [string]$ExplicitExe)
  if ($ExplicitExe) {
    if (-not (Test-Path $ExplicitExe)) { throw "Specified -Exe not found: $ExplicitExe" }
    return (Resolve-Path $ExplicitExe).Path
  }
  $gateExe = Join-Path $RootDir 'scratch\gate\target\release\neuromorph-nano.exe'
  if (Test-Path $gateExe) { return (Resolve-Path $gateExe).Path }
  $relExe = Join-Path $RootDir 'target\release\neuromorph-nano.exe'
  if (Test-Path $relExe) { return (Resolve-Path $relExe).Path }
  throw "No neuromorph-nano.exe found at $gateExe or $relExe"
}

function New-IsolatedHome {
  param([string]$RootDir, [string]$ExePath)
  $tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("nm-e2e-" + [guid]::NewGuid().ToString('N').Substring(0, 12))
  New-Item -ItemType Directory -Path $tmp -Force | Out-Null

  $nanoJson = Join-Path $RootDir 'nano.json'
  if (Test-Path $nanoJson) {
    Copy-Item -Path $nanoJson -Destination (Join-Path $tmp 'nano.json') -Force
  }

  $srcOut = Join-Path $RootDir 'interface\out'
  if (-not (Test-Path $srcOut)) { throw "Static UI build missing at $srcOut" }
  $dstOutParent = Join-Path $tmp 'interface'
  New-Item -ItemType Directory -Path $dstOutParent -Force | Out-Null
  Copy-Item -Path $srcOut -Destination (Join-Path $dstOutParent 'out') -Recurse -Force

  $srcDesigns = Join-Path $RootDir 'designs'
  if (-not (Test-Path $srcDesigns)) { throw "designs directory missing at $srcDesigns" }
  Copy-Item -Path $srcDesigns -Destination (Join-Path $tmp 'designs') -Recurse -Force

  $srcCrates = Join-Path $RootDir 'crates'
  if (-not (Test-Path $srcCrates)) { throw "crates directory missing at $srcCrates" }
  $dstCrates = Join-Path $tmp 'crates'
  New-Item -ItemType Directory -Path $dstCrates -Force | Out-Null
  Get-ChildItem -Path $srcCrates -Directory | ForEach-Object {
    $partJson = Join-Path $_.FullName 'part.json'
    if (Test-Path $partJson) {
      $crateDir = Join-Path $dstCrates $_.Name
      New-Item -ItemType Directory -Path $crateDir -Force | Out-Null
      Copy-Item -Path $partJson -Destination (Join-Path $crateDir 'part.json') -Force
    }
  }

  $dstRel = Join-Path $tmp 'target\release'
  New-Item -ItemType Directory -Path $dstRel -Force | Out-Null
  Copy-Item -Path $ExePath -Destination (Join-Path $dstRel 'neuromorph-nano.exe') -Force

  return $tmp
}

function Stop-ProcessTreeSafe {
  param([System.Diagnostics.Process]$Proc)
  if ($null -eq $Proc) { return }
  try {
    if (-not $Proc.HasExited) {
      & taskkill /PID $Proc.Id /T /F 2>$null | Out-Null
      $null = $Proc.WaitForExit(3000)
    }
  } catch {
    try { if (-not $Proc.HasExited) { $Proc.Kill() } } catch {}
  }
}

function Stop-LiveBinariesInHome {
  param([string]$HomeDir)
  if (-not $HomeDir) { return }
  $needle = (Join-Path $HomeDir 'scratch\live').ToLowerInvariant()
  try {
    $procs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
      $_.ExecutablePath -and $_.ExecutablePath.ToLowerInvariant().StartsWith($needle)
    }
    foreach ($p in $procs) {
      & taskkill /PID $p.ProcessId /T /F 2>$null | Out-Null
    }
  } catch {}
}

$Script:HttpClient = New-Object System.Net.Http.HttpClient
$Script:HttpClient.Timeout = [TimeSpan]::FromSeconds(15)

# 2026-11-26 (Luis Jake Gabriel III: every delete verified in more than one step): a delete is two
# calls. The first returns a preview and a confirm_token and deletes nothing; the same body with the
# token deletes. Returns @{ Preview = <first reply>; Result = <second reply> }.
function Invoke-Delete {
  param([string]$BaseUrl, [string]$Path, [string]$Body, [string]$Token)
  $first = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path $Path -Token $Token -Body $Body
  if ($first.Status -ne 200 -or -not $first.JsonOk -or -not $first.Json.confirm_token) {
    return @{ Preview = $first; Result = $first }
  }
  $obj = $Body | ConvertFrom-Json
  $obj | Add-Member -NotePropertyName 'confirm_token' -NotePropertyValue $first.Json.confirm_token -Force
  $second = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path $Path -Token $Token -Body ($obj | ConvertTo-Json -Compress)
  return @{ Preview = $first; Result = $second }
}

function Invoke-Api {
  param(
    [string]$BaseUrl,
    [string]$Method = 'GET',
    [string]$Path = '/',
    [string]$Body = '',
    [string]$Token = '',
    [string]$ContentType = 'application/json',
    [string]$Origin = '',
    [switch]$OmitContentType
  )
  $uri = $BaseUrl.TrimEnd('/') + $Path
  $req = New-Object System.Net.Http.HttpRequestMessage ([System.Net.Http.HttpMethod]::new($Method)), $uri
  if ($Token) {
    $null = $req.Headers.TryAddWithoutValidation('X-Nano-Token', $Token)
  }
  if ($Origin) {
    $null = $req.Headers.TryAddWithoutValidation('Origin', $Origin)
  }
  if ($Method -eq 'POST' -or $PSBoundParameters.ContainsKey('Body')) {
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($Body)
    $content = New-Object System.Net.Http.ByteArrayContent (,$bytes)
    if (-not $OmitContentType -and $ContentType) {
      $content.Headers.ContentType = [System.Net.Http.Headers.MediaTypeHeaderValue]::Parse($ContentType)
    }
    $req.Content = $content
  }
  $resp = $Script:HttpClient.SendAsync($req).GetAwaiter().GetResult()
  $respBody = $resp.Content.ReadAsStringAsync().GetAwaiter().GetResult()
  $respCtype = ''
  if ($resp.Content.Headers.ContentType) {
    $respCtype = $resp.Content.Headers.ContentType.MediaType
  }
  $jsonObj = $null
  $jsonOk = $false
  if ($respCtype -like '*json*' -or ($respBody.TrimStart().StartsWith('{') -or $respBody.TrimStart().StartsWith('['))) {
    try {
      if ($respBody.Trim() -eq 'null') {
        $jsonOk = $true
        $jsonObj = $null
      } else {
        $jsonObj = $respBody | ConvertFrom-Json
        $jsonOk = $true
      }
    } catch {
      $jsonOk = $false
    }
  }
  return [PSCustomObject]@{
    Status      = [int]$resp.StatusCode
    ContentType = $respCtype
    Body        = $respBody
    JsonOk      = $jsonOk
    Json        = $jsonObj
  }
}

function Has-Prop {
  param($Obj, [string]$Name)
  if ($null -eq $Obj) { return $false }
  return ($null -ne $Obj.PSObject.Properties[$Name])
}

function Start-McpSession {
  param(
    [string]$LaunchExe,
    [string]$RepoRoot,
    [string]$TempHome,
    [string]$ModelsDir,
    [string]$ExtraArgs = ''
  )
  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName = $LaunchExe
  $argStr = 'launch serve --root "' + $RepoRoot + '"'
  if ($ExtraArgs) { $argStr = $argStr + ' ' + $ExtraArgs }
  $psi.Arguments = $argStr
  $psi.WorkingDirectory = $RepoRoot
  $psi.UseShellExecute = $false
  $psi.RedirectStandardInput = $true
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError = $true
  $psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
  $psi.StandardErrorEncoding = [System.Text.Encoding]::UTF8
  $psi.EnvironmentVariables['NANO_HOME'] = $TempHome
  $psi.EnvironmentVariables['NANO_MODELS'] = $ModelsDir
  if ($psi.EnvironmentVariables.ContainsKey('CLAUDE_PROJECT_DIR')) {
    $null = $psi.EnvironmentVariables.Remove('CLAUDE_PROJECT_DIR')
  }
  $proc = [System.Diagnostics.Process]::Start($psi)
  $errTask = $proc.StandardError.ReadToEndAsync()
  return [PSCustomObject]@{
    Proc    = $proc
    ErrTask = $errTask
    NextId  = 1
  }
}

function Invoke-McpRequest {
  param(
    $Session,
    [string]$Method,
    $Params = @{},
    [int]$TimeoutSeconds = 180
  )
  $id = $Session.NextId
  $Session.NextId++
  $payload = @{
    jsonrpc = '2.0'
    id      = $id
    method  = $Method
    params  = $Params
  } | ConvertTo-Json -Depth 10 -Compress
  # Windows PowerShell 5.1 writes a child's standard input in the system code page, which would
  # turn characters outside ASCII (an em dash) into look-alikes; JSON escapes carry them exactly.
  $payload = [regex]::Replace($payload, '[^\x00-\x7F]', { param($m) '\u{0:x4}' -f [int][char]$m.Value })
  $Session.Proc.StandardInput.WriteLine($payload)
  $Session.Proc.StandardInput.Flush()

  $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
  while ([DateTime]::UtcNow -lt $deadline) {
    $readTask = $Session.Proc.StandardOutput.ReadLineAsync()
    $remainingMs = [int]([Math]::Max(100, ($deadline - [DateTime]::UtcNow).TotalMilliseconds))
    if (-not $readTask.Wait($remainingMs)) {
      throw "Timed out waiting for MCP response id=$id ($Method) after ${TimeoutSeconds}s"
    }
    $line = $readTask.Result
    if ($null -eq $line) {
      $errTxt = ''
      try { if ($Session.ErrTask.IsCompleted) { $errTxt = $Session.ErrTask.Result } } catch {}
      throw "MCP server EOF waiting for id=$id ($Method). stderr=$errTxt"
    }
    $trimmed = $line.Trim()
    if (-not $trimmed) { continue }
    $obj = $trimmed | ConvertFrom-Json
    if ((Has-Prop $obj 'id') -and ([int]$obj.id -eq $id)) {
      return $obj
    }
  }
  throw "Timed out waiting for MCP response id=$id ($Method)"
}

function Send-McpNotification {
  param($Session, [string]$Method)
  $payload = @{ jsonrpc = '2.0'; method = $Method } | ConvertTo-Json -Compress
  # Windows PowerShell 5.1 writes a child's standard input in the system code page, which would
  # turn characters outside ASCII (an em dash) into look-alikes; JSON escapes carry them exactly.
  $payload = [regex]::Replace($payload, '[^\x00-\x7F]', { param($m) '\u{0:x4}' -f [int][char]$m.Value })
  $Session.Proc.StandardInput.WriteLine($payload)
  $Session.Proc.StandardInput.Flush()
}

function Close-McpSession {
  param($Session, [int]$WaitSeconds = 45)
  if ($null -eq $Session -or $null -eq $Session.Proc) { return }
  try { $Session.Proc.StandardInput.Close() } catch {}
  try {
    if (-not $Session.Proc.WaitForExit($WaitSeconds * 1000)) {
      Stop-ProcessTreeSafe -Proc $Session.Proc
    }
  } catch {
    Stop-ProcessTreeSafe -Proc $Session.Proc
  }
}

$ResolvedExe = Resolve-NanoExe -RootDir $Root -ExplicitExe $Exe
$ResolvedPort = if ($Port -gt 0) { $Port } else { Get-FreeTcpPort }
$BaseUrl = "http://127.0.0.1:$ResolvedPort"
$ModelsDir = Join-Path $Root 'weights'

$OrigNanoHome = $env:NANO_HOME
$OrigNanoModels = $env:NANO_MODELS
$OrigClaudeDir = $env:CLAUDE_PROJECT_DIR

$TempHome = $null
$UiProc = $null
$UiErrTask = $null
$McpSession = $null

Write-Host "E2E Verify: exe=$ResolvedExe port=$ResolvedPort mode=$(if ($StrictMode) { 'FULL_M5_STRICT' } else { 'BASELINE_SMOKE' })"

try {
  $TempHome = New-IsolatedHome -RootDir $Root -ExePath $ResolvedExe
  $LaunchExe = Join-Path $TempHome 'target\release\neuromorph-nano.exe'
  Write-Host "E2E Verify: isolated NANO_HOME=$TempHome"

  # Set env vars for this process and child processes, then restore in finally.
  $env:NANO_HOME = $TempHome
  $env:NANO_MODELS = $ModelsDir
  Remove-Item Env:CLAUDE_PROJECT_DIR -ErrorAction SilentlyContinue

  # Launch `neuromorph-nano launch ui --port <port>`
  $uiPsi = New-Object System.Diagnostics.ProcessStartInfo
  $uiPsi.FileName = $LaunchExe
  $uiPsi.Arguments = "launch ui --port $ResolvedPort"
  $uiPsi.WorkingDirectory = $Root
  $uiPsi.UseShellExecute = $false
  $uiPsi.RedirectStandardInput = $true
  $uiPsi.RedirectStandardOutput = $true
  $uiPsi.RedirectStandardError = $true
  $uiPsi.EnvironmentVariables['NANO_HOME'] = $TempHome
  $uiPsi.EnvironmentVariables['NANO_MODELS'] = $ModelsDir
  if ($uiPsi.EnvironmentVariables.ContainsKey('CLAUDE_PROJECT_DIR')) {
    $null = $uiPsi.EnvironmentVariables.Remove('CLAUDE_PROJECT_DIR')
  }
  $UiProc = [System.Diagnostics.Process]::Start($uiPsi)
  $UiErrTask = $UiProc.StandardError.ReadToEndAsync()

  # Wait for HTTP readiness on /
  $ready = $false
  $htmlBody = ''
  $lastProbeErr = ''
  for ($i = 0; $i -lt 60; $i++) {
    if ($UiProc.HasExited) {
      $errOut = ''
      try { $errOut = $UiErrTask.Result } catch {}
      throw "launch ui exited prematurely with code $($UiProc.ExitCode): $errOut"
    }
    try {
      $probe = Invoke-Api -BaseUrl $BaseUrl -Method 'GET' -Path '/'
      if ($probe.Status -eq 200 -and $probe.Body -match '<meta name="nano-token"') {
        $ready = $true
        $htmlBody = $probe.Body
        break
      }
      $lastProbeErr = "status=$($probe.Status)"
    } catch {
      $lastProbeErr = $_.Exception.Message
    }
    Start-Sleep -Milliseconds 250
  }
  if (-not $ready) { throw "Timed out waiting for launch ui at $BaseUrl ($lastProbeErr)" }

  # Extract 64-hex token
  $tokenMatch = [regex]::Match($htmlBody, '<meta name="nano-token" content="([0-9a-f]{64})">')
  Assert-Check -Tier 1 -Area 'Auth' -Name 'HTML meta nano-token present (64 hex)' -Condition $tokenMatch.Success
  if (-not $tokenMatch.Success) { throw "Cannot proceed without nano-token" }
  $Token = $tokenMatch.Groups[1].Value

  # =========================================================================
  # TIER 1: FEATURE COVERAGE (16 GET endpoints + POST endpoints + budget.*)
  # =========================================================================
  $rStatus = Invoke-Api -BaseUrl $BaseUrl -Path '/api/status'
  Assert-Check -Tier 1 -Area 'HTTP-GET' -Name 'GET /api/status (200, api == 7)' `
    -Condition ($rStatus.Status -eq 200 -and $rStatus.JsonOk -and $rStatus.Json.api -eq 7 -and (Has-Prop $rStatus.Json 'models') -and (Has-Prop $rStatus.Json 'parts'))

  $rControls = Invoke-Api -BaseUrl $BaseUrl -Path '/api/controls'
  $controlsArr = @($rControls.Json)
  Assert-Check -Tier 1 -Area 'HTTP-GET' -Name 'GET /api/controls (200, array >= 13)' `
    -Condition ($rControls.Status -eq 200 -and $rControls.JsonOk -and $controlsArr.Count -ge 13 -and (Has-Prop $controlsArr[0] 'key') -and (Has-Prop $controlsArr[0] 'value'))

  $rParts = Invoke-Api -BaseUrl $BaseUrl -Path '/api/parts'
  Assert-Check -Tier 1 -Area 'HTTP-GET' -Name 'GET /api/parts (200, crate cards)' `
    -Condition ($rParts.Status -eq 200 -and $rParts.JsonOk -and (Has-Prop $rParts.Json 'nmapi') -and (Has-Prop $rParts.Json 'nmengn') -and (Has-Prop $rParts.Json 'nmtype'))

  $rSettings = Invoke-Api -BaseUrl $BaseUrl -Path '/api/settings'
  Assert-Check -Tier 1 -Area 'HTTP-GET' -Name 'GET /api/settings (200, designs)' `
    -Condition ($rSettings.Status -eq 200 -and $rSettings.JsonOk -and (Has-Prop $rSettings.Json 'anatomy') -and (Has-Prop $rSettings.Json 'models'))

  $rCartridges = Invoke-Api -BaseUrl $BaseUrl -Path '/api/cartridges'
  Assert-Check -Tier 1 -Area 'HTTP-GET' -Name 'GET /api/cartridges (200, JSON array)' `
    -Condition ($rCartridges.Status -eq 200 -and $rCartridges.JsonOk -and $rCartridges.Body.TrimStart().StartsWith('['))

  $rLibrary = Invoke-Api -BaseUrl $BaseUrl -Path '/api/cartridges/library'
  Assert-Check -Tier 1 -Area 'HTTP-GET' -Name 'GET /api/cartridges/library (200, catalogue/in_use)' `
    -Condition ($rLibrary.Status -eq 200 -and $rLibrary.JsonOk -and (Has-Prop $rLibrary.Json 'exists') -and (Has-Prop $rLibrary.Json 'in_use'))

  $rProjects = Invoke-Api -BaseUrl $BaseUrl -Path '/api/projects'
  Assert-Check -Tier 1 -Area 'HTTP-GET' -Name 'GET /api/projects (200, serving/projects)' `
    -Condition ($rProjects.Status -eq 200 -and $rProjects.JsonOk -and (Has-Prop $rProjects.Json 'serving') -and (Has-Prop $rProjects.Json 'projects'))

  $rMemory = Invoke-Api -BaseUrl $BaseUrl -Path '/api/memory'
  Assert-Check -Tier 1 -Area 'HTTP-GET' -Name 'GET /api/memory (200, valid JSON)' `
    -Condition ($rMemory.Status -eq 200 -and $rMemory.JsonOk)

  $rTiers = Invoke-Api -BaseUrl $BaseUrl -Path '/api/memory/tiers'
  Assert-Check -Tier 1 -Area 'HTTP-GET' -Name 'GET /api/memory/tiers (200, nano/projects/total)' `
    -Condition ($rTiers.Status -eq 200 -and $rTiers.JsonOk -and (Has-Prop $rTiers.Json 'nano') -and (Has-Prop $rTiers.Json 'total'))

  $rActivity = Invoke-Api -BaseUrl $BaseUrl -Path '/api/activity?window=1h'
  Assert-Check -Tier 1 -Area 'HTTP-GET' -Name 'GET /api/activity (200, window/components/tools)' `
    -Condition ($rActivity.Status -eq 200 -and $rActivity.JsonOk -and $rActivity.Json.window -eq '1h' -and (Has-Prop $rActivity.Json 'components') -and (Has-Prop $rActivity.Json 'calls'))

  $rAudit = Invoke-Api -BaseUrl $BaseUrl -Path '/api/audit'
  Assert-Check -Tier 1 -Area 'HTTP-GET' -Name 'GET /api/audit (200, audit/episodes)' `
    -Condition ($rAudit.Status -eq 200 -and $rAudit.JsonOk -and (Has-Prop $rAudit.Json 'audit') -and (Has-Prop $rAudit.Json 'episodes'))

  $rSession = Invoke-Api -BaseUrl $BaseUrl -Path '/api/session?project=nano'
  Assert-Check -Tier 1 -Area 'HTTP-GET' -Name 'GET /api/session?project=nano (200, sessions)' `
    -Condition ($rSession.Status -eq 200 -and $rSession.JsonOk -and $rSession.Json.project -eq 'nano' -and (Has-Prop $rSession.Json 'sessions'))

  $rLedger = Invoke-Api -BaseUrl $BaseUrl -Path '/api/ledger?project=nano'
  Assert-Check -Tier 1 -Area 'HTTP-GET' -Name 'GET /api/ledger?project=nano (200, chain/packs/rows)' `
    -Condition ($rLedger.Status -eq 200 -and $rLedger.JsonOk -and $rLedger.Json.project -eq 'nano' -and (Has-Prop $rLedger.Json 'chain') -and (Has-Prop $rLedger.Json 'rows'))

  $rOutcomes = Invoke-Api -BaseUrl $BaseUrl -Path '/api/outcomes?project=nano'
  Assert-Check -Tier 1 -Area 'HTTP-GET' -Name 'GET /api/outcomes?project=nano (200, outcomes)' `
    -Condition ($rOutcomes.Status -eq 200 -and $rOutcomes.JsonOk -and $rOutcomes.Json.project -eq 'nano' -and (Has-Prop $rOutcomes.Json 'outcomes'))

  $rExamples = Invoke-Api -BaseUrl $BaseUrl -Path '/api/examples?project=nano'
  Assert-Check -Tier 1 -Area 'HTTP-GET' -Name 'GET /api/examples?project=nano (200, examples/families)' `
    -Condition ($rExamples.Status -eq 200 -and $rExamples.JsonOk -and $rExamples.Json.project -eq 'nano' -and (Has-Prop $rExamples.Json 'examples') -and (Has-Prop $rExamples.Json 'families'))

  $rLearning = Invoke-Api -BaseUrl $BaseUrl -Path '/api/learning?project=nano'
  Assert-Check -Tier 1 -Area 'HTTP-GET' -Name 'GET /api/learning?project=nano (200, 6 cartridges/metrics/levels)' `
    -Condition ($rLearning.Status -eq 200 -and $rLearning.JsonOk -and $rLearning.Json.project -eq 'nano' -and (@($rLearning.Json.cartridges).Count -eq 6) -and (Has-Prop $rLearning.Json 'metrics') -and (Has-Prop $rLearning.Json 'levels'))

  $rDevices = Invoke-Api -BaseUrl $BaseUrl -Path '/api/devices'
  Assert-Check -Tier 1 -Area 'HTTP-GET' -Name 'GET /api/devices (200, network/devices/code_open)' `
    -Condition ($rDevices.Status -eq 200 -and $rDevices.JsonOk -and (Has-Prop $rDevices.Json 'network') -and (Has-Prop $rDevices.Json 'devices') -and (Has-Prop $rDevices.Json 'code_open'))

  # Baseline POST controls + undo
  $modeCtrl = $controlsArr | Where-Object { $_.key -eq 'mode' } | Select-Object -First 1
  $origMode = if ($modeCtrl) { $modeCtrl.value } else { 'full' }
  $altMode = if ($origMode -eq 'fast') { 'full' } else { 'fast' }
  $postModeBody = (@{ key = 'mode'; value = $altMode } | ConvertTo-Json -Compress)
  $rPostMode = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/controls' -Token $Token -Body $postModeBody
  $modeAuditId = if ($rPostMode.JsonOk) { $rPostMode.Json.audit_id } else { $null }
  Assert-Check -Tier 1 -Area 'HTTP-POST' -Name 'POST /api/controls (baseline discrete mode)' `
    -Condition ($rPostMode.Status -eq 200 -and $rPostMode.JsonOk -and -not [string]::IsNullOrEmpty($modeAuditId))

  $rUndoMode = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/undo' -Token $Token -Body (@{ id = $modeAuditId } | ConvertTo-Json -Compress)
  $restoredMode = ($rUndoMode.Json.controls | Where-Object { $_.key -eq 'mode' } | Select-Object -First 1).value
  Assert-Check -Tier 1 -Area 'HTTP-POST' -Name 'POST /api/undo (restores discrete control)' `
    -Condition ($rUndoMode.Status -eq 200 -and $rUndoMode.JsonOk -and $restoredMode -eq $origMode)

  # POST /api/pair/code
  $rPairCode = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/pair/code' -Token $Token -Body '{}'
  Assert-Check -Tier 1 -Area 'HTTP-POST' -Name 'POST /api/pair/code (200, 6-digit code)' `
    -Condition ($rPairCode.Status -eq 200 -and $rPairCode.JsonOk -and ($rPairCode.Json.code -match '^\d{6}$') -and ([int64]$rPairCode.Json.expires_ms -gt 0))

  # 2026-11-26 (Luis: "plug in and unplug the learned units"): every learned unit by compartment;
  # a compartment switched off and back through the audit undo; an unknown unit refused.
  $lu = Invoke-Api -BaseUrl $BaseUrl -Path '/api/learned-units'
  $luKeys = @($lu.Json.compartments | ForEach-Object { $_.key })
  Assert-Check -Tier 1 -Area 'Learned-Units' -Name 'GET /api/learned-units lists compartments with their units (nano and shared always)' `
    -Condition ($lu.Status -eq 200 -and $lu.JsonOk -and ($luKeys -contains 'nano') -and ($luKeys -contains 'shared') -and (Has-Prop $lu.Json 'acting') -and (Has-Prop $lu.Json 'file'))
  $luOff = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/learned-units' -Token $Token -Body '{"compartment":"shared","on":false}'
  $sharedOff = ($luOff.Json.learned_units.compartments | Where-Object { $_.key -eq 'shared' } | Select-Object -First 1).on
  Assert-Check -Tier 1 -Area 'Learned-Units' -Name 'POST /api/learned-units switches a compartment off with an audit row' `
    -Condition ($luOff.Status -eq 200 -and $luOff.JsonOk -and [string]$luOff.Json.audit_id -ne '' -and $sharedOff -eq $false)
  $luUndo = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/undo' -Token $Token -Body (@{ id = $luOff.Json.audit_id } | ConvertTo-Json -Compress)
  $luBack = Invoke-Api -BaseUrl $BaseUrl -Path '/api/learned-units'
  $sharedBack = ($luBack.Json.compartments | Where-Object { $_.key -eq 'shared' } | Select-Object -First 1).on
  Assert-Check -Tier 1 -Area 'Learned-Units' -Name 'The audit undo switches the compartment back on' -Condition ($luUndo.Status -eq 200 -and $sharedBack -eq $true)
  $luBad = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/learned-units' -Token $Token -Body '{"unit":"nano/no-such-unit","on":false}'
  Assert-Check -Tier 2 -Area 'Learned-Units' -Name 'An unknown learned unit is refused (400)' -Condition ($luBad.Status -eq 400)

  # 2026-11-26 (Luis: "connect form our official hugging face cartridge libraries or 3rd parties"):
  # cartridge sources. Nothing here reaches the network: add, the gated remove, undo, refusals.
  $cs = Invoke-Api -BaseUrl $BaseUrl -Path '/api/cartridge-sources'
  Assert-Check -Tier 1 -Area 'Sources' -Name 'GET /api/cartridge-sources lists the official library first, with the open-weights fallback named' `
    -Condition ($cs.Status -eq 200 -and $cs.JsonOk -and $cs.Json.sources[0].id -eq 'official' -and $cs.Json.sources[0].official -eq $true -and [string]$cs.Json.fallback -match 'Qwen3')
  $csAdd = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/cartridge-sources' -Token $Token -Body '{"action":"add","repo":"someone/their-cartridges","revision":"v1"}'
  $added = @($csAdd.Json.sources.sources | Where-Object { $_.id -eq 'someone--their-cartridges' })
  Assert-Check -Tier 1 -Area 'Sources' -Name 'POST add a third-party library (audited, no token sent by default)' `
    -Condition ($csAdd.Status -eq 200 -and [string]$csAdd.Json.audit_id -ne '' -and $added.Count -eq 1 -and $added[0].token -eq $false -and $added[0].revision -eq 'v1')
  $csBad = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/cartridge-sources' -Token $Token -Body '{"action":"add","repo":"not a repo"}'
  $csFetch = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/cartridge-sources' -Token $Token -Body '{"action":"fetch","id":"someone--their-cartridges","path":"stacks/x.stack.json"}'
  Assert-Check -Tier 2 -Area 'Sources' -Name 'A bad repository name and a fetch before browsing are refused (400)' -Condition ($csBad.Status -eq 400 -and $csFetch.Status -eq 400)
  $csPrev = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/cartridge-sources' -Token $Token -Body '{"action":"remove","id":"someone--their-cartridges"}'
  $csRm = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/cartridge-sources' -Token $Token -Body (@{ action = 'remove'; id = 'someone--their-cartridges'; confirm_token = $csPrev.Json.confirm_token } | ConvertTo-Json -Compress)
  $csUndo = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/undo' -Token $Token -Body (@{ id = $csRm.Json.audit_id } | ConvertTo-Json -Compress)
  $csAfter = Invoke-Api -BaseUrl $BaseUrl -Path '/api/cartridge-sources'
  Assert-Check -Tier 1 -Area 'Sources' -Name 'Removing a source previews first, removes with its token, and Audit undo adds it back' `
    -Condition ($csPrev.Json.step -eq 'preview' -and $csRm.Status -eq 200 -and $csUndo.Status -eq 200 -and @($csAfter.Json.sources | Where-Object { $_.id -eq 'someone--their-cartridges' }).Count -eq 1)
  # 2026-11-27 (B3): a base cartridge choice and a role's model change keep their before state, and
  # Audit undo writes it back through the same writer (read back from the isolated settings.json).
  $readSettings = { Get-Content -Raw -LiteralPath (Join-Path $TempHome 'settings.json') | ConvertFrom-Json }
  $bcSet = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/cartridges/bases' -Token $Token -Body '{"model":"qwen3-1.7b","choice":"open"}'
  $bcMid = & $readSettings
  $bcUndo = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/undo' -Token $Token -Body (@{ id = $bcSet.Json.audit_id } | ConvertTo-Json -Compress)
  $bcAfter = & $readSettings
  $bcBack = ($null -eq $bcAfter.base_cartridges) -or ($null -eq $bcAfter.base_cartridges.'qwen3-1.7b')
  Assert-Check -Tier 1 -Area 'Undo' -Name 'A base cartridge choice is undone from Audit (open back to auto)' `
    -Condition ($bcSet.Status -eq 200 -and $bcMid.base_cartridges.'qwen3-1.7b' -eq 'open' -and $bcUndo.Status -eq 200 -and $bcBack) `
    -Detail "set=$($bcSet.Status) undo=$($bcUndo.Status) $($bcUndo.Body)"
  $rmSet = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/models/roles' -Token $Token -Body '{"role":"writer","model":"qwen3-0.6b"}'
  $rmMid = & $readSettings
  $rmUndo = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/undo' -Token $Token -Body (@{ id = $rmSet.Json.audit_id } | ConvertTo-Json -Compress)
  $rmRoles = Invoke-Api -BaseUrl $BaseUrl -Path '/api/models/roles'
  $rmWriter = @($rmRoles.Json.roles | Where-Object { $_.role -eq 'writer' }) | Select-Object -First 1
  Assert-Check -Tier 1 -Area 'Undo' -Name 'A role model change is undone from Audit (the writer back on its default model)' `
    -Condition ($rmSet.Status -eq 200 -and $rmMid.role_models.writer -eq 'qwen3-0.6b' -and $rmUndo.Status -eq 200 -and $rmWriter.model -eq 'qwen3-1.7b') `
    -Detail "set=$($rmSet.Status) $($rmSet.Body) undo=$($rmUndo.Status) writer=$($rmWriter.model)"
  $csOff = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/cartridge-sources' -Token $Token -Body '{"action":"remove","id":"official"}'
  Assert-Check -Tier 2 -Area 'Sources' -Name 'The official library cannot be removed (400)' -Condition ($csOff.Status -eq 400)

  # Stated preference cartridges: the user's words to rules (a preview writes nothing, confirm
  # writes a library version and plugs it in), a draft checked (prose rewritten, evidence and code
  # untouched), an edit as a new version, unplug and undo, the gated removal and its undo.
  $em = [string][char]0x2014
  $prSay = @{ text = "No em dashes. Use 'use' instead of 'utilize'; for .kicad_pcb files, keep power traces at least 0.5 mm wide"; name = 'e2e writing' }
  $prPrev = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/preferences/say' -Token $Token -Body ($prSay | ConvertTo-Json -Compress)
  $prKinds = @($prPrev.Json.rules | ForEach-Object { $_.kind })
  $prList0 = Invoke-Api -BaseUrl $BaseUrl -Path '/api/preferences'
  Assert-Check -Tier 1 -Area 'Preferences' -Name 'POST /api/preferences/say previews each rule (replace, vocabulary, scoped guidance) and writes nothing' `
    -Condition ($prPrev.Status -eq 200 -and $prPrev.Json.step -eq 'preview' -and ($prKinds -join ',') -eq 'replace,vocabulary,guidance' -and @($prList0.Json.cartridges | Where-Object { $_.id -eq 'stated-e2e-writing' }).Count -eq 0) `
    -Detail ($prKinds -join ',')
  $prSay.confirm = $true
  $prW = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/preferences/say' -Token $Token -Body ($prSay | ConvertTo-Json -Compress)
  $prList = Invoke-Api -BaseUrl $BaseUrl -Path '/api/preferences'
  $prC = @($prList.Json.cartridges | Where-Object { $_.id -eq 'stated-e2e-writing' }) | Select-Object -First 1
  Assert-Check -Tier 1 -Area 'Preferences' -Name 'Confirmed, it is a versioned preference cartridge in the library, plugged in, with an audit row' `
    -Condition ($prW.Status -eq 200 -and $prW.Json.key -eq 'stated-e2e-writing@1' -and [string]$prW.Json.audit_id -ne '' -and $prC.plugged -eq $true -and $prC.family -eq 'preference' -and [int]$prList.Json.in_force -ge 3) `
    -Detail "key=$($prW.Json.key) plugged=$($prC.plugged) in_force=$($prList.Json.in_force)"
  $prCarts = Invoke-Api -BaseUrl $BaseUrl -Path '/api/cartridges'
  $prCompat = Invoke-Api -BaseUrl $BaseUrl -Path '/api/cartridges/compat'
  Assert-Check -Tier 1 -Area 'Preferences' -Name 'It is listed with the library cartridges and in the compatibility read' `
    -Condition (@($prCarts.Json | Where-Object { $_.id -eq 'stated-e2e-writing' -and $_.family -eq 'preference' }).Count -eq 1 -and $prCompat.Body -match 'stated-e2e-writing')
  $prDraft = "We utilize the cache $em it is fast.`nsrc/a.rs:3 // utilize $em kept"
  $prA = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/preferences/apply' -Token $Token -Body (@{ text = $prDraft } | ConvertTo-Json -Compress)
  Assert-Check -Tier 1 -Area 'Preferences' -Name 'POST /api/preferences/apply rewrites the prose and leaves the evidence line byte for byte' `
    -Condition ($prA.Status -eq 200 -and $prA.Json.changed -eq $true -and $prA.Json.text -eq "We use the cache, it is fast.`nsrc/a.rs:3 // utilize $em kept") -Detail ([string]$prA.Json.text)
  $prCode = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/preferences/apply' -Token $Token -Body (@{ text = "let x = 1; // $em"; path = 'src/x.rs' } | ConvertTo-Json -Compress)
  Assert-Check -Tier 1 -Area 'Preferences' -Name 'A code draft is checked, never rewritten (the em dash reported with its line)' `
    -Condition ($prCode.Status -eq 200 -and $prCode.Json.mode -eq 'code' -and $prCode.Json.changed -eq $false -and @($prCode.Json.violations).Count -ge 1)
  $prE = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/preferences/edit' -Token $Token -Body (@{ id = 'stated-e2e-writing'; add = 'max line length 100'; drop = @('r2'); confirm = $true } | ConvertTo-Json -Compress)
  $prList2 = Invoke-Api -BaseUrl $BaseUrl -Path '/api/preferences'
  $prC2 = @($prList2.Json.cartridges | Where-Object { $_.id -eq 'stated-e2e-writing' }) | Select-Object -First 1
  Assert-Check -Tier 1 -Area 'Preferences' -Name 'An edit is a new version that stays plugged in' `
    -Condition ($prE.Status -eq 200 -and $prE.Json.key -eq 'stated-e2e-writing@2' -and $prC2.plugged -eq $true -and @($prC2.versions).Count -eq 2 -and (@($prC2.rules | ForEach-Object { $_.id }) -join ',') -eq 'r1,r3,r4') `
    -Detail ((@($prC2.rules | ForEach-Object { $_.id }) -join ','))
  $prOff = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/preferences/plug' -Token $Token -Body '{"id":"stated-e2e-writing","on":false}'
  $prOffUndo = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/undo' -Token $Token -Body (@{ id = $prOff.Json.audit_id } | ConvertTo-Json -Compress)
  $prList3 = Invoke-Api -BaseUrl $BaseUrl -Path '/api/preferences'
  $prC3 = @($prList3.Json.cartridges | Where-Object { $_.id -eq 'stated-e2e-writing' }) | Select-Object -First 1
  Assert-Check -Tier 1 -Area 'Preferences' -Name 'Unplugging is a library use switch, and Audit undo plugs it back in' `
    -Condition ($prOff.Status -eq 200 -and $prOff.Json.plugged -eq $false -and $prOffUndo.Status -eq 200 -and $prC3.plugged -eq $true)
  $prRmPrev = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/preferences/remove' -Token $Token -Body '{"id":"stated-e2e-writing"}'
  $prRm = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/preferences/remove' -Token $Token -Body (@{ id = 'stated-e2e-writing'; confirm_token = $prRmPrev.Json.confirm_token } | ConvertTo-Json -Compress)
  $prGone = @((Invoke-Api -BaseUrl $BaseUrl -Path '/api/preferences').Json.cartridges | Where-Object { $_.id -eq 'stated-e2e-writing' }).Count
  $prRmUndo = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/undo' -Token $Token -Body (@{ id = $prRm.Json.audit_id } | ConvertTo-Json -Compress)
  $prBack = @((Invoke-Api -BaseUrl $BaseUrl -Path '/api/preferences').Json.cartridges | Where-Object { $_.id -eq 'stated-e2e-writing' -and $_.plugged -eq $true }).Count
  Assert-Check -Tier 1 -Area 'Preferences' -Name 'Removing previews first, removes with its token, and Audit undo puts it back plugged in' `
    -Condition ($prRmPrev.Json.step -eq 'preview' -and $prRm.Status -eq 200 -and $prGone -eq 0 -and $prRmUndo.Status -eq 200 -and $prBack -eq 1)
  $prBad = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/preferences/say' -Token $Token -Body '{"text":"no em dashes","scope":"planet:mars"}'
  $prBadEdit = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/preferences/edit' -Token $Token -Body '{"id":"stated-e2e-writing","drop":["r99"],"confirm":true}'
  Assert-Check -Tier 2 -Area 'Preferences' -Name 'A bad scope and an unknown rule are refused (400)' -Condition ($prBad.Status -eq 400 -and $prBadEdit.Status -eq 400)
  # A want: added in the user's words, its strength set per rule (a new version, audited), and a
  # draft that lacks the wanted term noted as information, never as a violation.
  $prWant = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/preferences/edit' -Token $Token -Body (@{ id = 'stated-e2e-writing'; add = 'strongly prefer "colour"'; confirm = $true } | ConvertTo-Json -Compress)
  $prWc = @((Invoke-Api -BaseUrl $BaseUrl -Path '/api/preferences').Json.cartridges | Where-Object { $_.id -eq 'stated-e2e-writing' }) | Select-Object -First 1
  $prWr = @($prWc.rules | Where-Object { $_.kind -eq 'want' }) | Select-Object -First 1
  $prWs = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/preferences/edit' -Token $Token -Body (@{ id = 'stated-e2e-writing'; rule = [string]$prWr.id; strength = 0.25; confirm = $true } | ConvertTo-Json -Compress)
  $prWc2 = @((Invoke-Api -BaseUrl $BaseUrl -Path '/api/preferences').Json.cartridges | Where-Object { $_.id -eq 'stated-e2e-writing' }) | Select-Object -First 1
  $prWr2 = @($prWc2.rules | Where-Object { $_.id -eq $prWr.id }) | Select-Object -First 1
  $prWa = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/preferences/apply' -Token $Token -Body (@{ text = 'The color is red.' } | ConvertTo-Json -Compress)
  Assert-Check -Tier 1 -Area 'Preferences' -Name 'A want is added in plain words, its strength edited per rule, and a draft without it is noted, not flagged' `
    -Condition ($prWant.Status -eq 200 -and $prWr.label -eq 'want' -and [string]$prWr.term -eq 'colour' -and [double]$prWr.strength -eq 1.0 -and $prWs.Status -eq 200 -and [double]$prWr2.strength -eq 0.25 -and $prWa.Status -eq 200 -and @($prWa.Json.wants_missed).Count -ge 1 -and @($prWa.Json.violations).Count -eq 0) `
    -Detail "add=$($prWant.Status) term=$($prWr.term) strength=$($prWr.strength)->$($prWr2.strength) missed=$(@($prWa.Json.wants_missed).Count) violations=$(@($prWa.Json.violations).Count)"
  # Left unplugged for the MCP checks below, which plug it in through nm_learning.
  $null = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/preferences/plug' -Token $Token -Body '{"id":"stated-e2e-writing","on":false}'

  # 2026-11-26: the two-step delete gate. The first call previews and deletes nothing; a wrong or
  # reused token is refused; the same call with its own token deletes.
  $dPrev = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/ledger/clear' -Token $Token -Body '{"project":"nano"}'
  Assert-Check -Tier 1 -Area 'Delete-Gate' -Name 'A delete first answers with a preview and a confirm_token (nothing deleted)' `
    -Condition ($dPrev.Status -eq 200 -and $dPrev.JsonOk -and $dPrev.Json.step -eq 'preview' -and $dPrev.Json.deleted -eq $false -and [string]$dPrev.Json.confirm_token -match '^[0-9a-f]{32,}$' -and (Has-Prop $dPrev.Json.preview 'undo'))
  $dBad = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/examples/clear' -Token $Token -Body (@{ project = 'nano'; confirm_token = $dPrev.Json.confirm_token } | ConvertTo-Json -Compress)
  Assert-Check -Tier 2 -Area 'Delete-Gate' -Name 'A confirm_token made for another request is refused (400)' -Condition ($dBad.Status -eq 400)
  $dReuse = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/ledger/clear' -Token $Token -Body (@{ project = 'nano'; confirm_token = $dPrev.Json.confirm_token } | ConvertTo-Json -Compress)
  Assert-Check -Tier 2 -Area 'Delete-Gate' -Name 'A used confirm_token is refused (400)' -Condition ($dReuse.Status -eq 400)

  # 2026-11-26 (Phase B, every feature in the interface, MCP and the API): tensors, map contents
  # with edit, import and undo, cartridge values, trial, stacks and imports, learning units, lenses,
  # memory notes, pins, sweeps, objectives, the world-model refresh and the MCP clients' connection.
  $pbStatus = (Invoke-Api -BaseUrl $BaseUrl -Path '/api/status').Json
  $pbModel = @($pbStatus.models | Where-Object { $_.card -eq $true } | Select-Object -First 1)
  $pbW = Invoke-Api -BaseUrl $BaseUrl -Path ('/api/models/weights?limit=5' + $(if ($pbModel.Count -gt 0) { '&model=' + $pbModel[0].id } else { '' }))
  if ($pbModel.Count -gt 0) {
    Assert-Check -Tier 1 -Area 'Phase-B' -Name 'GET /api/models/weights lists an installed model''s tensors (names, shapes, dtypes) and clusters' `
      -Condition ($pbW.Status -eq 200 -and $pbW.JsonOk -and [int64]$pbW.Json.tensors -gt 0 -and @($pbW.Json.tensor_list).Count -eq 5 -and (Has-Prop $pbW.Json.tensor_list[0] 'shape') -and (Has-Prop $pbW.Json 'clusters')) -Detail "model=$($pbModel[0].id)"
  } else {
    Assert-Check -Tier 1 -Area 'Phase-B' -Name 'GET /api/models/weights names a missing model as an error' -Condition ($pbW.Status -eq 200 -and $pbW.JsonOk -and (Has-Prop $pbW.Json 'error'))
  }
  $pbBadModel = Invoke-Api -BaseUrl $BaseUrl -Path '/api/models/weights?model=..%2Fx'
  Assert-Check -Tier 2 -Area 'Phase-B' -Name 'GET /api/models/weights refuses a model id that is not a folder name' -Condition ($pbBadModel.JsonOk -and (Has-Prop $pbBadModel.Json 'error'))

  # A map imported from a file on this machine, read, edited (with a backup), undone.
  $pbMapSrc = Join-Path $TempHome 'tol-e2e.json'
  [System.IO.File]::WriteAllText($pbMapSrc, '{"L0":{"low25":0.5,"low50":1.0},"L1":{"low25":0.25,"low50":0.75}}')
  $pbImp = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/parameter-maps/import' -Token $Token -Body (@{ role = 'reranker'; path = $pbMapSrc } | ConvertTo-Json -Compress)
  Assert-Check -Tier 1 -Area 'Phase-B' -Name 'POST /api/parameter-maps/import copies a checked map in with an audit row' `
    -Condition ($pbImp.Status -eq 200 -and $pbImp.JsonOk -and $pbImp.Json.file -eq 'tol-e2e.json' -and $pbImp.Json.replaced -eq $false -and [string]$pbImp.Json.audit_id -ne '')
  $pbImpAgain = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/parameter-maps/import' -Token $Token -Body (@{ role = 'reranker'; path = $pbMapSrc } | ConvertTo-Json -Compress)
  Assert-Check -Tier 2 -Area 'Phase-B' -Name 'An import over an existing map without replace is refused (400)' -Condition ($pbImpAgain.Status -eq 400)
  $pbImpNet = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/parameter-maps/import' -Token $Token -Body '{"role":"reranker","path":"\\\\server\\share\\tol-x.json"}'
  Assert-Check -Tier 2 -Area 'Phase-B' -Name 'An import from a network path is refused (400)' -Condition ($pbImpNet.Status -eq 400)
  $pbRead = Invoke-Api -BaseUrl $BaseUrl -Path '/api/parameter-maps/read?role=reranker&file=tol-e2e.json&layer=1'
  Assert-Check -Tier 1 -Area 'Phase-B' -Name 'GET /api/parameter-maps/read returns one layer of a map' `
    -Condition ($pbRead.Status -eq 200 -and $pbRead.JsonOk -and $pbRead.Json.mode -eq 'value' -and [double]$pbRead.Json.content.low25 -eq 0.25 -and $pbRead.Json.editable -eq $true)
  $pbSave = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/parameter-maps/save' -Token $Token -Body '{"role":"reranker","file":"tol-e2e.json","layer":"1","content":{"low25":0.3,"low50":0.75}}'
  $pbAfter = Invoke-Api -BaseUrl $BaseUrl -Path '/api/parameter-maps/read?role=reranker&file=tol-e2e.json&layer=1'
  Assert-Check -Tier 1 -Area 'Phase-B' -Name 'POST /api/parameter-maps/save edits one layer, keeps a backup and an audit row' `
    -Condition ($pbSave.Status -eq 200 -and $pbSave.JsonOk -and $pbSave.Json.changed -eq $true -and [string]$pbSave.Json.backup -like '*backups*' -and [double]$pbAfter.Json.content.low25 -eq 0.3)
  $pbShape = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/parameter-maps/save' -Token $Token -Body '{"role":"reranker","file":"tol-e2e.json","layer":"1","content":{"low25":0.3}}'
  Assert-Check -Tier 2 -Area 'Phase-B' -Name 'A save that changes the map''s shape is refused (400)' -Condition ($pbShape.Status -eq 400)
  $pbUndo = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/undo' -Token $Token -Body (@{ id = $pbSave.Json.audit_id } | ConvertTo-Json -Compress)
  $pbBack = Invoke-Api -BaseUrl $BaseUrl -Path '/api/parameter-maps/read?role=reranker&file=tol-e2e.json&layer=1'
  Assert-Check -Tier 1 -Area 'Phase-B' -Name 'The audit undo puts the map as it was back' -Condition ($pbUndo.Status -eq 200 -and [double]$pbBack.Json.content.low25 -eq 0.25)
  $pbImpUndo = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/undo' -Token $Token -Body (@{ id = $pbImp.Json.audit_id } | ConvertTo-Json -Compress)
  $pbGone = Invoke-Api -BaseUrl $BaseUrl -Path '/api/parameter-maps/read?role=reranker&file=tol-e2e.json'
  Assert-Check -Tier 1 -Area 'Phase-B' -Name 'The audit undo of an import moves the map out (kept under backups)' -Condition ($pbImpUndo.Status -eq 200 -and (Has-Prop $pbGone.Json 'error'))
  $null = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/parameter-maps/import' -Token $Token -Body (@{ role = 'reranker'; path = $pbMapSrc } | ConvertTo-Json -Compress)

  # 2026-11-26 (Luis Jake Gabriel III: "Successful data from failure data needs to be kept separate"):
  # a map labelled other than healthy leaves the library for the quarantine in the same step, comes
  # back only once labelled healthy again, and a map in use is never labelled failed.
  $qImp = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/parameter-maps/import' -Token $Token -Body (@{ role = 'reranker'; path = $pbMapSrc; name = 'tol-e2e-q.json' } | ConvertTo-Json -Compress)
  $qLabel = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/parameter-maps/lineage' -Token $Token -Body '{"role":"reranker","file":"tol-e2e-q.json","outcome":"control","outcome_why":"e2e: a deliberate baseline"}'
  $qMaps = Invoke-Api -BaseUrl $BaseUrl -Path '/api/parameter-maps'
  $qRole = @($qMaps.Json.roles) | Where-Object { $_.role -eq 'reranker' } | Select-Object -First 1
  $qInArchive = @($qRole.archived | Where-Object { $_.file -eq 'tol-e2e-q.json' }).Count -eq 1
  $qInLibrary = @($qRole.maps | Where-Object { $_.file -eq 'tol-e2e-q.json' }).Count -eq 1
  Assert-Check -Tier 1 -Area 'Maps' -Name 'A map labelled control leaves the parameter map library for the quarantine in the same step' `
    -Condition ($qImp.Status -eq 200 -and $qLabel.Status -eq 200 -and $qLabel.Json.quarantined -eq $true -and $qInArchive -and -not $qInLibrary)
  $qRestoreNo = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/parameter-maps/restore' -Token $Token -Body '{"role":"reranker","files":["tol-e2e-q.json"]}'
  Assert-Check -Tier 1 -Area 'Maps' -Name 'A quarantined map does not come back into the parameter map library while it is not labelled healthy (400)' -Condition ($qRestoreNo.Status -eq 400)
  $qHealthy = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/parameter-maps/lineage' -Token $Token -Body '{"role":"reranker","file":"tol-e2e-q.json","outcome":"used","outcome_why":"e2e"}'
  $qRestore = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/parameter-maps/restore' -Token $Token -Body '{"role":"reranker","files":["tol-e2e-q.json"]}'
  Assert-Check -Tier 1 -Area 'Maps' -Name 'Labelled healthy again, it can be restored' -Condition ($qHealthy.Status -eq 200 -and $qRestore.Status -eq 200)
  $qJf = @($qRole.maps | Where-Object { $_.kind -eq 'job filter' } | Select-Object -First 1)
  if ($qJf.Count -eq 1) {
    $qFail = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/parameter-maps/lineage' -Token $Token -Body (@{ role = 'reranker'; file = $qJf[0].file; outcome = 'failed'; outcome_why = 'e2e' } | ConvertTo-Json -Compress)
    Assert-Check -Tier 1 -Area 'Maps' -Name 'A map in use is not labelled failed or quarantined (it is fixed in place) (400)' -Condition ($qFail.Status -eq 400)
  }
  $null = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/undo' -Token $Token -Body (@{ id = $qImp.Json.audit_id } | ConvertTo-Json -Compress)

  # 2026-11-26 (Luis Jake Gabriel III: "This should all be part of the neuromorph nano and integrated
  # fully into the mcp the api and the interface"): the lab's builds from one route. The isolated home
  # has no upstream weights, so only the contract is checked here (what is refused before a build starts).
  $lbKind = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/lab/build' -Token $Token -Body '{"kind":"sculpt","role":"reader"}'
  $lbShare = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/lab/build' -Token $Token -Body '{"kind":"map","role":"reader","share":0.9}'
  $lbRole = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/lab/build' -Token $Token -Body '{"kind":"base","role":"nobody"}'
  $lbNoTok = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/lab/build' -Body '{"kind":"base","role":"reader"}'
  Assert-Check -Tier 1 -Area 'Lab' -Name 'POST /api/lab/build refuses an unknown kind, a share outside 0.05 to 0.60, an unknown role, and a write without the local token' `
    -Condition ($lbKind.Status -eq 400 -and $lbShare.Status -eq 400 -and $lbRole.Status -eq 400 -and $lbNoTok.Status -eq 403) `
    -Detail "kind=$($lbKind.Status) share=$($lbShare.Status) role=$($lbRole.Status) token=$($lbNoTok.Status)"

  # Distillation: the teacher data view, an import (and its undo), a label refused for an unknown
  # outcome, and the build's refusals before anything starts (a kind and technique that do not go
  # together, one not available yet, an unknown target). No weights are needed for these.
  $dsView = Invoke-Api -BaseUrl $BaseUrl -Path '/api/training/distil'
  Assert-Check -Tier 1 -Area 'Distil' -Name 'GET /api/training/distil lists the teacher data, the kinds, the techniques, which fills which, and what is available' `
    -Condition ($dsView.Status -eq 200 -and $dsView.JsonOk -and @($dsView.Json.targets).Count -eq 4 -and @($dsView.Json.techniques) -contains 'lens' -and @($dsView.Json.built).Count -ge 1 -and @($dsView.Json.fills).Count -ge 4)
  $dsFile = Join-Path $TempHome 'e2e-teacher-pairs.jsonl'
  Set-Content -LiteralPath $dsFile -Encoding utf8 -Value @('{"prompt":"Say what this does: x = 1","completion":"Sets `x` to one.","teacher":"e2e"}', '{"prompt":"Say what this does: y = 2","completion":"Sets `y` to two.","teacher":"e2e"}')
  $dsNoLic = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/training/distil/import' -Token $Token -Body (@{ file = $dsFile } | ConvertTo-Json -Compress)
  $dsImp = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/training/distil/import' -Token $Token -Body (@{ file = $dsFile; set = 'e2e'; licence = 'own' } | ConvertTo-Json -Compress)
  $dsAgain = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/training/distil/import' -Token $Token -Body (@{ file = $dsFile; set = 'e2e'; licence = 'own' } | ConvertTo-Json -Compress)
  $dsBadLabel = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/training/distil/label' -Token $Token -Body '{"set":"e2e","outcome":"great"}'
  Assert-Check -Tier 1 -Area 'Distil' -Name 'POST /api/training/distil/import refuses pairs without a licence, imports with one, passes over pairs already there; a label outside the outcomes is refused' `
    -Condition ($dsNoLic.Status -eq 400 -and $dsImp.Status -eq 200 -and [int]$dsImp.Json.rows -eq 2 -and $dsAgain.Status -eq 200 -and [int]$dsAgain.Json.rows -eq 0 -and $dsBadLabel.Status -eq 400) `
    -Detail "nolicence=$($dsNoLic.Status) import=$($dsImp.Status)/$($dsImp.Json.rows) again=$($dsAgain.Json.rows) label=$($dsBadLabel.Status)"
  $dsUndo = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/undo' -Token $Token -Body (@{ id = $dsImp.Json.audit_id } | ConvertTo-Json -Compress)
  $dsAfter = Invoke-Api -BaseUrl $BaseUrl -Path '/api/training/distil'
  Assert-Check -Tier 1 -Area 'Distil' -Name 'Undo of a teacher data import takes its pairs out again' -Condition ($dsUndo.Status -eq 200 -and [int]$dsAfter.Json.rows_total -eq 0) -Detail "undo=$($dsUndo.Status) rows=$($dsAfter.Json.rows_total)"
  $dsPair = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/lab/build' -Token $Token -Body '{"kind":"distil","role":"writer","target":"filter","technique":"lens"}'
  $dsLater = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/lab/build' -Token $Token -Body '{"kind":"distil","role":"writer","target":"readout","technique":"readout"}'
  $dsTarget = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/lab/build' -Token $Token -Body '{"kind":"distil","role":"writer","target":"poster","technique":"lens"}'
  Assert-Check -Tier 1 -Area 'Distil' -Name 'POST /api/lab/build kind distil refuses a technique that cannot fill the kind, a kind not available yet and an unknown kind, each with its reason' `
    -Condition ($dsPair.Status -eq 400 -and $dsLater.Status -eq 400 -and $dsTarget.Status -eq 400 -and ([string]$dsPair.Body -match 'cannot fill') -and ([string]$dsLater.Body -match 'read-out cartridge')) `
    -Detail "pair=$($dsPair.Status) later=$($dsLater.Status) target=$($dsTarget.Status)"

  # The user's own repositories: a local bare repository stands in for one (nothing leaves the
  # machine); add, back up, list snapshots, take off the list.
  $bkRemote = Join-Path $TempHome 'e2e-backup.git'
  New-Item -ItemType Directory -Force -Path $bkRemote | Out-Null
  & git -c safe.directory=* -C $bkRemote init -q --bare 2>$null | Out-Null
  $bkAdd = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/backups' -Token $Token -Body (@{ action = 'add'; url = $bkRemote; parts = @('memory', 'settings') } | ConvertTo-Json -Compress)
  $bkView = Invoke-Api -BaseUrl $BaseUrl -Path '/api/backups'
  $bkId = [string](@($bkView.Json.repos | Where-Object { $_.url -eq $bkRemote }) | Select-Object -First 1).id
  $bkNow = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/backups' -Token $Token -Body (@{ action = 'backup'; id = $bkId } | ConvertTo-Json -Compress)
  $bkSnaps = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/backups' -Token $Token -Body (@{ action = 'snapshots'; id = $bkId } | ConvertTo-Json -Compress)
  $bkCommit = [string](@($bkSnaps.Json.snapshots) | Select-Object -First 1).commit
  $bkPrev = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/backups' -Token $Token -Body (@{ action = 'preview'; id = $bkId; commit = $bkCommit } | ConvertTo-Json -Compress)
  $bkNoConfirm = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/backups' -Token $Token -Body (@{ action = 'restore'; id = $bkId; commit = $bkCommit } | ConvertTo-Json -Compress)
  $bkRemove = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/backups' -Token $Token -Body (@{ action = 'remove'; id = $bkId } | ConvertTo-Json -Compress)
  Assert-Check -Tier 1 -Area 'Backups' -Name 'POST /api/backups adds your own repository, backs up to it, lists its snapshot, previews a restore, refuses an unconfirmed restore and takes it off the list' `
    -Condition ($bkAdd.Status -eq 200 -and $bkId -and $bkNow.Status -eq 200 -and [int]$bkNow.Json.files -ge 1 -and $bkCommit.Length -eq 40 -and $bkPrev.Status -eq 200 -and $bkNoConfirm.Status -eq 400 -and $bkRemove.Status -eq 200) `
    -Detail "add=$($bkAdd.Status) id=$bkId now=$($bkNow.Status) files=$($bkNow.Json.files) commit=$bkCommit preview=$($bkPrev.Status) noconfirm=$($bkNoConfirm.Status) remove=$($bkRemove.Status)"
  # A live teacher: a teaching set opened, its open prompts taken, answers sent as the agent would.
  $tcStart = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/training/distil/teach' -Token $Token -Body (@{ action = 'start'; set = 'e2e-live'; prompts = @('Name the module that reads cartridges', 'Name the module that writes notes'); instruction = 'one short line'; licence = 'own' } | ConvertTo-Json -Compress)
  $tcNext = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/training/distil/teach' -Token $Token -Body (@{ action = 'next'; set = 'e2e-live' } | ConvertTo-Json -Compress)
  $tcFirst = @($tcNext.Json.prompts) | Select-Object -First 1
  $tcAnswer = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/training/distil/teach' -Token $Token -Body (@{ action = 'answer'; set = 'e2e-live'; teacher = 'e2e-agent'; answers = @(@{ prompt_id = $tcFirst.prompt_id; completion = 'library.rs reads them' }) } | ConvertTo-Json -Compress -Depth 5)
  $tcData = Invoke-Api -BaseUrl $BaseUrl -Path '/api/training/distil'
  Assert-Check -Tier 1 -Area 'Distil' -Name 'A live teacher opens a teaching set, takes its open prompts and its answers join the teacher data, named by the teacher' `
    -Condition ($tcStart.Status -eq 200 -and @($tcNext.Json.prompts).Count -eq 2 -and $tcAnswer.Status -eq 200 -and [int]$tcAnswer.Json.open -eq 1 -and [int]$tcData.Json.rows_total -ge 1) `
    -Detail "start=$($tcStart.Status) next=$(@($tcNext.Json.prompts).Count) answer=$($tcAnswer.Status) open=$($tcAnswer.Json.open) rows=$($tcData.Json.rows_total)"
  if ($tcAnswer.Json.import.audit_id) { Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/undo' -Token $Token -Body (@{ id = $tcAnswer.Json.import.audit_id } | ConvertTo-Json -Compress) | Out-Null }

  # Cartridges: inspect the settings cartridge, set a value (a new version) and undo, a stack made
  # and undone, a stack on trial and back, an import refused for a file that is not a stack file.
  # The library is made at a server's first start or by the library command; make it here.
  $pbPrevHome = $env:NANO_HOME
  $env:NANO_HOME = $TempHome
  & $LaunchExe library list *> $null
  $env:NANO_HOME = $pbPrevHome
  $pbIns = Invoke-Api -BaseUrl $BaseUrl -Path '/api/cartridges/inspect?key=gtx-1650-4gb'
  $pbFastK = ($pbIns.Json.values | Where-Object { $_.key -eq 'fast_k' } | Select-Object -First 1).value
  Assert-Check -Tier 1 -Area 'Phase-B' -Name 'GET /api/cartridges/inspect shows a settings cartridge''s values and stacks' `
    -Condition ($pbIns.Status -eq 200 -and $pbIns.JsonOk -and $pbIns.Json.values_editable -eq $true -and [string]$pbFastK -ne '' -and @($pbIns.Json.stacks).Count -ge 1)
  $pbNewK = if ($pbFastK -eq '5') { '6' } else { '5' }
  $pbVal = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/cartridges/values' -Token $Token -Body (@{ key = $pbIns.Json.key; values = @{ fast_k = $pbNewK } } | ConvertTo-Json -Compress)
  Assert-Check -Tier 1 -Area 'Phase-B' -Name 'POST /api/cartridges/values makes a new version with the value and switches its uses' `
    -Condition ($pbVal.Status -eq 200 -and $pbVal.JsonOk -and $pbVal.Json.changed -eq $true -and $pbVal.Json.key -ne $pbIns.Json.key -and [string]$pbVal.Json.audit_id -ne '')
  $pbValBad = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/cartridges/values' -Token $Token -Body (@{ key = $pbIns.Json.key; values = @{ 'no.such.key' = '1' } } | ConvertTo-Json -Compress)
  Assert-Check -Tier 2 -Area 'Phase-B' -Name 'A value for a key that is not an operating key is refused (400)' -Condition ($pbValBad.Status -eq 400)
  $pbValUndo = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/undo' -Token $Token -Body (@{ id = $pbVal.Json.audit_id } | ConvertTo-Json -Compress)
  Assert-Check -Tier 1 -Area 'Phase-B' -Name 'The audit undo switches the uses back to the stacks they ran' -Condition ($pbValUndo.Status -eq 200 -and $pbValUndo.JsonOk)
  $pbStack = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/cartridges/stack' -Token $Token -Body (@{ id = 'e2e-stack'; use = 'job:e2e'; cartridges = @($pbIns.Json.key); switch = $true } | ConvertTo-Json -Compress)
  Assert-Check -Tier 1 -Area 'Phase-B' -Name 'POST /api/cartridges/stack makes a stack, switches its use, with an audit row' `
    -Condition ($pbStack.Status -eq 200 -and $pbStack.JsonOk -and $pbStack.Json.stack.id -eq 'e2e-stack' -and $pbStack.Json.switched -eq $true)
  $pbStackUndo = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/undo' -Token $Token -Body (@{ id = $pbStack.Json.audit_id } | ConvertTo-Json -Compress)
  $pbLibAfter = (Invoke-Api -BaseUrl $BaseUrl -Path '/api/cartridges/library').Json
  $pbStackLeft = @($pbLibAfter.catalogue.stacks | Where-Object { $_.id -eq 'e2e-stack' })
  Assert-Check -Tier 1 -Area 'Phase-B' -Name 'The audit undo switches the use back and takes the stack out' -Condition ($pbStackUndo.Status -eq 200 -and $pbStackLeft.Count -eq 0)
  $pbGtxStack = @($pbIns.Json.stacks)[0].key
  $pbTrial = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/cartridges/trial' -Token $Token -Body (@{ name = $pbGtxStack; trial = $true } | ConvertTo-Json -Compress)
  Assert-Check -Tier 1 -Area 'Phase-B' -Name 'POST /api/cartridges/trial puts a stack on trial with a library trial row' -Condition ($pbTrial.Status -eq 200 -and $pbTrial.JsonOk -and (Has-Prop $pbTrial.Json 'audit_id'))
  if ($pbTrial.Json.audit_id) {
    $pbTrialUndo = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/undo' -Token $Token -Body (@{ id = $pbTrial.Json.audit_id } | ConvertTo-Json -Compress)
    Assert-Check -Tier 1 -Area 'Phase-B' -Name 'The audit undo takes the stack off trial again' -Condition ($pbTrialUndo.Status -eq 200)
  }
  $pbCartImp = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/cartridges/import' -Token $Token -Body (@{ path = $pbMapSrc } | ConvertTo-Json -Compress)
  Assert-Check -Tier 2 -Area 'Phase-B' -Name 'POST /api/cartridges/import refuses a file that is not a stack file (400)' -Condition ($pbCartImp.Status -eq 400)

  # Cartridge parameters: one read format for every object that holds parameters, and an edit into
  # a new version beside the source (here a job filter written into the isolated home, plugged out,
  # so nothing serves it), the source unchanged, a bad shape refused, and undo setting it aside.
  $cpDir = Join-Path $TempHome 'library\parameter-maps\reranker'
  New-Item -ItemType Directory -Force -Path $cpDir | Out-Null
  $cpSrc = Join-Path $cpDir 'job-filter-e2e-params.json'
  Set-Content -Path $cpSrc -Encoding ascii -Value '{"contract":"nmengn-sockets/1","cartridge":"reranker-e2e-params@1","model":"probe","job":"e2e-params","enabled":false,"parts":[{"target":"L2.o","gain":[1.5,0.5,1.25],"bias":[0.1,-0.2,0.3]},{"target":"L5.swigate","gain":[2.0,1.0],"bias":[0.0,0.5]}]}'
  $cpSrcHash = (Get-FileHash -Path $cpSrc -Algorithm SHA256).Hash
  $cpList = Invoke-Api -BaseUrl $BaseUrl -Path '/api/cartridges/parameters'
  $cpRow = @($cpList.Json.objects | Where-Object { $_.key -eq 'filter:reranker/e2e-params' })[0]
  Assert-Check -Tier 1 -Area 'Parameters' -Name 'GET /api/cartridges/parameters lists every object holding parameters by label alone (no tensors, values or norms read)' `
    -Condition ($cpList.Status -eq 200 -and $cpList.JsonOk -and $cpList.Json.format -eq 'nm-parameters/1' -and $null -ne $cpRow -and $cpRow.state -eq 'plugged out' -and $cpRow.read -eq $false -and -not (Has-Prop $cpRow 'tensors') -and -not (Has-Prop $cpRow 'layers') -and [int64]$cpRow.bytes -gt 0) `
    -Detail "row=$($cpRow | ConvertTo-Json -Compress)"
  $cpSec = Invoke-Api -BaseUrl $BaseUrl -Path '/api/cartridges/parameters?key=filter%3Areranker%2Fe2e-params'
  $cpGain = @($cpSec.Json.tensors | Where-Object { $_.name -eq 'L2.o.gain' })[0]
  Assert-Check -Tier 1 -Area 'Parameters' -Name 'The parameters section gives each point''s gain and bias with layer, shape, norms and a histogram' `
    -Condition ($cpSec.Status -eq 200 -and $cpSec.Json.holds -eq $true -and @($cpSec.Json.tensors).Count -eq 4 -and $cpGain.layer -eq 2 -and $cpGain.size -eq 3 -and [double]$cpGain.max_abs -eq 1.5 -and @($cpGain.summary.histogram.counts).Count -eq 8)
  $cpPage = Invoke-Api -BaseUrl $BaseUrl -Path '/api/cartridges/parameters?key=filter%3Areranker%2Fe2e-params&tensor=L2.o.bias&offset=1&limit=2'
  Assert-Check -Tier 1 -Area 'Parameters' -Name 'A tensor''s values are read by page' -Condition ($cpPage.Status -eq 200 -and @($cpPage.Json.values).Count -eq 2 -and [double]@($cpPage.Json.values)[0] -eq -0.2)
  $cpIns = Invoke-Api -BaseUrl $BaseUrl -Path '/api/cartridges/inspect?key=gtx-1650-4gb'
  Assert-Check -Tier 1 -Area 'Parameters' -Name 'GET /api/cartridges/inspect carries the parameters section; a settings cartridge says it holds no parameters' `
    -Condition ($cpIns.Status -eq 200 -and $cpIns.Json.parameters.holds -eq $false -and [string]$cpIns.Json.parameters.note -eq 'this cartridge holds no parameters')
  $cpBad = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/cartridges/parameters' -Token $Token -Body '{"key":"filter:reranker/e2e-params","values":{"L2.o.gain":[1,2]}}'
  $cpBadScale = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/cartridges/parameters' -Token $Token -Body '{"key":"filter:reranker/e2e-params","scale":9}'
  $cpRo = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/cartridges/parameters' -Token $Token -Body '{"key":"gtx-1650-4gb","scale":0.5}'
  Assert-Check -Tier 2 -Area 'Parameters' -Name 'The shape check refuses a bad edit, a strength above 4 is refused, and a cartridge without parameters is refused (400)' `
    -Condition ($cpBad.Status -eq 400 -and $cpBad.Body -like '*shape check*' -and $cpBadScale.Status -eq 400 -and $cpRo.Status -eq 400) -Detail "shape=$($cpBad.Status) scale=$($cpBadScale.Status) settings=$($cpRo.Status)"
  $cpEdit = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/cartridges/parameters' -Token $Token -Body '{"key":"filter:reranker/e2e-params","scale":0.5,"layers":{"5":"off"}}'
  $cpNew = Join-Path $cpDir 'job-filter-e2e-params-p2.json'
  Assert-Check -Tier 1 -Area 'Parameters' -Name 'POST /api/cartridges/parameters makes a new version (plugged out) with an audit row; the source is unchanged' `
    -Condition ($cpEdit.Status -eq 200 -and $cpEdit.Json.changed -eq $true -and $cpEdit.Json.key -eq 'filter:reranker/e2e-params-p2' -and [string]$cpEdit.Json.audit_id -ne '' -and (Test-Path $cpNew) -and (Get-FileHash -Path $cpSrc -Algorithm SHA256).Hash -eq $cpSrcHash -and @($cpEdit.Json.parameters.tensors).Count -eq 2) `
    -Detail "status=$($cpEdit.Status) key=$($cpEdit.Json.key)"
  if ($cpEdit.Json.audit_id) {
    $cpUndo = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/undo' -Token $Token -Body (@{ id = $cpEdit.Json.audit_id } | ConvertTo-Json -Compress)
    Assert-Check -Tier 1 -Area 'Parameters' -Name 'The audit undo sets the new version aside (into the role''s archive), deleting nothing' `
      -Condition ($cpUndo.Status -eq 200 -and -not (Test-Path $cpNew) -and (Test-Path (Join-Path $cpDir 'archive\job-filter-e2e-params-p2.json')))
  }

  # The selection engine: an operation on a path, previewed (nothing written), then applied; a
  # region read with the map values at the same place; the maps beside the object; an export.
  $cpPrev = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/cartridges/parameters' -Token $Token -Body '{"key":"filter:reranker/e2e-params","op":"set","value":2,"path":"L2.o.gain[0:2]","preview":true}'
  Assert-Check -Tier 1 -Area 'Parameters' -Name 'A preview of an operation on a path returns the difference and writes nothing' `
    -Condition ($cpPrev.Status -eq 200 -and $cpPrev.Json.preview -eq $true -and $cpPrev.Json.diff.changed -eq 2 -and -not (Test-Path (Join-Path $cpDir 'job-filter-e2e-params-p3.json')))
  $cpOp = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/cartridges/parameters' -Token $Token -Body '{"key":"filter:reranker/e2e-params","op":"clamp","lo":0.75,"hi":1.25,"selection":{"tensor":"L2.o.gain"}}'
  Assert-Check -Tier 1 -Area 'Parameters' -Name 'An operation on a selection makes a new version with an audit row' -Condition ($cpOp.Status -eq 200 -and $cpOp.Json.changed -eq $true -and [string]$cpOp.Json.audit_id -ne '')
  $cpReg = Invoke-Api -BaseUrl $BaseUrl -Path '/api/cartridges/parameters?key=filter%3Areranker%2Fe2e-params&path=L2.o.gain%5B1%3A3%5D'
  $cpMaps = Invoke-Api -BaseUrl $BaseUrl -Path '/api/cartridges/parameters?key=filter%3Areranker%2Fe2e-params&maps=1'
  Assert-Check -Tier 1 -Area 'Parameters' -Name 'A region read gives the selected values, their distribution and the map values at the same place; maps lists the maps beside the object' `
    -Condition ($cpReg.Status -eq 200 -and @($cpReg.Json.values).Count -eq 2 -and $null -ne $cpReg.Json.distribution -and (Has-Prop $cpReg.Json 'maps') -and $cpMaps.Status -eq 200 -and (Has-Prop $cpMaps.Json 'maps'))
  $cpExp = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/cartridges/parameters/export' -Token $Token -Body '{"key":"filter:reranker/e2e-params","path":"L2.o.bias"}'
  Assert-Check -Tier 1 -Area 'Parameters' -Name 'POST /api/cartridges/parameters/export writes a selection as nm-parameter-values/1' -Condition ($cpExp.Status -eq 200 -and $cpExp.Json.format -eq 'nm-parameter-values/1' -and (Test-Path ([string]$cpExp.Json.file)))

  # The parameter overlay: edits, masks and freezes held beside the object (memory/overlays/), the
  # object's file never written; a preview writes nothing; a freeze refuses an edit with its reason;
  # unmask restores exactly; clear goes through the delete gate; undo through the audit log.
  $ovKey = 'filter:reranker/e2e-params'
  $ovPrev = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/parameters/overlay' -Token $Token -Body (@{ key = $ovKey; action = 'edit'; op = 'set'; value = 3; path = 'L2.o.gain[0:2]'; preview = $true } | ConvertTo-Json -Compress)
  $ovDir = Join-Path $TempHome 'memory\overlays'
  Assert-Check -Tier 1 -Area 'Overlay' -Name 'An overlay edit previewed gives the difference and when it takes effect, and writes nothing' `
    -Condition ($ovPrev.Status -eq 200 -and $ovPrev.Json.preview -eq $true -and $ovPrev.Json.diff.changed -eq 2 -and [string]$ovPrev.Json.takes_effect.when -eq 'applied at the next call' -and @(Get-ChildItem -Path $ovDir -Filter '*.json' -ErrorAction SilentlyContinue).Count -eq 0) -Detail "status=$($ovPrev.Status)"
  $ovEdit = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/parameters/overlay' -Token $Token -Body (@{ key = $ovKey; action = 'edit'; op = 'set'; value = 3; path = 'L2.o.gain[0:2]' } | ConvertTo-Json -Compress)
  $ovReg = Invoke-Api -BaseUrl $BaseUrl -Path '/api/cartridges/parameters?key=filter%3Areranker%2Fe2e-params&path=L2.o.gain'
  Assert-Check -Tier 1 -Area 'Overlay' -Name 'POST /api/parameters/overlay writes the edit in the overlay with an audit row; region reads show the overlaid values; the job filter file is unchanged' `
    -Condition ($ovEdit.Status -eq 200 -and [string]$ovEdit.Json.audit_id -ne '' -and $null -ne $ovReg.Json.overlay -and [double]@($ovReg.Json.overlay)[0][0] -eq 3 -and (Get-FileHash -Path $cpSrc -Algorithm SHA256).Hash -eq $cpSrcHash) -Detail "status=$($ovEdit.Status) body=$($ovEdit.Body)"
  $ovFreeze = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/parameters/overlay' -Token $Token -Body (@{ key = $ovKey; action = 'freeze'; path = 'L5' } | ConvertTo-Json -Compress)
  $ovRefused = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/parameters/overlay' -Token $Token -Body (@{ key = $ovKey; action = 'mask'; path = 'L5.swigate.bias' } | ConvertTo-Json -Compress)
  Assert-Check -Tier 2 -Area 'Overlay' -Name 'A freeze makes its region read-only: a mask touching it is refused with the freeze named (400)' `
    -Condition ($ovFreeze.Status -eq 200 -and $ovRefused.Status -eq 400 -and $ovRefused.Body -like '*frozen by*') -Detail "freeze=$($ovFreeze.Status) mask=$($ovRefused.Status)"
  $ovMask = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/parameters/overlay' -Token $Token -Body (@{ key = $ovKey; action = 'mask'; path = 'L2' } | ConvertTo-Json -Compress)
  $ovMasked = Invoke-Api -BaseUrl $BaseUrl -Path '/api/cartridges/parameters?key=filter%3Areranker%2Fe2e-params&path=L2.o.gain'
  $ovUnmask = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/parameters/overlay' -Token $Token -Body (@{ key = $ovKey; action = 'unmask'; path = 'L2' } | ConvertTo-Json -Compress)
  $ovBack = Invoke-Api -BaseUrl $BaseUrl -Path '/api/cartridges/parameters?key=filter%3Areranker%2Fe2e-params&path=L2.o.gain'
  Assert-Check -Tier 1 -Area 'Overlay' -Name 'A mask sets a gain to 1 (masked) and unmask returns the edited value exactly' `
    -Condition ($ovMask.Status -eq 200 -and [double]@($ovMasked.Json.overlay)[0][0] -eq 1 -and @($ovMasked.Json.overlay)[0][1] -eq $true -and $ovUnmask.Status -eq 200 -and [double]@($ovBack.Json.overlay)[0][0] -eq 3) -Detail "mask=$($ovMask.Status) unmask=$($ovUnmask.Status)"
  $ovView = Invoke-Api -BaseUrl $BaseUrl -Path '/api/parameters/overlay?key=filter%3Areranker%2Fe2e-params'
  Assert-Check -Tier 1 -Area 'Overlay' -Name 'GET /api/parameters/overlay lists the entries with their regions and when a change takes effect' `
    -Condition ($ovView.Status -eq 200 -and @($ovView.Json.entries).Count -eq 2 -and $ovView.Json.counts.freezes -eq 1 -and (Has-Prop $ovView.Json 'takes_effect'))
  $ovClearHeld = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/parameters/overlay' -Token $Token -Body (@{ key = $ovKey; action = 'clear' } | ConvertTo-Json -Compress)
  Assert-Check -Tier 2 -Area 'Overlay' -Name 'Clearing an overlay that holds a freeze is refused at the preview: unfreezing is explicit (400)' -Condition ($ovClearHeld.Status -eq 400 -and $ovClearHeld.Body -like '*unfreeze*')
  $ovUnfreeze = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/parameters/overlay' -Token $Token -Body (@{ key = $ovKey; action = 'unfreeze'; path = 'L5' } | ConvertTo-Json -Compress)
  $ovClear1 = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/parameters/overlay' -Token $Token -Body (@{ key = $ovKey; action = 'clear' } | ConvertTo-Json -Compress)
  $ovClear2 = $null
  if ($ovClear1.Json.confirm_token) {
    $ovClear2 = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/parameters/overlay' -Token $Token -Body (@{ key = $ovKey; action = 'clear'; confirm_token = $ovClear1.Json.confirm_token } | ConvertTo-Json -Compress)
  }
  Assert-Check -Tier 1 -Area 'Overlay' -Name 'Clear goes through the delete gate: a preview with a token, then the same request with it' `
    -Condition ($ovUnfreeze.Status -eq 200 -and $ovClear1.Status -eq 200 -and $ovClear1.Json.deleted -eq $false -and $null -ne $ovClear2 -and $ovClear2.Status -eq 200) -Detail "unfreeze=$($ovUnfreeze.Status) first=$($ovClear1.Status)"
  if ($ovClear2 -and $ovClear2.Json.audit_id) {
    $ovUndo = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/undo' -Token $Token -Body (@{ id = $ovClear2.Json.audit_id } | ConvertTo-Json -Compress)
    $ovAfter = Invoke-Api -BaseUrl $BaseUrl -Path '/api/parameters/overlay?key=filter%3Areranker%2Fe2e-params'
    Assert-Check -Tier 1 -Area 'Overlay' -Name 'The audit undo of a clear puts the entries back' -Condition ($ovUndo.Status -eq 200 -and @($ovAfter.Json.entries).Count -eq 1)
  }
  $ovStruct = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/parameters/overlay' -Token $Token -Body (@{ key = $ovKey; action = 'edit'; op = 'drop'; path = 'L2.o.gain' } | ConvertTo-Json -Compress)
  Assert-Check -Tier 2 -Area 'Overlay' -Name 'A structural change is refused in the overlay and pointed to a new version (400)' -Condition ($ovStruct.Status -eq 400 -and $ovStruct.Body -like '*new version*')

  # Learning units and lenses.
  $pbUnits = Invoke-Api -BaseUrl $BaseUrl -Path '/api/units'
  Assert-Check -Tier 1 -Area 'Phase-B' -Name 'GET /api/units lists the registered units with the registry path' -Condition ($pbUnits.Status -eq 200 -and $pbUnits.JsonOk -and (Has-Prop $pbUnits.Json 'units') -and (Has-Prop $pbUnits.Json 'registry'))
  $pbUnreg = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/units' -Token $Token -Body '{"action":"unregister","id":"no-such-unit"}'
  Assert-Check -Tier 2 -Area 'Phase-B' -Name 'Unregistering an unknown unit is refused at the preview (400)' -Condition ($pbUnreg.Status -eq 400)
  $pbReg = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/units' -Token $Token -Body '{"action":"register","id":"e2e-unit"}'
  Assert-Check -Tier 2 -Area 'Phase-B' -Name 'Register from the interface server says it needs a serving process with the engine bound (400)' -Condition ($pbReg.Status -eq 400 -and $pbReg.Body -like '*serving process*')
  $pbLenses = Invoke-Api -BaseUrl $BaseUrl -Path '/api/lenses'
  Assert-Check -Tier 1 -Area 'Phase-B' -Name 'GET /api/lenses lists the project lenses' -Condition ($pbLenses.Status -eq 200 -and $pbLenses.JsonOk -and (Has-Prop $pbLenses.Json 'lenses'))
  $pbLensRm = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/lenses' -Token $Token -Body '{"action":"remove","id":"embedder-lens-0000000000000000"}'
  Assert-Check -Tier 2 -Area 'Phase-B' -Name 'Removing an unknown lens is refused at the preview (400)' -Condition ($pbLensRm.Status -eq 400)

  # Memory: a stored note (gate, never-save screen), pinned and the pin undone, a sweep; objectives.
  $pbMem = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/memory/add' -Token $Token -Body '{"project":"nano","kind":"store","text":"e2e: the release ships on the 27th"}'
  $pbMemId = $pbMem.Json.result.stored
  Assert-Check -Tier 1 -Area 'Phase-B' -Name 'POST /api/memory/add stores a note through the gate (applied now)' `
    -Condition ($pbMem.Status -eq 200 -and $pbMem.JsonOk -and $pbMem.Json.applied -eq 'now' -and $pbMem.Json.result.admitted -eq $true -and [string]$pbMemId -ne '')
  $pbItems = Invoke-Api -BaseUrl $BaseUrl -Path '/api/memory/items?project=nano&q=ships'
  Assert-Check -Tier 1 -Area 'Phase-B' -Name 'The stored note is in the memory list' -Condition ($pbItems.Status -eq 200 -and @($pbItems.Json.items | Where-Object { $_.id -eq $pbMemId }).Count -eq 1)
  $pbSecret = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/memory/add' -Token $Token -Body '{"project":"nano","kind":"store","text":"the key is sk-ABCdef1234567890abcdefGHIJ"}'
  Assert-Check -Tier 1 -Area 'Phase-B' -Name 'A note holding a secret is refused by the never-save screen' -Condition ($pbSecret.Status -eq 200 -and $pbSecret.Json.result.admitted -eq $false)
  $pbPin = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/memory/pin' -Token $Token -Body (@{ project = 'nano'; id = $pbMemId; pin = $true } | ConvertTo-Json -Compress)
  Assert-Check -Tier 1 -Area 'Phase-B' -Name 'POST /api/memory/pin pins the memory' -Condition ($pbPin.Status -eq 200 -and $pbPin.Json.result.class -eq 'pinned')
  $pbPinUndo = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/undo' -Token $Token -Body (@{ id = $pbPin.Json.audit_id } | ConvertTo-Json -Compress)
  Assert-Check -Tier 1 -Area 'Phase-B' -Name 'The audit undo unpins it' -Condition ($pbPinUndo.Status -eq 200)
  $pbTemp = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/memory/add' -Token $Token -Body '{"project":"nano","kind":"temp","text":"e2e temp note"}'
  $pbSweep = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/memory/sweep' -Token $Token -Body '{"project":"nano"}'
  Assert-Check -Tier 1 -Area 'Phase-B' -Name 'A temp note is kept and POST /api/memory/sweep sweeps' -Condition ($pbTemp.Status -eq 200 -and [string]$pbTemp.Json.result.temp_note -ne '' -and $pbSweep.Status -eq 200 -and (Has-Prop $pbSweep.Json.result 'kept'))
  $pbGoal = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/goals' -Token $Token -Body '{"project":"nano","objective":"e2e: finish phase B"}'
  $pbGoalId = @($pbGoal.Json.goals.objectives | Where-Object { $_.objective -eq 'e2e: finish phase B' })[0].id
  Assert-Check -Tier 1 -Area 'Phase-B' -Name 'POST /api/goals adds an open objective' -Condition ($pbGoal.Status -eq 200 -and $null -ne $pbGoalId -and [int]$pbGoal.Json.goals.open -ge 1)
  $pbDone = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/goals' -Token $Token -Body (@{ project = 'nano'; id = [int64]$pbGoalId; status = 'done' } | ConvertTo-Json -Compress)
  $pbDoneUndo = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/undo' -Token $Token -Body (@{ id = $pbDone.Json.audit_id } | ConvertTo-Json -Compress)
  $pbGoals = Invoke-Api -BaseUrl $BaseUrl -Path '/api/goals?project=nano'
  $pbGoalNow = @($pbGoals.Json.objectives | Where-Object { [string]$_.id -eq [string]$pbGoalId })[0].status
  Assert-Check -Tier 1 -Area 'Phase-B' -Name 'An objective closed is opened again by the audit undo' -Condition ($pbDone.Status -eq 200 -and $pbDoneUndo.Status -eq 200 -and $pbGoalNow -eq 'open')
  # Guidance (reminders, loops and the step graph for the coding agent, /api/guidance): audited
  # writes, the step graph's readiness and cycle refusal, the loop cap, the gated removal and its
  # undo, then the hooks delivering a reminder and a loop continuing the agent at a stop.
  $gdAdd = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/guidance' -Token $Token -Body '{"project":"nano","action":"add","kind":"reminder","text":"e2e: run the tests","trigger":"prompt","every":2,"repeat":"max","max":2}'
  $gdRid = [string]$gdAdd.Json.id
  Assert-Check -Tier 1 -Area 'Guidance' -Name 'POST /api/guidance adds a reminder, audited and set by the user' `
    -Condition ($gdAdd.Status -eq 200 -and $gdAdd.JsonOk -and $gdAdd.Json.kind -eq 'reminder' -and [string]$gdAdd.Json.audit_id -ne '' -and @($gdAdd.Json.guidance.reminders | Where-Object { $_.id -eq $gdRid -and $_.by -eq 'user' }).Count -eq 1) -Detail $gdAdd.Body
  $gdA = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/guidance' -Token $Token -Body '{"project":"nano","action":"add","kind":"step","id":"e2e-a","title":"e2e: write the parser"}'
  $gdB = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/guidance' -Token $Token -Body '{"project":"nano","action":"add","kind":"step","title":"e2e: test the parser","depends_on":["e2e-a"],"check":"test","paths":["src/**"]}'
  $gdBid = [string]$gdB.Json.id
  $gdCycle = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/guidance' -Token $Token -Body (@{ project = 'nano'; action = 'edit'; kind = 'step'; id = 'e2e-a'; depends_on = @($gdBid) } | ConvertTo-Json -Compress)
  $gdView = Invoke-Api -BaseUrl $BaseUrl -Path '/api/guidance?project=nano'
  $gdStA = @($gdView.Json.steps | Where-Object { $_.id -eq 'e2e-a' })[0].status
  $gdStB = @($gdView.Json.steps | Where-Object { $_.id -eq $gdBid })[0].status
  Assert-Check -Tier 1 -Area 'Guidance' -Name 'The step graph reports ready and blocked steps and refuses a cycle (400)' `
    -Condition ($gdA.Status -eq 200 -and $gdB.Status -eq 200 -and $gdCycle.Status -eq 400 -and $gdCycle.Body -like '*cycle*' -and $gdStA -eq 'ready' -and $gdStB -eq 'blocked' -and $gdView.Json.next_step -eq 'e2e-a') -Detail "a=$gdStA b=$gdStB cycle=$($gdCycle.Status) $($gdCycle.Body)"
  $gdCap = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/guidance' -Token $Token -Body '{"project":"nano","action":"add","kind":"loop","prompt":"e2e: keep going","until":"manual","max":21}'
  $gdLoop = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/guidance' -Token $Token -Body '{"project":"nano","action":"add","kind":"loop","prompt":"e2e: fix and test again.","until":"check-passes"}'
  $gdLid = [string]$gdLoop.Json.id
  $gdLoopNow = @($gdLoop.Json.guidance.loops | Where-Object { $_.id -eq $gdLid })[0]
  $gdEnd = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/guidance' -Token $Token -Body (@{ project = 'nano'; action = 'done'; kind = 'loop'; id = $gdLid } | ConvertTo-Json -Compress)
  $gdEnded = @($gdEnd.Json.guidance.loops | Where-Object { $_.id -eq $gdLid })[0].state
  Assert-Check -Tier 1 -Area 'Guidance' -Name 'Loops: a maximum over 20 is refused, a loop starts active at 0 of 3 iterations and is ended by hand' `
    -Condition ($gdCap.Status -eq 400 -and $gdLoop.Status -eq 200 -and $gdLoopNow.state -eq 'active' -and [int]$gdLoopNow.iterations -eq 0 -and [int]$gdLoopNow.max -eq 3 -and $gdEnded -eq 'ended') -Detail "cap=$($gdCap.Status) state=$($gdLoopNow.state) ended=$gdEnded"
  $gdDel = Invoke-Delete -BaseUrl $BaseUrl -Path '/api/guidance' -Token $Token -Body (@{ project = 'nano'; action = 'remove'; kind = 'reminder'; id = $gdRid } | ConvertTo-Json -Compress)
  $gdGone = @($gdDel.Result.Json.guidance.reminders | Where-Object { $_.id -eq $gdRid }).Count -eq 0
  $gdUndo = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/undo' -Token $Token -Body (@{ id = $gdDel.Result.Json.audit_id } | ConvertTo-Json -Compress)
  $gdBack = @((Invoke-Api -BaseUrl $BaseUrl -Path '/api/guidance?project=nano').Json.reminders | Where-Object { $_.id -eq $gdRid }).Count -eq 1
  Assert-Check -Tier 1 -Area 'Guidance' -Name 'Removing a reminder takes the preview and its token, and the audit undo puts it back' `
    -Condition ($gdDel.Preview.Json.step -eq 'preview' -and $gdDel.Preview.Json.confirm_word -eq $gdRid -and $gdDel.Result.Status -eq 200 -and $gdGone -and $gdUndo.Status -eq 200 -and $gdBack) -Detail "preview=$($gdDel.Preview.Json.step) gone=$gdGone undo=$($gdUndo.Status) back=$gdBack"
  $gdOnce = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/guidance' -Token $Token -Body '{"project":"nano","action":"add","kind":"loop","prompt":"e2e: one more pass.","until":"manual","max":1}'
  $gdPrevDir = $env:CLAUDE_PROJECT_DIR
  $env:CLAUDE_PROJECT_DIR = $TempHome
  $gdHook1 = ('{"hook_event_name":"UserPromptSubmit","session_id":"e2e-guidance","prompt_id":"p1","prompt":"go"}' | & $ResolvedExe hook UserPromptSubmit) -join "`n"
  $gdHook2 = ('{"hook_event_name":"UserPromptSubmit","session_id":"e2e-guidance","prompt_id":"p2","prompt":"again"}' | & $ResolvedExe hook UserPromptSubmit) -join "`n"
  $gdStop1 = ('{"hook_event_name":"Stop","session_id":"e2e-guidance","stop_hook_active":false}' | & $ResolvedExe hook Stop) -join "`n"
  $gdStop2 = ('{"hook_event_name":"Stop","session_id":"e2e-guidance","stop_hook_active":true}' | & $ResolvedExe hook Stop) -join "`n"
  if ($null -eq $gdPrevDir) { Remove-Item Env:CLAUDE_PROJECT_DIR -ErrorAction SilentlyContinue } else { $env:CLAUDE_PROJECT_DIR = $gdPrevDir }
  Assert-Check -Tier 1 -Area 'Guidance' -Name 'The hooks deliver: the next ready step at the 1st prompt, the every-2nd-prompt reminder at the 2nd, prefixed' `
    -Condition ($gdHook1 -match 'next ready step is e2e-a' -and $gdHook1 -notmatch 'e2e: run the tests' -and $gdHook2 -match 'Neuromorph Nano reminder \(set by the user\): e2e: run the tests') -Detail "1=$gdHook1 2=$gdHook2"
  Assert-Check -Tier 1 -Area 'Guidance' -Name 'A loop continues the agent at a stop (decision block) up to its maximum, then ends with a non-blocking note' `
    -Condition ($gdOnce.Status -eq 200 -and $gdStop1 -match '"decision":"block"' -and $gdStop1 -match 'e2e: one more pass' -and $gdStop2 -notmatch '"decision"' -and $gdStop2 -match 'stopped after 1 iteration') -Detail "stop1=$gdStop1 stop2=$gdStop2"
  $gdAfter = Invoke-Api -BaseUrl $BaseUrl -Path '/api/guidance?project=nano'
  $gdKinds = @($gdAfter.Json.deliveries | ForEach-Object { "$($_.kind):$($_.channel)" })
  Assert-Check -Tier 1 -Area 'Guidance' -Name 'GET /api/guidance lists what was sent to the agent (reminder, step, loop block and note) and the exhausted loop' `
    -Condition ($gdAfter.Status -eq 200 -and ($gdKinds -contains 'reminder:context') -and ($gdKinds -contains 'step:context') -and ($gdKinds -contains 'loop:block') -and ($gdKinds -contains 'loop:note') -and @($gdAfter.Json.loops | Where-Object { $_.id -eq $gdOnce.Json.id -and $_.state -eq 'exhausted' }).Count -eq 1) -Detail ($gdKinds -join ',')
  # 2026-11-26: combined conditions, the stall exit, budgets and the outcome rows of ended loops.
  $gdComb = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/guidance' -Token $Token -Body '{"project":"nano","action":"add","kind":"loop","prompt":"e2e: test and finish the steps.","all_of":["check-passes","steps-done"]}'
  $gdCombBad = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/guidance' -Token $Token -Body '{"project":"nano","action":"add","kind":"loop","prompt":"e2e: x","until":"any-of","parts":["check-passes"]}'
  $gdCid = [string]$gdComb.Json.id
  $gdCombNow = @($gdComb.Json.guidance.loops | Where-Object { $_.id -eq $gdCid })[0]
  $gdCombEnd = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/guidance' -Token $Token -Body (@{ project = 'nano'; action = 'done'; kind = 'loop'; id = $gdCid } | ConvertTo-Json -Compress)
  Assert-Check -Tier 1 -Area 'Guidance' -Name 'A loop takes all_of two conditions, names the parts not yet met, and refuses any-of with one part (400)' `
    -Condition ($gdComb.Status -eq 200 -and $gdCombNow.until -eq 'all-of' -and @($gdCombNow.parts).Count -eq 2 -and [string]$gdCombNow.now.detail -like '*all-of, not yet met*check-passes*' -and $gdCombBad.Status -eq 400 -and $gdCombBad.Body -like '*2 to 4*' -and $gdCombEnd.Status -eq 200) -Detail "add=$($gdComb.Status) until=$($gdCombNow.until) now=$($gdCombNow.now.detail) bad=$($gdCombBad.Status) $($gdCombBad.Body)"
  $gdStallBad = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/guidance' -Token $Token -Body '{"project":"nano","action":"add","kind":"loop","prompt":"e2e: x","until":"check-passes","stall":11}'
  $gdBudBad = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/guidance' -Token $Token -Body '{"project":"nano","action":"add","kind":"loop","prompt":"e2e: x","until":"check-passes","budget_ms":10}'
  $gdStall = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/guidance' -Token $Token -Body '{"project":"nano","action":"add","kind":"loop","prompt":"e2e: run the tests.","until":"check-passes","stall":1}'
  $env:CLAUDE_PROJECT_DIR = $TempHome
  $gdSt1 = ('{"hook_event_name":"Stop","session_id":"e2e-guidance-stall","stop_hook_active":false}' | & $ResolvedExe hook Stop) -join "`n"
  $gdSt2 = ('{"hook_event_name":"Stop","session_id":"e2e-guidance-stall","stop_hook_active":true}' | & $ResolvedExe hook Stop) -join "`n"
  $gdBud = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/guidance' -Token $Token -Body '{"project":"nano","action":"add","kind":"loop","prompt":"e2e: keep polishing.","until":"manual","max":5,"budget_ms":1000}'
  Start-Sleep -Milliseconds 1300
  $gdBu1 = ('{"hook_event_name":"Stop","session_id":"e2e-guidance-budget","stop_hook_active":false}' | & $ResolvedExe hook Stop) -join "`n"
  if ($null -eq $gdPrevDir) { Remove-Item Env:CLAUDE_PROJECT_DIR -ErrorAction SilentlyContinue } else { $env:CLAUDE_PROJECT_DIR = $gdPrevDir }
  Assert-Check -Tier 1 -Area 'Guidance' -Name 'Stall exit: a check-passes loop with stall 1 continues once, then ends as stalled with one note naming the reason; stall 11 is refused (400)' `
    -Condition ($gdStallBad.Status -eq 400 -and $gdStall.Status -eq 200 -and $gdSt1 -match '"decision":"block"' -and $gdSt2 -notmatch '"decision"' -and $gdSt2 -match 'ended as stalled' -and $gdSt2 -match 'no build or test has run') -Detail "bad=$($gdStallBad.Status) st1=$gdSt1 st2=$gdSt2"
  Assert-Check -Tier 1 -Area 'Guidance' -Name 'Budget: a loop with a 1 s budget ends as budget-spent at its first stop after it; a budget under 1000 ms is refused (400)' `
    -Condition ($gdBudBad.Status -eq 400 -and $gdBud.Status -eq 200 -and $gdBu1 -notmatch '"decision"' -and $gdBu1 -match 'ended as budget-spent' -and $gdBu1 -match 'time budget') -Detail "bad=$($gdBudBad.Status) bu1=$gdBu1"
  $gdOut = Invoke-Api -BaseUrl $BaseUrl -Path '/api/guidance?project=nano&outcomes=50'
  $gdEnds = @{}
  foreach ($r in @($gdOut.Json.outcomes)) { $gdEnds[[string]$r.id] = $r }
  $gdRowStall = $gdEnds[[string]$gdStall.Json.id]
  $gdRowBud = $gdEnds[[string]$gdBud.Json.id]
  $gdRowComb = $gdEnds[$gdCid]
  $gdRowOnce = $gdEnds[[string]$gdOnce.Json.id]
  Assert-Check -Tier 1 -Area 'Guidance' -Name 'GET /api/guidance lists one outcome row per ended loop (stalled, budget-spent, ended, cap) with its unmet reasons, budget used and verdict' `
    -Condition ($gdOut.Status -eq 200 -and [int]$gdOut.Json.outcomes_total -ge 4 -and $gdRowStall.end -eq 'stalled' -and @($gdRowStall.unmet).Count -eq 1 -and $gdRowStall.verdicts.state -eq 'refuted' -and $gdRowBud.end -eq 'budget-spent' -and [int]$gdRowBud.budget.used_ms -ge 1000 -and $gdRowComb.end -eq 'ended' -and $gdRowComb.condition.until -eq 'all-of' -and $gdRowOnce.end -eq 'cap') -Detail ((@($gdOut.Json.outcomes) | ForEach-Object { "$($_.id):$($_.end)" }) -join ',')
  # 2026-11-26: OR-joins (depends_any) and probe steps in the step graph.
  $gdOr = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/guidance' -Token $Token -Body (@{ project = 'nano'; action = 'add'; kind = 'step'; id = 'e2e-or'; title = 'e2e: after either'; depends_any = @('e2e-a', $gdBid) } | ConvertTo-Json -Compress)
  $gdOrBefore = @($gdOr.Json.guidance.steps | Where-Object { $_.id -eq 'e2e-or' })[0].status
  $gdOrA = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/guidance' -Token $Token -Body '{"project":"nano","action":"done","kind":"step","id":"e2e-a"}'
  $gdOrAfter = @($gdOrA.Json.guidance.steps | Where-Object { $_.id -eq 'e2e-or' })[0].status
  $gdOrCycle = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/guidance' -Token $Token -Body '{"project":"nano","action":"edit","kind":"step","id":"e2e-a","depends_any":["e2e-or"]}'
  Assert-Check -Tier 1 -Area 'Guidance' -Name 'An OR-join step (depends_any) is blocked until one of its steps is done, then ready; a cycle through an OR edge is refused (400)' `
    -Condition ($gdOr.Status -eq 200 -and $gdOrBefore -eq 'blocked' -and $gdOrA.Status -eq 200 -and $gdOrAfter -eq 'ready' -and $gdOrCycle.Status -eq 400 -and $gdOrCycle.Body -like '*cycle*') -Detail "add=$($gdOr.Status) before=$gdOrBefore after=$gdOrAfter cycle=$($gdOrCycle.Status) $($gdOrCycle.Body)"
  $gdProbeBad = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/guidance' -Token $Token -Body '{"project":"nano","action":"add","kind":"step","title":"e2e: x","probe":true}'
  $gdProbe = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/guidance' -Token $Token -Body '{"project":"nano","action":"add","kind":"step","id":"e2e-probe","title":"e2e: the tests already pass","depends_on":["e2e-or"],"check":"test","probe":true}'
  $gdProbeNow = @($gdProbe.Json.guidance.steps | Where-Object { $_.id -eq 'e2e-probe' })[0]
  Assert-Check -Tier 1 -Area 'Guidance' -Name 'A probe step needs a check (400 without one) and is kept as a probe' `
    -Condition ($gdProbeBad.Status -eq 400 -and $gdProbeBad.Body -like '*probe needs a check*' -and $gdProbe.Status -eq 200 -and $gdProbeNow.probe -eq $true -and $gdProbeNow.check -eq 'test' -and $gdProbeNow.status -eq 'blocked') -Detail "bad=$($gdProbeBad.Status) $($gdProbeBad.Body) probe=$($gdProbeNow.probe) status=$($gdProbeNow.status)"
  $pbRefresh = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/comprehension/refresh' -Token $Token -Body '{"project":"nano"}'
  Assert-Check -Tier 2 -Area 'Phase-B' -Name 'A world-model refresh with no notes is refused at the preview (400)' -Condition ($pbRefresh.Status -eq 400 -and $pbRefresh.Body -like '*no world-model notes*')
  $pbConn = Invoke-Api -BaseUrl $BaseUrl -Path ('/api/connect?project=' + [uri]::EscapeDataString($TempHome))
  $pbClients = @($pbConn.Json.clients | ForEach-Object { $_.client })
  Assert-Check -Tier 1 -Area 'Phase-B' -Name 'GET /api/connect lists the six clients with copyable commands, none connected in a fresh folder' `
    -Condition ($pbConn.Status -eq 200 -and $pbConn.JsonOk -and $pbClients.Count -eq 6 -and ($pbClients -contains 'antigravity') -and @($pbConn.Json.clients | Where-Object { $_.connected -eq $true }).Count -eq 0 -and [string]$pbConn.Json.clients[0].command -like 'neuromorph-nano connect claude --project *')

  # POST /api/ledger/clear
  $rLedgerClear = (Invoke-Delete -BaseUrl $BaseUrl -Path '/api/ledger/clear' -Token $Token -Body '{"project":"nano"}').Result
  Assert-Check -Tier 1 -Area 'HTTP-POST' -Name 'POST /api/ledger/clear (200, project=nano)' `
    -Condition ($rLedgerClear.Status -eq 200 -and $rLedgerClear.JsonOk -and $rLedgerClear.Json.project -eq 'nano' -and (Has-Prop $rLedgerClear.Json 'audit_id'))

  # POST /api/examples/clear
  $rExamplesClear = (Invoke-Delete -BaseUrl $BaseUrl -Path '/api/examples/clear' -Token $Token -Body '{"project":"nano"}').Result
  Assert-Check -Tier 1 -Area 'HTTP-POST' -Name 'POST /api/examples/clear (200, project=nano)' `
    -Condition ($rExamplesClear.Status -eq 200 -and $rExamplesClear.JsonOk -and $rExamplesClear.Json.project -eq 'nano' -and (Has-Prop $rExamplesClear.Json 'audit_id'))

  # POST /api/dense/clear
  $rDenseClear = (Invoke-Delete -BaseUrl $BaseUrl -Path '/api/dense/clear' -Token $Token -Body '{"project":"nano"}').Result
  Assert-Check -Tier 1 -Area 'HTTP-POST' -Name 'POST /api/dense/clear (200, project=nano)' `
    -Condition ($rDenseClear.Status -eq 200 -and $rDenseClear.JsonOk -and $rDenseClear.Json.project -eq 'nano' -and (Has-Prop $rDenseClear.Json 'audit_id'))

  # POST /api/learning/cycle (roadmap Phase 2, 2026-11-24: one unified cycle, each learner promoted
  # only when its held-out score beats the current one, so the version moves only on a promotion;
  # a rollback with no version undoes the newest cycle across every learner it touched)
  $rLearnCycle = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/learning/cycle' -Token $Token -Body '{"project":"nano","action":"cycle"}'
  Assert-Check -Tier 1 -Area 'HTTP-POST' -Name 'POST /api/learning/cycle action=cycle (200, a cycle record with its audit id)' `
    -Condition ($rLearnCycle.Status -eq 200 -and $rLearnCycle.JsonOk -and (Has-Prop $rLearnCycle.Json 'cycle') -and ([int]$rLearnCycle.Json.version -ge 1) -and (Has-Prop $rLearnCycle.Json 'audit_id'))

  $rLearnRollback = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/learning/cycle' -Token $Token -Body '{"project":"nano","action":"rollback"}'
  Assert-Check -Tier 1 -Area 'HTTP-POST' -Name 'POST /api/learning/cycle action=rollback (200, the newest cycle undone with its audit id)' `
    -Condition ($rLearnRollback.Status -eq 200 -and $rLearnRollback.JsonOk -and (Has-Prop $rLearnRollback.Json 'audit_id'))

  # Budget, quota, and tool count integer controls specification
  $BudgetSpecs = @(
    @{ key = 'budget.pack_tokens';     min = 1000; max = 16000;   step = 250;   unit = 'tokens'; default = '4000';  validTest = '2500' },
    @{ key = 'budget.tentacle_tokens'; min = 2000; max = 32000;   step = 500;   unit = 'tokens'; default = '12000'; validTest = '8500' },
    @{ key = 'budget.judge_window';    min = 128;  max = 2048;    step = 64;    unit = 'tokens'; default = '384';   validTest = '512'  },
    @{ key = 'budget.fast_k';          min = 1;    max = 16;      step = 1;     unit = 'items';  default = '4';     validTest = '7'    },
    @{ key = 'budget.max_run_s';       min = 5;    max = 180;     step = 5;     unit = 's';      default = '60';    validTest = '45'   },
    @{ key = 'budget.gpu_mb_max';      min = 1024; max = 81920;   step = 128;   unit = 'MiB';    default = 'auto';  validTest = '3072' },
    @{ key = 'budget.focused_tokens';  min = 500;  max = 8000;    step = 250;   unit = 'tokens'; default = '1800';  validTest = '2250' },
    @{ key = 'quota.session_tokens';   min = 0;    max = 500000;  step = 5000;  unit = 'tokens'; default = '0';     validTest = '75000' },
    @{ key = 'quota.daily_tokens';     min = 0;    max = 2000000; step = 25000; unit = 'tokens'; default = '0';     validTest = '250000' },
    @{ key = 'quota.session_packs';    min = 0;    max = 500;     step = 5;     unit = 'packs';  default = '0';     validTest = '50' },
    @{ key = 'mcp.tool_count';         min = 1;    max = 25;      step = 1;     unit = 'tools';  default = '25';    validTest = '5' },
    @{ key = 'prompts.days';           min = 1;    max = 3650;    step = 1;     unit = 'days';   default = '30';    validTest = '14' }
  )

  if ($StrictMode) {
    $controlsLatest = @((Invoke-Api -BaseUrl $BaseUrl -Path '/api/controls').Json)
    foreach ($spec in $BudgetSpecs) {
      $c = $controlsLatest | Where-Object { $_.key -eq $spec.key } | Select-Object -First 1
      $schemaOk = ($null -ne $c) -and
        ($c.kind -eq 'integer') -and
        ([int]$c.min -eq $spec.min) -and
        ([int]$c.max -eq $spec.max) -and
        ([int]$c.step -eq $spec.step) -and
        ($c.unit -eq $spec.unit) -and
        ($c.default -eq $spec.default)
      Assert-Check -Tier 1 -Area 'Budget-Schema' -Name "Control $($spec.key) metadata matches Contract 1" `
        -Condition $schemaOk `
        -Detail $(if ($null -eq $c) { 'missing from /api/controls' } else { "kind=$($c.kind) min=$($c.min) max=$($c.max) step=$($c.step) unit=$($c.unit)" })

      $rSetB = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/controls' -Token $Token `
        -Body (@{ key = $spec.key; value = $spec.validTest } | ConvertTo-Json -Compress)
      $bAudit = if ($rSetB.JsonOk) { $rSetB.Json.audit_id } else { $null }
      $bVal = if ($rSetB.JsonOk) { ($rSetB.Json.controls | Where-Object { $_.key -eq $spec.key } | Select-Object -First 1).value } else { $null }
      Assert-Check -Tier 1 -Area 'Budget-POST' -Name "POST /api/controls $($spec.key)=$($spec.validTest) (200)" `
        -Condition ($rSetB.Status -eq 200 -and $bVal -eq $spec.validTest)
      if ($bAudit) {
        $null = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/undo' -Token $Token -Body (@{ id = $bAudit } | ConvertTo-Json -Compress)
      }
    }
  } else {
    foreach ($spec in $BudgetSpecs) {
      Record-Check -Tier 1 -Area 'Budget-Schema' -Name "Control $($spec.key) metadata matches Contract 1" -Status 'SKIP' -Detail 'BaselineSmoke'
      Record-Check -Tier 1 -Area 'Budget-POST' -Name "POST /api/controls $($spec.key)=$($spec.validTest) (200)" -Status 'SKIP' -Detail 'BaselineSmoke'
    }
  }

  # =========================================================================
  # TIER 2: BOUNDARY & ERROR CASES
  # =========================================================================
  $validCtrlBody = '{"key":"mode","value":"fast"}'

  $rNoTok = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/controls' -Body $validCtrlBody
  Assert-Check -Tier 2 -Area 'Auth-Error' -Name 'Missing X-Nano-Token -> HTTP 403' -Condition ($rNoTok.Status -eq 403) -Detail "status=$($rNoTok.Status)"

  $rBadTok = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/controls' -Token ('0' * 64) -Body $validCtrlBody
  Assert-Check -Tier 2 -Area 'Auth-Error' -Name 'Invalid X-Nano-Token -> HTTP 403' -Condition ($rBadTok.Status -eq 403) -Detail "status=$($rBadTok.Status)"

  $rEvilOrigin = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/controls' -Token $Token -Origin 'http://evil.example' -Body $validCtrlBody
  Assert-Check -Tier 2 -Area 'Auth-Error' -Name 'Foreign Origin http://evil.example -> HTTP 403' -Condition ($rEvilOrigin.Status -eq 403) -Detail "status=$($rEvilOrigin.Status)"

  $rTextPlain = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/controls' -Token $Token -ContentType 'text/plain' -Body $validCtrlBody
  Assert-Check -Tier 2 -Area 'Auth-Error' -Name 'Non-JSON Content-Type text/plain -> HTTP 403' -Condition ($rTextPlain.Status -eq 403) -Detail "status=$($rTextPlain.Status)"

  $rNoCtype = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/controls' -Token $Token -OmitContentType -Body $validCtrlBody
  Assert-Check -Tier 2 -Area 'Auth-Error' -Name 'Missing Content-Type -> HTTP 403' -Condition ($rNoCtype.Status -eq 403) -Detail "status=$($rNoCtype.Status)"

  $rDevToken = Invoke-Api -BaseUrl $BaseUrl -Path '/api/token'
  Assert-Check -Tier 2 -Area 'Auth-Error' -Name 'GET /api/token without --dev -> HTTP 404' -Condition ($rDevToken.Status -eq 404) -Detail "status=$($rDevToken.Status)"

  $rBadJson = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/controls' -Token $Token -Body '{malformed_json'
  Assert-Check -Tier 2 -Area 'Validation' -Name 'Malformed JSON body -> HTTP 400 (error)' `
    -Condition ($rBadJson.Status -eq 400 -and $rBadJson.JsonOk -and (Has-Prop $rBadJson.Json 'error')) -Detail "status=$($rBadJson.Status)"

  $rUnkKey = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/controls' -Token $Token -Body '{"key":"unknown.control.xyz","value":"1"}'
  Assert-Check -Tier 2 -Area 'Validation' -Name 'Unknown control key -> HTTP 400 (error)' `
    -Condition ($rUnkKey.Status -eq 400 -and $rUnkKey.JsonOk -and (Has-Prop $rUnkKey.Json 'error')) -Detail "status=$($rUnkKey.Status)"

  $rBadDiscrete = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/controls' -Token $Token -Body '{"key":"mode","value":"invalid_mode_xyz"}'
  Assert-Check -Tier 2 -Area 'Validation' -Name 'Invalid discrete control value -> HTTP 400 (error)' `
    -Condition ($rBadDiscrete.Status -eq 400 -and $rBadDiscrete.JsonOk -and (Has-Prop $rBadDiscrete.Json 'error')) -Detail "status=$($rBadDiscrete.Status)"

  $rUnkGet = Invoke-Api -BaseUrl $BaseUrl -Path '/api/nonexistent'
  Assert-Check -Tier 2 -Area 'Routing' -Name 'Unknown GET /api/nonexistent -> HTTP 404' `
    -Condition ($rUnkGet.Status -eq 404 -and $rUnkGet.JsonOk -and (Has-Prop $rUnkGet.Json 'error')) -Detail "status=$($rUnkGet.Status)"

  $rUnkPost = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/nonexistent' -Token $Token -Body '{}'
  Assert-Check -Tier 2 -Area 'Routing' -Name 'Unknown POST /api/nonexistent -> HTTP 400' `
    -Condition ($rUnkPost.Status -eq 400 -and $rUnkPost.JsonOk -and (Has-Prop $rUnkPost.Json 'error')) -Detail "status=$($rUnkPost.Status)"

  $rRevokeUnk = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/devices/revoke' -Token $Token -Body '{"id":"nonexistent-device-id"}'
  Assert-Check -Tier 2 -Area 'Validation' -Name 'POST /api/devices/revoke unknown id -> HTTP 400' `
    -Condition ($rRevokeUnk.Status -eq 400 -and $rRevokeUnk.JsonOk -and (Has-Prop $rRevokeUnk.Json 'error')) -Detail "status=$($rRevokeUnk.Status)"

  $rUndoUnk = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/undo' -Token $Token -Body '{"id":"a-nonexistent-999"}'
  Assert-Check -Tier 2 -Area 'Validation' -Name 'POST /api/undo unknown audit id -> HTTP 400' `
    -Condition ($rUndoUnk.Status -eq 400 -and $rUndoUnk.JsonOk -and (Has-Prop $rUndoUnk.Json 'error')) -Detail "status=$($rUndoUnk.Status)"

  $rLedgerUnk = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/ledger/clear' -Token $Token -Body '{"project":"no-such-proj-999"}'
  Assert-Check -Tier 2 -Area 'Validation' -Name 'POST /api/ledger/clear unknown project -> HTTP 400' `
    -Condition ($rLedgerUnk.Status -eq 400 -and $rLedgerUnk.JsonOk -and (Has-Prop $rLedgerUnk.Json 'error')) -Detail "status=$($rLedgerUnk.Status)"

  $rExUnk = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/examples/clear' -Token $Token -Body '{"project":"no-such-proj-999"}'
  Assert-Check -Tier 2 -Area 'Validation' -Name 'POST /api/examples/clear unknown project -> HTTP 400' `
    -Condition ($rExUnk.Status -eq 400 -and $rExUnk.JsonOk -and (Has-Prop $rExUnk.Json 'error')) -Detail "status=$($rExUnk.Status)"

  foreach ($pEp in @('/api/session', '/api/ledger', '/api/outcomes', '/api/examples')) {
    $rMissingProj = Invoke-Api -BaseUrl $BaseUrl -Path $pEp
    Assert-Check -Tier 2 -Area 'Validation' -Name "GET $pEp missing ?project= -> JSON error" `
      -Condition ($rMissingProj.Status -eq 200 -and $rMissingProj.JsonOk -and (Has-Prop $rMissingProj.Json 'error')) -Detail "status=$($rMissingProj.Status)"
  }

  # Boundary Value Analysis on all 6 budget.* controls (full M5 mode)
  if ($StrictMode) {
    foreach ($spec in $BudgetSpecs) {
      $k = $spec.key
      $minStr = [string]$spec.min
      $maxStr = [string]$spec.max
      $belowStr = [string]($spec.min - 1)
      $aboveStr = [string]($spec.max + 1)

      $rMin = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/controls' -Token $Token -Body (@{ key = $k; value = $minStr } | ConvertTo-Json -Compress)
      Assert-Check -Tier 2 -Area 'Budget-BVA' -Name "$k min ($minStr) accepted -> 200" -Condition ($rMin.Status -eq 200) -Detail "status=$($rMin.Status)"
      if ($rMin.JsonOk -and $rMin.Json.audit_id) {
        $null = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/undo' -Token $Token -Body (@{ id = $rMin.Json.audit_id } | ConvertTo-Json -Compress)
      }

      $rMax = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/controls' -Token $Token -Body (@{ key = $k; value = $maxStr } | ConvertTo-Json -Compress)
      Assert-Check -Tier 2 -Area 'Budget-BVA' -Name "$k max ($maxStr) accepted -> 200" -Condition ($rMax.Status -eq 200) -Detail "status=$($rMax.Status)"
      if ($rMax.JsonOk -and $rMax.Json.audit_id) {
        $null = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/undo' -Token $Token -Body (@{ id = $rMax.Json.audit_id } | ConvertTo-Json -Compress)
      }

      $rBelow = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/controls' -Token $Token -Body (@{ key = $k; value = $belowStr } | ConvertTo-Json -Compress)
      Assert-Check -Tier 2 -Area 'Budget-BVA' -Name "$k min-1 ($belowStr) rejected -> 400" `
        -Condition ($rBelow.Status -eq 400 -and $rBelow.JsonOk -and (Has-Prop $rBelow.Json 'error')) -Detail "status=$($rBelow.Status)"

      $rAbove = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/controls' -Token $Token -Body (@{ key = $k; value = $aboveStr } | ConvertTo-Json -Compress)
      Assert-Check -Tier 2 -Area 'Budget-BVA' -Name "$k max+1 ($aboveStr) rejected -> 400" `
        -Condition ($rAbove.Status -eq 400 -and $rAbove.JsonOk -and (Has-Prop $rAbove.Json 'error')) -Detail "status=$($rAbove.Status)"

      $rNonInt = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/controls' -Token $Token -Body (@{ key = $k; value = 'abc' } | ConvertTo-Json -Compress)
      Assert-Check -Tier 2 -Area 'Budget-BVA' -Name "$k non-integer 'abc' rejected -> 400" `
        -Condition ($rNonInt.Status -eq 400 -and $rNonInt.JsonOk -and (Has-Prop $rNonInt.Json 'error')) -Detail "status=$($rNonInt.Status)"

      $rFloat = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/controls' -Token $Token -Body (@{ key = $k; value = '12.5' } | ConvertTo-Json -Compress)
      Assert-Check -Tier 2 -Area 'Budget-BVA' -Name "$k float '12.5' rejected -> 400" `
        -Condition ($rFloat.Status -eq 400 -and $rFloat.JsonOk -and (Has-Prop $rFloat.Json 'error')) -Detail "status=$($rFloat.Status)"
    }
  } else {
    foreach ($spec in $BudgetSpecs) {
      $k = $spec.key
      foreach ($sub in @('min', 'max', 'min-1', 'max+1', 'non-integer', 'float')) {
        Record-Check -Tier 2 -Area 'Budget-BVA' -Name "$k $sub check" -Status 'SKIP' -Detail 'BaselineSmoke'
      }
    }
  }

  # =========================================================================
  # TIER 3: CROSS-FEATURE COMBINATIONS
  # =========================================================================
  $settingsPath = Join-Path $TempHome 'settings.json'

  # 3.1 Discrete control roundtrip across API + settings.json + audit + undo
  $c1Before = (@((Invoke-Api -BaseUrl $BaseUrl -Path '/api/controls').Json) | Where-Object { $_.key -eq 'priming' } | Select-Object -First 1).value
  $c1Target = if ($c1Before -eq 'off') { 'on' } else { 'off' }
  $rSet1 = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/controls' -Token $Token -Body (@{ key = 'priming'; value = $c1Target } | ConvertTo-Json -Compress)
  $id1 = $rSet1.Json.audit_id
  $ctrl1 = (@((Invoke-Api -BaseUrl $BaseUrl -Path '/api/controls').Json) | Where-Object { $_.key -eq 'priming' } | Select-Object -First 1).value
  $settings1 = if (Test-Path $settingsPath) { (Get-Content $settingsPath -Raw | ConvertFrom-Json).priming } else { $null }
  $audit1 = @((Invoke-Api -BaseUrl $BaseUrl -Path '/api/audit?limit=50').Json.audit) | Where-Object { $_.id -eq $id1 } | Select-Object -First 1
  Assert-Check -Tier 3 -Area 'Persistence' -Name 'Discrete control POST -> /api/controls + settings.json + /api/audit' `
    -Condition ($rSet1.Status -eq 200 -and $ctrl1 -eq $c1Target -and $settings1 -eq $c1Target -and $null -ne $audit1 -and $audit1.to -eq $c1Target)

  $rUndo1 = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/undo' -Token $Token -Body (@{ id = $id1 } | ConvertTo-Json -Compress)
  $ctrl1After = (@((Invoke-Api -BaseUrl $BaseUrl -Path '/api/controls').Json) | Where-Object { $_.key -eq 'priming' } | Select-Object -First 1).value
  $settings1After = if (Test-Path $settingsPath) { (Get-Content $settingsPath -Raw | ConvertFrom-Json).priming } else { $null }
  $auditUndo1 = @((Invoke-Api -BaseUrl $BaseUrl -Path '/api/audit?limit=50').Json.audit) | Where-Object { $_.undo_of -eq $id1 } | Select-Object -First 1
  Assert-Check -Tier 3 -Area 'Persistence' -Name 'Discrete control undo -> /api/controls + settings.json + /api/audit restored' `
    -Condition ($rUndo1.Status -eq 200 -and $ctrl1After -eq $c1Before -and $settings1After -eq $c1Before -and $null -ne $auditUndo1)

  # 3.2 Budget pack_tokens roundtrip across API + settings.json + audit + undo (Full M5 mode)
  if ($StrictMode) {
    $rSetPack = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/controls' -Token $Token -Body '{"key":"budget.pack_tokens","value":"2500"}'
    $packAuditId = if ($rSetPack.JsonOk) { $rSetPack.Json.audit_id } else { $null }
    $packCtrl = (@((Invoke-Api -BaseUrl $BaseUrl -Path '/api/controls').Json) | Where-Object { $_.key -eq 'budget.pack_tokens' } | Select-Object -First 1).value
    $packSettings = if (Test-Path $settingsPath) { (Get-Content $settingsPath -Raw | ConvertFrom-Json).'budget.pack_tokens' } else { $null }
    $packAuditRow = @((Invoke-Api -BaseUrl $BaseUrl -Path '/api/audit?limit=50').Json.audit) | Where-Object { $_.id -eq $packAuditId } | Select-Object -First 1
    Assert-Check -Tier 3 -Area 'Budget-Roundtrip' -Name 'budget.pack_tokens=2500 -> /api/controls + settings.json + /api/audit' `
      -Condition ($rSetPack.Status -eq 200 -and $packCtrl -eq '2500' -and $packSettings -eq '2500' -and $null -ne $packAuditRow)

    $rUndoPack = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/undo' -Token $Token -Body (@{ id = $packAuditId } | ConvertTo-Json -Compress)
    $packCtrlAfter = (@((Invoke-Api -BaseUrl $BaseUrl -Path '/api/controls').Json) | Where-Object { $_.key -eq 'budget.pack_tokens' } | Select-Object -First 1).value
    $packSettingsAfter = if (Test-Path $settingsPath) { (Get-Content $settingsPath -Raw | ConvertFrom-Json).'budget.pack_tokens' } else { $null }
    Assert-Check -Tier 3 -Area 'Budget-Roundtrip' -Name 'budget.pack_tokens undo -> restored to 4000 across API & settings.json' `
      -Condition ($rUndoPack.Status -eq 200 -and $packCtrlAfter -eq '4000' -and $packSettingsAfter -eq '4000')
  } else {
    Record-Check -Tier 3 -Area 'Budget-Roundtrip' -Name 'budget.pack_tokens=2500 -> /api/controls + settings.json + /api/audit' -Status 'SKIP' -Detail 'BaselineSmoke'
    Record-Check -Tier 3 -Area 'Budget-Roundtrip' -Name 'budget.pack_tokens undo -> restored to 4000 across API & settings.json' -Status 'SKIP' -Detail 'BaselineSmoke'
  }

  # 3.3 Pair code -> GET /api/devices reflects code_open && code_expires_ms > 0
  $null = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/pair/code' -Token $Token -Body '{}'
  $devAfter = (Invoke-Api -BaseUrl $BaseUrl -Path '/api/devices').Json
  Assert-Check -Tier 3 -Area 'Pairing' -Name 'POST /api/pair/code -> GET /api/devices code_open & code_expires_ms > 0' `
    -Condition ($devAfter.code_open -eq $true -and [int64]$devAfter.code_expires_ms -gt 0)

  # 3.4 Clear ledger & examples -> GET /api/audit records privacy actions
  $rClearL = (Invoke-Delete -BaseUrl $BaseUrl -Path '/api/ledger/clear' -Token $Token -Body '{"project":"nano"}').Result
  $rClearE = (Invoke-Delete -BaseUrl $BaseUrl -Path '/api/examples/clear' -Token $Token -Body '{"project":"nano"}').Result
  $auditRows = @((Invoke-Api -BaseUrl $BaseUrl -Path '/api/audit?limit=100').Json.audit)
  $foundL = $auditRows | Where-Object { $_.id -eq $rClearL.Json.audit_id } | Select-Object -First 1
  $foundE = $auditRows | Where-Object { $_.id -eq $rClearE.Json.audit_id } | Select-Object -First 1
  Assert-Check -Tier 3 -Area 'Clear-Audit' -Name 'POST /api/ledger/clear & examples/clear -> recorded in /api/audit' `
    -Condition ($null -ne $foundL -and $null -ne $foundE)

  # 3.5 No-op control update returns audit_id == null
  $rNoop = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/controls' -Token $Token -Body (@{ key = 'mode'; value = $origMode } | ConvertTo-Json -Compress)
  Assert-Check -Tier 3 -Area 'Persistence' -Name 'POST /api/controls no-op update -> audit_id is null' `
    -Condition ($rNoop.Status -eq 200 -and $rNoop.JsonOk -and $null -eq $rNoop.Json.audit_id)

  # 3.6 Cartridges & library alignment
  $carts = @((Invoke-Api -BaseUrl $BaseUrl -Path '/api/cartridges').Json)
  $lib = (Invoke-Api -BaseUrl $BaseUrl -Path '/api/cartridges/library').Json
  Assert-Check -Tier 3 -Area 'Cartridges' -Name 'Cartridges list aligns with /api/cartridges/library' `
    -Condition ($carts.Count -ge 0 -and (Has-Prop $lib 'exists'))

  # =========================================================================
  # TIER 4: REAL-WORLD MCP + BUDGET SCENARIOS (Sequential launch serve)
  # =========================================================================
  $McpRepoRoot = Join-Path $Root 'tests\dev-large\material-code'
  if (-not (Test-Path $McpRepoRoot)) { $McpRepoRoot = $Root }
  Write-Host "Tier 4: Starting Session A (launch serve default --flow on $McpRepoRoot)..."
  $McpSession = Start-McpSession -LaunchExe $LaunchExe -RepoRoot $McpRepoRoot -TempHome $TempHome -ModelsDir $ModelsDir

  $initReply = Invoke-McpRequest -Session $McpSession -Method 'initialize' -Params @{
    protocolVersion = '2025-06-18'
    capabilities    = @{}
    clientInfo      = @{ name = 'e2e-verify'; version = '1.0' }
  } -TimeoutSeconds $McpTimeoutSeconds
  Assert-Check -Tier 4 -Area 'MCP-Flow' -Name 'initialize returns serverInfo neuromorph-nano' `
    -Condition ($initReply.result.serverInfo.name -eq 'neuromorph-nano')

  Send-McpNotification -Session $McpSession -Method 'notifications/initialized'

  $toolsReply = Invoke-McpRequest -Session $McpSession -Method 'tools/list' -Params @{} -TimeoutSeconds $McpTimeoutSeconds
  $toolNames = @($toolsReply.result.tools | ForEach-Object { $_.name })
  Assert-Check -Tier 4 -Area 'MCP-Flow' -Name 'tools/list contains nm_context_pack and >= 19 tools' `
    -Condition ($toolNames -contains 'nm_context_pack' -and $toolNames.Count -ge 19)
  # 2026-11-27: every tool the server's instructions name is listed by default (Claude Code calls
  # only listed tools; nm_settings and nm_cartridges were left out at the old default of 19).
  $instrText = [string]$initReply.result.instructions
  $namedTools = @([regex]::Matches($instrText, 'nm_[a-z_]+') | ForEach-Object { $_.Value } | Sort-Object -Unique)
  $unlisted = @($namedTools | Where-Object { $toolNames -notcontains $_ })
  Assert-Check -Tier 4 -Area 'MCP-Flow' -Name 'every tool named in the server instructions is in tools/list (nm_settings, nm_cartridges included)' `
    -Condition ($namedTools.Count -ge 19 -and $unlisted.Count -eq 0 -and $toolNames -contains 'nm_settings' -and $toolNames -contains 'nm_cartridges') `
    -Detail ("named {0}, unlisted: {1}" -f $namedTools.Count, ($unlisted -join ','))

  # 4a: Real codebase query (Call 1)
  $sw1 = [System.Diagnostics.Stopwatch]::StartNew()
  $call1Reply = Invoke-McpRequest -Session $McpSession -Method 'tools/call' -Params @{
    name      = 'nm_context_pack'
    arguments = @{
      question = 'Where is write_allowed implemented in crates/nmcore and what status code does it return?'
      format   = 'json'
      form     = 'full'
    }
  } -TimeoutSeconds $McpTimeoutSeconds
  $sw1.Stop()
  $call1Ms = [int]$sw1.ElapsedMilliseconds
  $pack1 = $call1Reply.result.content[0].text | ConvertFrom-Json
  $allCited1 = ($pack1.packets.Count -gt 0) -and ((@($pack1.packets | Where-Object { -not [string]::IsNullOrWhiteSpace($_.source) })).Count -eq $pack1.packets.Count)
  Assert-Check -Tier 4 -Area 'MCP-Flow' -Name 'Real codebase query returns non-withheld pack with .source citations and tokens > 0' `
    -Condition (($call1Reply.result.isError -ne $true) -and ($pack1.withheld -eq $false) -and $allCited1 -and ([int]$pack1.tokens -gt 0)) `
    -Detail "call1=${call1Ms}ms packets=$($pack1.packets.Count) tokens=$($pack1.tokens)"

  # 4c: Warm second call latency check on same serve session
  $sw2 = [System.Diagnostics.Stopwatch]::StartNew()
  $call2Reply = Invoke-McpRequest -Session $McpSession -Method 'tools/call' -Params @{
    name      = 'nm_context_pack'
    arguments = @{
      question = 'Where is find_root implemented in crates/nmtype?'
      format   = 'json'
      form     = 'full'
    }
  } -TimeoutSeconds $McpTimeoutSeconds
  $sw2.Stop()
  $call2Ms = [int]$sw2.ElapsedMilliseconds
  $pack2 = $call2Reply.result.content[0].text | ConvertFrom-Json
  Assert-Check -Tier 4 -Area 'MCP-Latency' -Name 'Warm Call 2 succeeds with lower/warm latency vs Call 1' `
    -Condition (($call2Reply.result.isError -ne $true) -and ($pack2.withheld -eq $false) -and (($call2Ms -lt $call1Ms) -or ($call2Ms -le 5000))) `
    -Detail "call1=${call1Ms}ms call2=${call2Ms}ms"

  # 4b: dec08 decoy query under default --flow
  $decoyFlowReply = Invoke-McpRequest -Session $McpSession -Method 'tools/call' -Params @{
    name      = 'nm_context_pack'
    arguments = @{
      question = 'How do you configure content keys to use BLAKE3 instead of SHA-1?'
      format   = 'json'
      form     = 'full'
    }
  } -TimeoutSeconds $McpTimeoutSeconds
  $decoyFlowPack = $decoyFlowReply.result.content[0].text | ConvertFrom-Json
  if ($StrictMode) {
    Assert-Check -Tier 4 -Area 'MCP-Decoy' -Name 'dec08 decoy query withheld=true & packets=[] under default --flow' `
      -Condition (($decoyFlowReply.result.isError -ne $true) -and ($decoyFlowPack.withheld -eq $true) -and (@($decoyFlowPack.packets).Count -eq 0)) `
      -Detail "withheld=$($decoyFlowPack.withheld) packets=$(@($decoyFlowPack.packets).Count) reason=$($decoyFlowPack.reason)"
  } else {
    Record-Check -Tier 4 -Area 'MCP-Decoy' -Name 'dec08 decoy query withheld=true & packets=[] under default --flow' -Status 'SKIP' `
      -Detail "BaselineSmoke (pre-M1 F2: withheld=$($decoyFlowPack.withheld) packets=$(@($decoyFlowPack.packets).Count))"
  }

  # 2026-11-26 (Phase B): the same features over MCP, and a memory write queued for the serving
  # process that holds the project's memory journal (applied by it within seconds).
  $pbMcpIns = Invoke-McpRequest -Session $McpSession -Method 'tools/call' -Params @{ name = 'nm_cartridges'; arguments = @{ op = 'inspect'; key = 'gtx-1650-4gb' } } -TimeoutSeconds $McpTimeoutSeconds
  $pbMcpMap = Invoke-McpRequest -Session $McpSession -Method 'tools/call' -Params @{ name = 'nm_cartridges'; arguments = @{ op = 'map_read'; role = 'reranker'; file = 'tol-e2e.json'; layer = '0' } } -TimeoutSeconds $McpTimeoutSeconds
  $pbMcpConn = Invoke-McpRequest -Session $McpSession -Method 'tools/call' -Params @{ name = 'nm_settings'; arguments = @{ op = 'connect'; project = $McpRepoRoot } } -TimeoutSeconds $McpTimeoutSeconds
  $cpMcp = Invoke-McpRequest -Session $McpSession -Method 'tools/call' -Params @{ name = 'nm_cartridges'; arguments = @{ op = 'parameters'; key = 'filter:reranker/e2e-params' } } -TimeoutSeconds $McpTimeoutSeconds
  $cpMcpBad = Invoke-McpRequest -Session $McpSession -Method 'tools/call' -Params @{ name = 'nm_cartridges'; arguments = @{ op = 'parameters_edit'; key = 'filter:reranker/e2e-params'; values = @{ 'L2.o.gain' = @(1, 2) } } } -TimeoutSeconds $McpTimeoutSeconds
  $cpMcpPrev = Invoke-McpRequest -Session $McpSession -Method 'tools/call' -Params @{ name = 'nm_cartridges'; arguments = @{ op = 'parameters_edit'; key = 'filter:reranker/e2e-params'; edit = 'negate'; path = 'L2.o.bias'; preview = $true } } -TimeoutSeconds $McpTimeoutSeconds
  $cpMcpPrevText = [string]($cpMcpPrev.result.content | Select-Object -First 1).text
  Assert-Check -Tier 4 -Area 'Parameters' -Name 'MCP nm_cartridges parameters_edit previews an operation by path (edit, path, preview) through the same route' `
    -Condition (($cpMcpPrev.result.isError -ne $true) -and ($cpMcpPrevText -match '"preview"')) -Detail $cpMcpPrevText
  $cpMcpText = [string]($cpMcp.result.content | Select-Object -First 1).text
  $cpMcpBadText = [string]($cpMcpBad.result.content | Select-Object -First 1).text
  Assert-Check -Tier 4 -Area 'Parameters' -Name 'MCP nm_cartridges parameters and parameters_edit answer through the same route (the section, and the same shape check)' `
    -Condition (($cpMcp.result.isError -ne $true) -and ($cpMcpText -match 'nm-parameters/1') -and ($cpMcpBad.result.isError -eq $true) -and ($cpMcpBadText -match 'shape check')) -Detail "read=$($cpMcp.result.isError) edit=$cpMcpBadText"
  # The overlay over MCP: the same route as the API and the Parameters page.
  $ovMcpMask = Invoke-McpRequest -Session $McpSession -Method 'tools/call' -Params @{ name = 'nm_cartridges'; arguments = @{ op = 'mask'; key = 'filter:reranker/e2e-params'; path = 'L5.swigate.gain'; preview = $true } } -TimeoutSeconds $McpTimeoutSeconds
  $ovMcpRead = Invoke-McpRequest -Session $McpSession -Method 'tools/call' -Params @{ name = 'nm_cartridges'; arguments = @{ op = 'overlay'; key = 'filter:reranker/e2e-params' } } -TimeoutSeconds $McpTimeoutSeconds
  $ovMcpClear = Invoke-McpRequest -Session $McpSession -Method 'tools/call' -Params @{ name = 'nm_cartridges'; arguments = @{ op = 'overlay_clear'; key = 'filter:reranker/e2e-params' } } -TimeoutSeconds $McpTimeoutSeconds
  $ovMcpMaskText = [string]($ovMcpMask.result.content | Select-Object -First 1).text
  $ovMcpReadText = [string]($ovMcpRead.result.content | Select-Object -First 1).text
  $ovMcpClearText = [string]($ovMcpClear.result.content | Select-Object -First 1).text
  Assert-Check -Tier 4 -Area 'Overlay' -Name 'MCP nm_cartridges mask previews, overlay reads and overlay_clear stops at the delete gate, through the same route' `
    -Condition (($ovMcpMask.result.isError -ne $true) -and ($ovMcpMaskText -match '"preview"') -and ($ovMcpRead.result.isError -ne $true) -and ($ovMcpReadText -match 'nm-overlay/1') -and ($ovMcpClearText -match 'confirm_token')) -Detail "mask=$ovMcpMaskText clear=$ovMcpClearText"
  $lbMcp = Invoke-McpRequest -Session $McpSession -Method 'tools/call' -Params @{ name = 'nm_cartridges'; arguments = @{ op = 'map_build'; role = 'reader'; share = '0.9' } } -TimeoutSeconds $McpTimeoutSeconds
  $lbMcpText = [string]($lbMcp.result.content | Select-Object -First 1).text
  Assert-Check -Tier 4 -Area 'Lab' -Name 'MCP nm_cartridges map_build reaches the lab build route (its share rule answers)' `
    -Condition (($lbMcp.result.isError -eq $true) -and ($lbMcpText -match 'share')) -Detail $lbMcpText

  $dsMcpList = Invoke-McpRequest -Session $McpSession -Method 'tools/call' -Params @{ name = 'nm_learning'; arguments = @{ op = 'distil' } } -TimeoutSeconds $McpTimeoutSeconds
  $dsMcpBuild = Invoke-McpRequest -Session $McpSession -Method 'tools/call' -Params @{ name = 'nm_cartridges'; arguments = @{ op = 'distil_build'; role = 'writer'; target = 'filter'; technique = 'lens' } } -TimeoutSeconds $McpTimeoutSeconds
  $dsMcpText = [string]($dsMcpBuild.result.content | Select-Object -First 1).text
  $dsMcpDistils = Invoke-McpRequest -Session $McpSession -Method 'tools/call' -Params @{ name = 'nm_cartridges'; arguments = @{ op = 'distils' } } -TimeoutSeconds $McpTimeoutSeconds
  Assert-Check -Tier 4 -Area 'Distil' -Name 'MCP nm_learning distil and nm_cartridges distils answer; distil_build reaches the lab build route (its kind and technique rule answers)' `
    -Condition (($dsMcpList.result.isError -ne $true) -and ($dsMcpDistils.result.isError -ne $true) -and ($dsMcpBuild.result.isError -eq $true) -and ($dsMcpText -match 'cannot fill')) -Detail $dsMcpText
  Assert-Check -Tier 4 -Area 'Phase-B' -Name 'MCP nm_cartridges inspect and map_read, and nm_settings connect, answer through the interface routes' `
    -Condition (($pbMcpIns.result.isError -ne $true) -and ($pbMcpMap.result.isError -ne $true) -and ($pbMcpConn.result.isError -ne $true)) `
    -Detail "inspect=$($pbMcpIns.result.isError) map=$($pbMcpMap.result.isError) connect=$($pbMcpConn.result.isError)"
  # The agent's own guidance over MCP (nm_goal op ...), through the same /api/guidance route: kept
  # as set by the agent, listed, and removed in two calls (a preview with a token, then with it).
  $gdMcpAdd = Invoke-McpRequest -Session $McpSession -Method 'tools/call' -Params @{ name = 'nm_goal'; arguments = @{ op = 'reminder_add'; text = 'e2e: the agent reminds itself'; trigger = 'stop'; repeat = 'once' } } -TimeoutSeconds $McpTimeoutSeconds
  $gdMcpAddText = [string]($gdMcpAdd.result.content | Select-Object -First 1).text
  $gdMcpJson = $null
  try { $gdMcpJson = $gdMcpAddText | ConvertFrom-Json } catch { $gdMcpJson = $null }
  $gdMcpRid = if ($gdMcpJson) { [string]$gdMcpJson.id } else { '' }
  $gdMcpProject = if ($gdMcpJson) { [string]$gdMcpJson.project } else { '' }
  $gdMcpMine = if ($gdMcpJson) { @($gdMcpJson.guidance.reminders | Where-Object { $_.id -eq $gdMcpRid -and $_.by -eq 'agent' }).Count -eq 1 } else { $false }
  $gdMcpList = Invoke-McpRequest -Session $McpSession -Method 'tools/call' -Params @{ name = 'nm_goal'; arguments = @{ op = 'reminder_list' } } -TimeoutSeconds $McpTimeoutSeconds
  $gdMcpListText = [string]($gdMcpList.result.content | Select-Object -First 1).text
  $gdMcpPrev = Invoke-McpRequest -Session $McpSession -Method 'tools/call' -Params @{ name = 'nm_goal'; arguments = @{ op = 'remove'; kind = 'reminder'; item = $gdMcpRid } } -TimeoutSeconds $McpTimeoutSeconds
  $gdMcpPrevText = [string]($gdMcpPrev.result.content | Select-Object -First 1).text
  $gdMcpTok = ''
  try { $gdMcpTok = [string]($gdMcpPrevText | ConvertFrom-Json).confirm_token } catch { $gdMcpTok = '' }
  $gdMcpDel = Invoke-McpRequest -Session $McpSession -Method 'tools/call' -Params @{ name = 'nm_goal'; arguments = @{ op = 'remove'; kind = 'reminder'; item = $gdMcpRid; confirm_token = $gdMcpTok } } -TimeoutSeconds $McpTimeoutSeconds
  $gdMcpLeft = @((Invoke-Api -BaseUrl $BaseUrl -Path ('/api/guidance?project=' + $gdMcpProject)).Json.reminders | Where-Object { $_.id -eq $gdMcpRid }).Count
  Assert-Check -Tier 4 -Area 'Guidance' -Name 'MCP nm_goal sets the agent''s own reminder on its served project (set by the agent), lists it, and removes it in two calls' `
    -Condition (($gdMcpAdd.result.isError -ne $true) -and $gdMcpMine -and ($gdMcpListText -match [regex]::Escape($gdMcpRid)) -and ($gdMcpPrevText -match '"preview"') -and $gdMcpTok -ne '' -and ($gdMcpDel.result.isError -ne $true) -and $gdMcpLeft -eq 0) `
    -Detail "add=$gdMcpAddText prev=$gdMcpPrevText left=$gdMcpLeft"
  $gdMcpOut = Invoke-McpRequest -Session $McpSession -Method 'tools/call' -Params @{ name = 'nm_goal'; arguments = @{ op = 'loop_outcomes'; limit = 5 } } -TimeoutSeconds $McpTimeoutSeconds
  $gdMcpOutText = [string]($gdMcpOut.result.content | Select-Object -First 1).text
  $gdMcpErr = Invoke-McpRequest -Session $McpSession -Method 'tools/call' -Params @{ name = 'nm_goal'; arguments = @{ op = 'reminder_add'; text = 'e2e: x'; trigger = 'error-repeats' } } -TimeoutSeconds $McpTimeoutSeconds
  $gdMcpErrText = [string]($gdMcpErr.result.content | Select-Object -First 1).text
  Assert-Check -Tier 4 -Area 'Guidance' -Name 'MCP nm_goal loop_outcomes answers the ended loops, and an error-repeats reminder without its error is refused' `
    -Condition (($gdMcpOut.result.isError -ne $true) -and ($gdMcpOutText -match '"outcomes"') -and ($gdMcpErr.result.isError -eq $true) -and ($gdMcpErrText -match 'error')) -Detail "out=$gdMcpOutText err=$gdMcpErrText"
  $pbServed = @((Invoke-Api -BaseUrl $BaseUrl -Path '/api/projects').Json.serving | Where-Object { [string]$_.project_key -ne '' } | Select-Object -First 1)
  if ($pbServed.Count -gt 0) {
    $pbKey = $pbServed[0].project_key
    $pbQ = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/memory/add' -Token $Token -Body (@{ project = $pbKey; kind = 'store'; text = 'e2e queued note for the serving process' } | ConvertTo-Json -Compress)
    $pbSeen = $false
    for ($i = 0; $i -lt 20 -and -not $pbSeen; $i++) {
      Start-Sleep -Seconds 1
      $pbSeen = @((Invoke-Api -BaseUrl $BaseUrl -Path ('/api/memory/items?project=' + $pbKey + '&q=queued')).Json.items).Count -ge 1
    }
    Assert-Check -Tier 4 -Area 'Phase-B' -Name 'POST /api/memory/add while a serving process holds the journal is queued and applied by it' `
      -Condition ($pbQ.Status -eq 200 -and $pbQ.Json.applied -eq 'queued' -and $pbSeen) -Detail "applied=$($pbQ.Json.applied) seen=$pbSeen project=$pbKey"
  } else {
    Record-Check -Tier 4 -Area 'Phase-B' -Name 'POST /api/memory/add while a serving process holds the journal is queued and applied by it' -Status 'SKIP' -Detail 'no serving process listed a project key'
  }

  # Stated preferences over MCP (nm_learning op preference_*), as the user's agent passes their
  # words: plugged in, the serving process puts a "preferences in force" block in the pack within
  # seconds (the library is watched), a draft is checked, and it is unplugged again.
  $prmPlug = Invoke-McpRequest -Session $McpSession -Method 'tools/call' -Params @{ name = 'nm_learning'; arguments = @{ op = 'preference_plug'; id = 'stated-e2e-writing' } } -TimeoutSeconds $McpTimeoutSeconds
  $prmSay = Invoke-McpRequest -Session $McpSession -Method 'tools/call' -Params @{ name = 'nm_learning'; arguments = @{ op = 'preference_say'; text = 'never say leverage'; cartridge = 'stated-e2e-writing' } } -TimeoutSeconds $McpTimeoutSeconds
  $prmSayJ = [string]($prmSay.result.content | Select-Object -First 1).text
  Assert-Check -Tier 4 -Area 'Preferences' -Name 'MCP nm_learning preference_plug and a preference_say preview answer through the interface routes' `
    -Condition (($prmPlug.result.isError -ne $true) -and ($prmSay.result.isError -ne $true) -and ($prmSayJ -match '"step":\s*"preview"') -and ($prmSayJ -match '"ban"')) -Detail $prmSayJ.Substring(0, [Math]::Min(200, $prmSayJ.Length))
  $prmBlock = $false
  for ($i = 0; $i -lt 10 -and -not $prmBlock; $i++) {
    Start-Sleep -Seconds 1
    $prmPack = Invoke-McpRequest -Session $McpSession -Method 'tools/call' -Params @{ name = 'nm_context_pack'; arguments = @{ question = "Where is the preference reader of the serve path ($i)?"; format = 'json'; form = 'full' } } -TimeoutSeconds $McpTimeoutSeconds
    $prmJ = $prmPack.result.content[0].text | ConvertFrom-Json
    $prmBlock = @($prmJ.preferences_in_force | Where-Object { [string]$_ -match 'em dash' }).Count -ge 1
  }
  Assert-Check -Tier 4 -Area 'Preferences' -Name 'nm_context_pack carries the preferences in force once the cartridge is plugged in' -Condition $prmBlock
  $prmApply = Invoke-McpRequest -Session $McpSession -Method 'tools/call' -Params @{ name = 'nm_learning'; arguments = @{ op = 'preference_apply'; text = ('Fast ' + [char]0x2014 + ' cached.') } } -TimeoutSeconds $McpTimeoutSeconds
  $prmApplyJ = ([string]($prmApply.result.content | Select-Object -First 1).text) | ConvertFrom-Json
  Assert-Check -Tier 4 -Area 'Preferences' -Name 'MCP preference_apply gives the same transform as the API' -Condition (($prmApply.result.isError -ne $true) -and $prmApplyJ.text -eq 'Fast, cached.') -Detail ([string]$prmApplyJ.text)
  $prmOff = Invoke-McpRequest -Session $McpSession -Method 'tools/call' -Params @{ name = 'nm_learning'; arguments = @{ op = 'preference_unplug'; id = 'stated-e2e-writing' } } -TimeoutSeconds $McpTimeoutSeconds
  Assert-Check -Tier 4 -Area 'Preferences' -Name 'MCP preference_unplug unplugs it' -Condition ($prmOff.result.isError -ne $true)

  # Parity of the settings file: every key of settings.json is shown key by key by the API and
  # by MCP, one entry is set and undone through each, and the program's own parity check passes.
  $parSaved = Invoke-Api -BaseUrl $BaseUrl -Path '/api/controls/saved'
  $parKeys = @($parSaved.Json.keys | ForEach-Object { [string]$_.key })
  $parControls = @((Invoke-Api -BaseUrl $BaseUrl -Path '/api/controls').Json).Count
  $parWant = @('base_cartridges', 'base_compensation', 'role_models', 'note')
  Assert-Check -Tier 1 -Area 'Parity' -Name 'GET /api/controls/saved lists every key of settings.json beside the controls, the controls counted, none unknown' `
    -Condition ($parSaved.Status -eq 200 -and $parSaved.JsonOk -and (@($parWant | Where-Object { $parKeys -notcontains $_ }).Count -eq 0) -and ([int]$parSaved.Json.controls -eq $parControls) -and (@($parSaved.Json.unknown).Count -eq 0)) `
    -Detail "keys=$($parKeys -join ',') controls=$($parSaved.Json.controls)/$parControls unknown=$(@($parSaved.Json.unknown) -join ',')"
  $parNote = @($parSaved.Json.keys | Where-Object { $_.key -eq 'note' })[0]
  $parSettable = @($parSaved.Json.keys | Where-Object { $_.key -ne 'note' -and ($null -ne $_.read_only -or @($_.entries).Count -lt 4 -or [string]$_.set.route -ne 'POST /api/controls') })
  Assert-Check -Tier 1 -Area 'Parity' -Name 'Each settings.json map is settable entry by entry through POST /api/controls; note is read only with its reason' `
    -Condition (($null -ne $parNote) -and ([string]$parNote.read_only -ne '') -and $parSettable.Count -eq 0) -Detail "note=$([string]$parNote.read_only) not_settable=$($parSettable.Count)"
  $parEntry = @(@($parSaved.Json.keys | Where-Object { $_.key -eq 'base_compensation' })[0].entries)[0]
  $parKey = [string]$parEntry.set_key
  $parValueOf = { param($k) [string](@(@((Invoke-Api -BaseUrl $BaseUrl -Path '/api/controls/saved').Json.keys | Where-Object { $_.key -eq 'base_compensation' })[0].entries | Where-Object { $_.set_key -eq $k })[0].value) }
  $parSet = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/controls' -Token $Token -Body (@{ key = $parKey; value = 'on' } | ConvertTo-Json -Compress)
  $parOn = & $parValueOf $parKey
  $parUndo = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/undo' -Token $Token -Body (@{ id = [string]$parSet.Json.audit_id } | ConvertTo-Json -Compress)
  $parOff = & $parValueOf $parKey
  $parBad = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/controls' -Token $Token -Body (@{ key = 'base_compensation:no-such-model'; value = 'on' } | ConvertTo-Json -Compress)
  Assert-Check -Tier 3 -Area 'Parity' -Name 'POST /api/controls sets a settings.json entry (base_compensation:<model>), /api/undo sets it back, an unknown model is refused' `
    -Condition ($parSet.Status -eq 200 -and [string]$parSet.Json.audit_id -ne '' -and $parOn -eq 'on' -and $parUndo.Status -eq 200 -and $parOff -eq 'off' -and $parBad.Status -eq 400) `
    -Detail "key=$parKey set=$($parSet.Status) on=$parOn undo=$($parUndo.Status) off=$parOff bad=$($parBad.Status)"
  $parMcpSaved = Invoke-McpRequest -Session $McpSession -Method 'tools/call' -Params @{ name = 'nm_settings'; arguments = @{ op = 'saved' } } -TimeoutSeconds $McpTimeoutSeconds
  $parMcpSavedText = [string]($parMcpSaved.result.content | Select-Object -First 1).text
  $parMcpSet = Invoke-McpRequest -Session $McpSession -Method 'tools/call' -Params @{ name = 'nm_settings'; arguments = @{ op = 'set'; key = $parKey; value = 'on' } } -TimeoutSeconds $McpTimeoutSeconds
  $parMcpSetText = [string]($parMcpSet.result.content | Select-Object -First 1).text
  $parMcpOn = & $parValueOf $parKey
  $parMcpId = ''
  try { $parMcpId = [string]($parMcpSetText | ConvertFrom-Json).audit_id } catch { $parMcpId = '' }
  $parMcpUndo = Invoke-McpRequest -Session $McpSession -Method 'tools/call' -Params @{ name = 'nm_settings'; arguments = @{ op = 'undo'; id = $parMcpId } } -TimeoutSeconds $McpTimeoutSeconds
  $parMcpOff = & $parValueOf $parKey
  Assert-Check -Tier 4 -Area 'Parity' -Name 'MCP nm_settings saved lists the same keys; set and undo change the same settings.json entry the API reads' `
    -Condition (($parMcpSaved.result.isError -ne $true) -and (@($parWant | Where-Object { $parMcpSavedText -notmatch [regex]::Escape('"' + $_ + '"') }).Count -eq 0) -and ($parMcpSet.result.isError -ne $true) -and $parMcpOn -eq 'on' -and ($parMcpUndo.result.isError -ne $true) -and $parMcpOff -eq 'off') `
    -Detail "saved=$($parMcpSaved.result.isError) set=$($parMcpSet.result.isError) on=$parMcpOn undo=$($parMcpUndo.result.isError) off=$parMcpOff"
  $parMcpW = Invoke-McpRequest -Session $McpSession -Method 'tools/call' -Params @{ name = 'nm_weights'; arguments = @{ model = 'qwen3-1.7b'; limit = 2 } } -TimeoutSeconds $McpTimeoutSeconds
  $parMcpWText = [string]($parMcpW.result.content | Select-Object -First 1).text
  $parApiW = Invoke-Api -BaseUrl $BaseUrl -Path '/api/models/weights?model=qwen3-1.7b&limit=2'
  # Where the model is installed both answer its view; where it is not, both say so.
  $parApiWErr = (Has-Prop $parApiW.Json 'error')
  Assert-Check -Tier 4 -Area 'Parity' -Name 'MCP nm_weights with model answers the weight view of that model, as GET /api/models/weights does' `
    -Condition ($parApiW.Status -eq 200 -and ((-not $parApiWErr -and ($parMcpW.result.isError -ne $true) -and ($parMcpWText -match 'qwen3-1\.7b')) -or ($parApiWErr -and ($parMcpW.result.isError -eq $true)))) -Detail $parMcpWText.Substring(0, [Math]::Min(160, $parMcpWText.Length))
  $parEap = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  $parCheckOut = (& $ResolvedExe check nmcore.parity 2>&1 | Out-String)
  $parCheckCode = $LASTEXITCODE
  $ErrorActionPreference = $parEap
  Assert-Check -Tier 1 -Area 'Parity' -Name 'neuromorph-nano check nmcore.parity passes (controls, settings.json keys, MCP ops and interface routes in step)' `
    -Condition ($parCheckCode -eq 0 -and $parCheckOut -match 'ok\s+nmcore\.parity') -Detail $parCheckOut.Trim()

  Close-McpSession -Session $McpSession
  $McpSession = $null

  # Verify audit log propagation visible from ui /api/audit
  $auditAfterMcp = @((Invoke-Api -BaseUrl $BaseUrl -Path '/api/audit?limit=100').Json.audit)
  $mcpToolRows = @($auditAfterMcp | Where-Object { $_.kind -eq 'tool' -and $_.tool -eq 'nm_context_pack' })
  Assert-Check -Tier 4 -Area 'MCP-Audit' -Name 'MCP tool calls recorded in isolated audit.jsonl and visible via /api/audit' `
    -Condition ($mcpToolRows.Count -ge 3) -Detail "tool_rows=$($mcpToolRows.Count)"

  # 4d + dec08 under --whole --neural (Full M5 mode only)
  if ($StrictMode) {
    $rSetBudget1500 = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/controls' -Token $Token -Body '{"key":"budget.pack_tokens","value":"1500"}'
    $budget1500AuditId = if ($rSetBudget1500.JsonOk) { $rSetBudget1500.Json.audit_id } else { $null }

    Write-Host "Tier 4: Starting Session B (launch serve --whole --neural with budget.pack_tokens=1500)..."
    $McpSession = Start-McpSession -LaunchExe $LaunchExe -RepoRoot $McpRepoRoot -TempHome $TempHome -ModelsDir $ModelsDir -ExtraArgs '--whole --neural'
    $null = Invoke-McpRequest -Session $McpSession -Method 'initialize' -Params @{
      protocolVersion = '2025-06-18'
      capabilities    = @{}
      clientInfo      = @{ name = 'e2e-verify-whole'; version = '1.0' }
    } -TimeoutSeconds $McpTimeoutSeconds
    Send-McpNotification -Session $McpSession -Method 'notifications/initialized'

    # Broad real query omitting `tokens` so budget.pack_tokens=1500 governs
    $swWhole = [System.Diagnostics.Stopwatch]::StartNew()
    $wholeBudgetReply = Invoke-McpRequest -Session $McpSession -Method 'tools/call' -Params @{
      name      = 'nm_context_pack'
      arguments = @{
        question = 'Where does the server answer a single incoming JSON-RPC message from the client, picking the reply by method?'
        format   = 'json'
        form     = 'full'
      }
    } -TimeoutSeconds $McpTimeoutSeconds
    $swWhole.Stop()
    $wholeBudgetPack = $wholeBudgetReply.result.content[0].text | ConvertFrom-Json
    $capOk = ($wholeBudgetReply.result.isError -ne $true) -and
      ($wholeBudgetPack.withheld -eq $false) -and
      (@($wholeBudgetPack.packets).Count -ge 1) -and
      (([int]$wholeBudgetPack.tokens -le 1500) -or (@($wholeBudgetPack.packets).Count -eq 1))
    Assert-Check -Tier 4 -Area 'MCP-Budget' -Name 'budget.pack_tokens=1500 caps omitted tokens on fresh --whole --neural serve' `
      -Condition $capOk `
      -Detail "elapsed=$($swWhole.ElapsedMilliseconds)ms tokens=$($wholeBudgetPack.tokens) packets=$(@($wholeBudgetPack.packets).Count)"

    # dec08 decoy under --whole --neural
    $decoyWholeReply = Invoke-McpRequest -Session $McpSession -Method 'tools/call' -Params @{
      name      = 'nm_context_pack'
      arguments = @{
        question = 'How do you configure content keys to use BLAKE3 instead of SHA-1?'
        format   = 'json'
        form     = 'full'
      }
    } -TimeoutSeconds $McpTimeoutSeconds
    $decoyWholePack = $decoyWholeReply.result.content[0].text | ConvertFrom-Json
    Assert-Check -Tier 4 -Area 'MCP-Decoy' -Name 'dec08 decoy query withheld=true & packets=[] under --whole --neural' `
      -Condition (($decoyWholeReply.result.isError -ne $true) -and ($decoyWholePack.withheld -eq $true) -and (@($decoyWholePack.packets).Count -eq 0)) `
      -Detail "withheld=$($decoyWholePack.withheld) packets=$(@($decoyWholePack.packets).Count) reason=$($decoyWholePack.reason)"

    Close-McpSession -Session $McpSession
    $McpSession = $null

    if ($budget1500AuditId) {
      $null = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/undo' -Token $Token -Body (@{ id = $budget1500AuditId } | ConvertTo-Json -Compress)
    }

    # 4e: Dynamic MCP Tool Activation (mcp.tool_count=5, mcp.dynamic_tools=on)
    $rSetToolCount = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/controls' -Token $Token -Body '{"key":"mcp.tool_count","value":"5"}'
    $tcAuditId = if ($rSetToolCount.JsonOk) { $rSetToolCount.Json.audit_id } else { $null }
    $McpSession = Start-McpSession -LaunchExe $LaunchExe -RepoRoot $McpRepoRoot -TempHome $TempHome -ModelsDir $ModelsDir -ExtraArgs ''
    $initDyn = Invoke-McpRequest -Session $McpSession -Method 'initialize' -Params @{
      protocolVersion = '2025-06-18'
      capabilities    = @{}
      clientInfo      = @{ name = 'e2e-verify-dyn'; version = '1.0' }
    } -TimeoutSeconds $McpTimeoutSeconds
    Send-McpNotification -Session $McpSession -Method 'notifications/initialized'
    $listBefore = Invoke-McpRequest -Session $McpSession -Method 'tools/list' -Params @{} -TimeoutSeconds $McpTimeoutSeconds
    $namesBefore = @($listBefore.result.tools | ForEach-Object { $_.name })
    $statusCall = Invoke-McpRequest -Session $McpSession -Method 'tools/call' -Params @{
      name      = 'nm_status'
      arguments = @{}
    } -TimeoutSeconds $McpTimeoutSeconds
    $listAfter = Invoke-McpRequest -Session $McpSession -Method 'tools/list' -Params @{} -TimeoutSeconds $McpTimeoutSeconds
    $namesAfter = @($listAfter.result.tools | ForEach-Object { $_.name })
    Close-McpSession -Session $McpSession
    $McpSession = $null
    if ($tcAuditId) {
      $null = Invoke-Api -BaseUrl $BaseUrl -Method 'POST' -Path '/api/undo' -Token $Token -Body (@{ id = $tcAuditId } | ConvertTo-Json -Compress)
    }
    Assert-Check -Tier 4 -Area 'MCP-Dynamic' -Name 'Dynamic tool activation starts at 5 tools and promotes unadvertised nm_status into tools/list' `
      -Condition (($initDyn.result.capabilities.tools.listChanged -eq $true) -and ($namesBefore.Count -eq 5) -and ($statusCall.result.isError -ne $true) -and ($namesAfter.Count -eq 6) -and ($namesAfter -contains 'nm_status')) `
      -Detail "before=$($namesBefore.Count) after=$($namesAfter.Count) listChanged=$($initDyn.result.capabilities.tools.listChanged)"
  } else {
    Record-Check -Tier 4 -Area 'MCP-Budget' -Name 'budget.pack_tokens=1500 caps omitted tokens on fresh --whole --neural serve' -Status 'SKIP' -Detail 'BaselineSmoke'
    Record-Check -Tier 4 -Area 'MCP-Decoy' -Name 'dec08 decoy query withheld=true & packets=[] under --whole --neural' -Status 'SKIP' -Detail 'BaselineSmoke'
    Record-Check -Tier 4 -Area 'MCP-Dynamic' -Name 'Dynamic tool activation starts at 5 tools and promotes unadvertised nm_status into tools/list' -Status 'SKIP' -Detail 'BaselineSmoke'
  }
}
finally {
  if ($null -ne $McpSession) { Close-McpSession -Session $McpSession -WaitSeconds 5 }
  Stop-ProcessTreeSafe -Proc $UiProc
  Stop-LiveBinariesInHome -HomeDir $TempHome

  if ($null -eq $OrigNanoHome) { Remove-Item Env:NANO_HOME -ErrorAction SilentlyContinue } else { $env:NANO_HOME = $OrigNanoHome }
  if ($null -eq $OrigNanoModels) { Remove-Item Env:NANO_MODELS -ErrorAction SilentlyContinue } else { $env:NANO_MODELS = $OrigNanoModels }
  if ($null -eq $OrigClaudeDir) { Remove-Item Env:CLAUDE_PROJECT_DIR -ErrorAction SilentlyContinue } else { $env:CLAUDE_PROJECT_DIR = $OrigClaudeDir }

  if ($TempHome -and (Test-Path $TempHome)) {
    for ($k = 0; $k -lt 10; $k++) {
      try {
        Remove-Item -Path $TempHome -Recurse -Force -ErrorAction Stop
        break
      } catch {
        Start-Sleep -Milliseconds 300
      }
    }
  }
}

Write-Host ""
Write-Host "=== E2E API & MCP Summary ==="
for ($t = 1; $t -le 4; $t++) {
  $rows = @($Script:Results | Where-Object { $_.Tier -eq $t })
  $p = @($rows | Where-Object { $_.Status -eq 'PASS' }).Count
  $f = @($rows | Where-Object { $_.Status -eq 'FAIL' }).Count
  $s = @($rows | Where-Object { $_.Status -eq 'SKIP' }).Count
  Write-Host ("Tier {0}: PASS={1} FAIL={2} SKIP={3}" -f $t, $p, $f, $s)
}
$TotalPass = @($Script:Results | Where-Object { $_.Status -eq 'PASS' }).Count
$TotalFail = @($Script:Results | Where-Object { $_.Status -eq 'FAIL' }).Count
$TotalSkip = @($Script:Results | Where-Object { $_.Status -eq 'SKIP' }).Count
Write-Host ("Total : PASS={0} FAIL={1} SKIP={2}" -f $TotalPass, $TotalFail, $TotalSkip)

if ($TotalFail -gt 0 -or ($StrictMode -and $TotalSkip -gt 0)) {
  exit 1
}
exit 0
