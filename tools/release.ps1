#Requires -Version 5.1
<#
.SYNOPSIS
  The release seal of the Neuromorph Nano: fresh binaries into bin\, sealed cards, INTEGRITY.md,
  every check, the end-to-end suite, and with -Tag a local git tag.
.DESCRIPTION
  Steps, in order; the first one that fails stops the run and the summary says which:
    1. take scratch\LOCKS\build.lock (created only if it does not exist; holds the lane, PID and
       UTC time; removed at the end, also after a failure);
    2. copy scratch\target-claude\release\neuromorph-nano.exe and nano-lab.exe into bin\;
    3. tools\gen-cards.ps1, then bin\neuromorph-nano.exe seal on every crates\*\part.json (explicit
       paths), then check cards;
    4. tools\gen-integrity.ps1, then check integrity;
    5. bare check (every section and every crate);
    6. tools\e2e-verify.ps1 -Exe bin\neuromorph-nano.exe, then node interface\scripts\e2e-ui-verify.mjs --exe
       bin\neuromorph-nano.exe (strict: every page and interface control, isolated home; 2026-11-25);
    7. with -Tag <name>: git tag <name> in this repository only. Nothing is ever pushed.
  Before step 2 it builds the release binaries (cargo build --release --workspace --bins with
  CARGO_TARGET_DIR set to the folder above -Build, scratch\target-claude by default), so the
  binaries sealed are never older than their source.
.PARAMETER Tag
  A local git tag to create after every step passed (for example v1.0.0). Never pushed.
.PARAMETER Lane
  The name written into the lock file.
.PARAMETER SkipE2E
  Leave out the end-to-end suite (the summary marks it skipped; for a quick re-seal only).
#>
[CmdletBinding()]
param(
  [string]$Root = '',
  [string]$Tag = '',
  [string]$Lane = 'release',
  [string]$Build = '',
  [switch]$SkipE2E
)

$ErrorActionPreference = 'Stop'
# 2026-11-25: the folder above tools\ when -Root is not given ($PSScriptRoot can be empty in a
# parameter default under Windows PowerShell 5.1 when the script is started by a relative path).
if (-not $Root) { $Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition) }
$Root = (Resolve-Path -LiteralPath $Root).Path
if (-not $Build) { $Build = Join-Path $Root 'scratch\target-claude\release' }
$bin = Join-Path $Root 'bin'
$exe = Join-Path $bin 'neuromorph-nano.exe'
$lockDir = Join-Path $Root 'scratch\LOCKS'
$lock = Join-Path $lockDir 'build.lock'
$steps = New-Object System.Collections.Generic.List[object]
$started = Get-Date
$haveLock = $false
$failed = $null

function Add-Step([string]$name, [string]$state, [double]$secs, [string]$note = '') {
  $steps.Add([PSCustomObject]@{ Step = $name; State = $state; Seconds = [Math]::Round($secs, 1); Note = $note })
}

# Runs one step; a step fails on a thrown error or a non-zero exit code of the last program it ran.
function Invoke-Step([string]$name, [scriptblock]$body) {
  if ($script:failed) { Add-Step $name 'not run' 0; return }
  Write-Host ""
  Write-Host "=== $name ==="
  $t = Get-Date
  try {
    $global:LASTEXITCODE = 0
    & $body
    if ($LASTEXITCODE -ne 0) { throw "exit code $LASTEXITCODE" }
    Add-Step $name 'ok' ((Get-Date) - $t).TotalSeconds
  } catch {
    Add-Step $name 'FAILED' ((Get-Date) - $t).TotalSeconds $_.Exception.Message
    $script:failed = $name
  }
}

function Invoke-Nano([string[]]$argList) {
  # 2026-11-25: the program writes plain status lines to stderr (for example which graphics card
  # it chose); under ErrorActionPreference Stop, Windows PowerShell 5.1 turns any stderr line of a
  # native program into a terminating error. The exit code alone decides.
  $before = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try { & $exe @argList 2>&1 | ForEach-Object { "$_" } } finally { $ErrorActionPreference = $before }
  if ($LASTEXITCODE -ne 0) { throw "neuromorph-nano $($argList -join ' ') exited $LASTEXITCODE" }
}

try {
  Invoke-Step 'take scratch\LOCKS\build.lock' {
    New-Item -ItemType Directory -Force $lockDir | Out-Null
    $body = '{"lane": "' + $Lane + '", "pid": ' + $PID + ', "started_utc": "' + (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ') + '", "cmd": "tools\\release.ps1"}'
    try {
      # CreateNew fails when the file exists: the lock is taken only if nobody holds it.
      $fs = [IO.File]::Open($lock, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    } catch {
      if (-not (Test-Path -LiteralPath $lock)) { throw "cannot create $lock : $($_.Exception.Message)" }
      $held = ''
      try { $held = Get-Content -Raw -LiteralPath $lock } catch { }
      throw "build.lock is held: $held"
    }
    try {
      $bytes = [Text.Encoding]::UTF8.GetBytes($body + "`n")
      $fs.Write($bytes, 0, $bytes.Length)
    } finally { $fs.Dispose() }
    $script:haveLock = $true
    Write-Host "lock taken: $lock"
  }

  # 2026-11-26: the release binaries are built here, so a seal never passes on a binary older than
  # its source (cargo is incremental: nothing is rebuilt when nothing changed).
  Invoke-Step 'build the release binaries (cargo build --release --workspace --bins)' {
    $env:CARGO_TARGET_DIR = Split-Path -Parent $Build
    Push-Location -LiteralPath $Root
    try {
      & cargo build --release --workspace --bins
      if ($LASTEXITCODE -ne 0) { throw "cargo build exited $LASTEXITCODE" }
    } finally { Pop-Location }
  }

  Invoke-Step 'copy release binaries into bin\' {
    New-Item -ItemType Directory -Force $bin | Out-Null
    foreach ($name in @('neuromorph-nano.exe', 'nano-lab.exe')) {
      $src = Join-Path $Build $name
      if (-not (Test-Path -LiteralPath $src)) { throw "$src is missing (build the release first)" }
      Copy-Item -Force -LiteralPath $src -Destination (Join-Path $bin $name)
      $h = (Get-FileHash -LiteralPath (Join-Path $bin $name) -Algorithm SHA256).Hash.ToLower()
      Write-Host ("copied {0} ({1:N1} MiB, sha256 {2})" -f $name, ((Get-Item -LiteralPath $src).Length / 1MB), $h.Substring(0, 16))
    }
    & $exe --version
  }

  Invoke-Step 'cards: gen-cards, seal, check cards' {
    & (Join-Path $Root 'tools\gen-cards.ps1') -Root $Root
    $cards = @(Get-ChildItem -Path (Join-Path $Root 'crates') -Directory | ForEach-Object { Join-Path $_.FullName 'part.json' } | Where-Object { Test-Path -LiteralPath $_ } | Sort-Object)
    if ($cards.Count -eq 0) { throw 'no crates\*\part.json found' }
    Invoke-Nano (@('seal') + $cards)
    Invoke-Nano @('check', 'cards')
  }

  Invoke-Step 'integrity: gen-integrity, check integrity' {
    & (Join-Path $Root 'tools\gen-integrity.ps1') -Root $Root
    Invoke-Nano @('check', 'integrity')
  }

  Invoke-Step 'check (every section and crate)' {
    Invoke-Nano @('check')
  }

  if ($SkipE2E) {
    if (-not $failed) { Add-Step 'e2e-verify' 'skipped' 0 '-SkipE2E' }
  } else {
    Invoke-Step 'e2e-verify -Exe bin\neuromorph-nano.exe' {
      & (Join-Path $Root 'tools\e2e-verify.ps1') -Root $Root -Exe $exe
      if ($LASTEXITCODE -ne 0) { throw "e2e-verify exited $LASTEXITCODE" }
    }
    Invoke-Step 'e2e-ui-verify (strict, isolated home)' {
      & node (Join-Path $Root 'interface\scripts\e2e-ui-verify.mjs') --exe $exe
      if ($LASTEXITCODE -ne 0) { throw "e2e-ui-verify exited $LASTEXITCODE" }
    }
  }

  if ($Tag) {
    Invoke-Step "git tag $Tag (local only, never pushed)" {
      $existing = & git -c safe.directory=* -C $Root tag --list $Tag
      if ($existing) { throw "tag $Tag already exists" }
      & git -c safe.directory=* -C $Root tag -a $Tag -m "Neuromorph Nano $Tag"
      if ($LASTEXITCODE -ne 0) { throw "git tag exited $LASTEXITCODE" }
      Write-Host "tagged $Tag locally; push it yourself when ready"
    }
  }
} finally {
  if ($haveLock) {
    Remove-Item -Force -LiteralPath $lock -ErrorAction SilentlyContinue
    Write-Host "lock released: $lock"
  }
  Write-Host ""
  Write-Host "=== release summary ==="
  $steps | Format-Table -AutoSize | Out-String -Width 200 | Write-Host
  $mins = ((Get-Date) - $started).TotalMinutes
  if ($failed) {
    Write-Host ("RELEASE FAILED at: {0} ({1:N1} min)" -f $failed, $mins)
  } else {
    Write-Host ("RELEASE OK ({0:N1} min){1}" -f $mins, $(if ($Tag) { "; local tag $Tag" } else { '' }))
  }
}
if ($failed) { exit 1 }
exit 0
