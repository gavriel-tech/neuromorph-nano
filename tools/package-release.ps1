#Requires -Version 5.1
<#
.SYNOPSIS
  Assemble a clean public Windows x64 release archive and SHA256SUMS.txt with hygiene checks.
.DESCRIPTION
  Copies only public release payloads (bin\neuromorph-nano.exe, designs\, interface\out\,
  rules.md, INTEGRITY.md, license/notice docs, and tools\install.ps1) into a staging directory,
  enforces a strict zero-leakage hygiene scan (refusing private folders, personal paths, or
  unwanted artifacts), zips into dist\neuromorph-nano-<Version>-win-x64.zip, writes
  dist\SHA256SUMS.txt, and verifies the archive hash.
.EXAMPLE
  .\tools\package-release.ps1 -Version v0.1.0
#>
[CmdletBinding()]
param(
  [string]$Root = '',
  [string]$Version = 'v0.1.01',
  [string]$OutDir = '',
  [string]$BinDir = '',
  [switch]$IncludeLab
)

$ErrorActionPreference = 'Stop'
# 2026-11-25: the folder above tools\ when -Root is not given ($PSScriptRoot can be empty in a
# parameter default under Windows PowerShell 5.1 when the script is started by a relative path).
if (-not $Root) { $Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition) }
$Root = (Resolve-Path -LiteralPath $Root).Path
if (-not $OutDir) { $OutDir = Join-Path $Root 'dist' }
if (-not $BinDir) { $BinDir = Join-Path $Root 'bin' }
# Absolute paths, so the staged paths and the files found under them compare by prefix.
$OutDir = [IO.Path]::GetFullPath($OutDir)
$BinDir = [IO.Path]::GetFullPath($BinDir)

$bundleName = "neuromorph-nano-$Version-win-x64"
$zipPath = Join-Path $OutDir "$bundleName.zip"
$sumPath = Join-Path $OutDir 'SHA256SUMS.txt'
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$stageRoot = Join-Path $OutDir (".stage-" + [Guid]::NewGuid().ToString('N').Substring(0, 8))
$stageBundle = Join-Path $stageRoot $bundleName

# 2026-11-25: weights\ and library\ are staged from allowlists only (weight cards; the release
# catalogue, job filters and hardware cartridges); hygiene check 3 refuses anything else in them.
$forbiddenRootDirs = @('internal', 'scratch', 'runs', 'memory', '.git', 'target', 'node_modules', 'private', 'muscle', 'units', 'tests')
$forbiddenAnyDirs = @('internal', 'scratch', '.git', 'node_modules', 'private')
# 2026-11-26: the private folder (internal\, with the private agent rules in internal\agents\) never
# ships, and neither does the local file that loads those rules into development sessions.
$forbiddenFileNames = @('claude.local.md', 'interface-agents.md')
# Machine paths and the private repository's name, read from this machine at run time (so the tool
# itself names none of them): the Nano's folder, the user's profile folder, and the name of the
# remote this checkout pushes to. Each is matched as literal text.
$forbiddenPathPatterns = New-Object System.Collections.Generic.List[string]
foreach ($p in @($Root, $env:USERPROFILE)) {
  if ($p) {
    $forbiddenPathPatterns.Add([regex]::Escape($p.TrimEnd('\') + '\'))
    $forbiddenPathPatterns.Add([regex]::Escape(($p.TrimEnd('\') -replace '\\', '/') + '/'))
  }
}
$originUrl = ''
try { $originUrl = (& git -C $Root remote get-url origin 2>$null | Select-Object -First 1) } catch { }
if ($originUrl) {
  $privateName = [IO.Path]::GetFileNameWithoutExtension((($originUrl -split '[/:]') | Select-Object -Last 1))
  if ($privateName) { $forbiddenPathPatterns.Add([regex]::Escape($privateName) + '(?![A-Za-z0-9_-])') }
}

try {
  $exe = Join-Path $BinDir 'neuromorph-nano.exe'
  if (-not (Test-Path -LiteralPath $exe)) {
    throw "Missing binary: $exe (run cargo build --release and copy to bin\ first)"
  }

  New-Item -ItemType Directory -Force -Path (Join-Path $stageBundle 'bin') | Out-Null
  New-Item -ItemType Directory -Force -Path (Join-Path $stageBundle 'tools') | Out-Null

  Copy-Item -Force -LiteralPath $exe -Destination (Join-Path $stageBundle 'bin\neuromorph-nano.exe')
  if ($IncludeLab) {
    $lab = Join-Path $BinDir 'nano-lab.exe'
    if (Test-Path -LiteralPath $lab) {
      Copy-Item -Force -LiteralPath $lab -Destination (Join-Path $stageBundle 'bin\nano-lab.exe')
    }
  }

  $rootFiles = @(
    'nano.json',
    'README.md',
    'LICENSE',
    'COMMERCIAL.md',
    'CONTRIBUTING.md',
    'THIRD_PARTY_NOTICES.md',
    'INTEGRITY.md',
    'rules.md'
  )
  foreach ($f in $rootFiles) {
    $src = Join-Path $Root $f
    if (Test-Path -LiteralPath $src) {
      Copy-Item -Force -LiteralPath $src -Destination (Join-Path $stageBundle $f)
    } else {
      throw "Required release file missing: $src"
    }
  }

  $installScript = Join-Path $Root 'tools\install.ps1'
  if (Test-Path -LiteralPath $installScript) {
    Copy-Item -Force -LiteralPath $installScript -Destination (Join-Path $stageBundle 'tools\install.ps1')
  }

  $designsSrc = Join-Path $Root 'designs'
  if (Test-Path -LiteralPath $designsSrc) {
    Copy-Item -Recurse -Force -LiteralPath $designsSrc -Destination (Join-Path $stageBundle 'designs')
  }

  # 2026-11-25: what the installed program reads at start besides designs\ (found by installing
  # the zip outside the source tree): the part cards, the weight cards (the installer fetches the
  # model files they name), and the cartridge library's small files.
  Push-Location -LiteralPath $Root
  try {
    $cards = @(& git -c safe.directory=* ls-files 'crates/*/part.json')
  } finally {
    Pop-Location
  }
  if ($cards.Count -eq 0) { throw 'No part cards found (git ls-files crates/*/part.json).' }
  foreach ($c in $cards) {
    $dest = Join-Path $stageBundle ($c -replace '/', '\')
    New-Item -ItemType Directory -Force -Path (Split-Path $dest -Parent) | Out-Null
    Copy-Item -Force -LiteralPath (Join-Path $Root $c) -Destination $dest
  }

  $modelIds = @('qwen3-0.6b', 'qwen3-1.7b', 'qwen3-embedding-0.6b', 'qwen3-reranker-0.6b')
  foreach ($m in $modelIds) {
    $card = Join-Path $Root "weights\$m\part.json"
    if (-not (Test-Path -LiteralPath $card)) { throw "Weight card missing: $card" }
    $destDir = Join-Path $stageBundle "weights\$m"
    New-Item -ItemType Directory -Force -Path $destDir | Out-Null
    Copy-Item -Force -LiteralPath $card -Destination (Join-Path $destDir 'part.json')
  }
  $weightsReadme = Join-Path $Root 'weights\README.md'
  if (Test-Path -LiteralPath $weightsReadme) { Copy-Item -Force -LiteralPath $weightsReadme -Destination (Join-Path $stageBundle 'weights\README.md') }

  # The cartridge library: a release catalogue holding only what ships or is fetched (the four base
  # cartridges, the hardware cartridges, the stacks made only of those), the job filters the served
  # path uses by default, and library\cartridges\release-files.json naming each base cartridge file
  # the installer fetches (path, SHA-256, bytes). This machine's own stack state (memory\stacks\),
  # learned and adapter cartridges stay out.
  $libSrc = Join-Path $Root 'library'
  $libDst = Join-Path $stageBundle 'library'
  $catSrc = Join-Path $libSrc 'cartridges\catalogue.json'
  if (-not (Test-Path -LiteralPath $catSrc)) { $catSrc = Join-Path $libSrc 'catalogue.json' }
  $catalogue = Get-Content -Raw -LiteralPath $catSrc | ConvertFrom-Json
  # Keep only the serving base cartridges and the latest version of each hardware cartridge (no
  # superseded versions, trial-only learned cartridges, or adapters).
  $servingBaseKeys = @()
  $baseDesignPath = Join-Path $Root 'designs\base-cartridges.json'
  if (Test-Path -LiteralPath $baseDesignPath) {
    $bd = Get-Content -Raw -LiteralPath $baseDesignPath | ConvertFrom-Json
    foreach ($p in $bd.serving.PSObject.Properties) {
      $servingBaseKeys += "$($p.Value.cartridge)@$($p.Value.version)"
    }
  }
  $allPresent = @($catalogue.cartridges | Where-Object { $_.file -like 'base/*' -or $_.file -like 'hardware/*' } | Where-Object { Test-Path -LiteralPath (Join-Path $libSrc ("cartridges\" + ($_.file -replace '/', '\'))) })
  $keepBase = @($allPresent | Where-Object { $_.file -like 'base/*' -and ($servingBaseKeys.Count -eq 0 -or $servingBaseKeys -contains "$($_.id)@$($_.version)") })
  $keepHw = @($allPresent | Where-Object { $_.file -like 'hardware/*' } | Group-Object id | ForEach-Object { $_.Group | Sort-Object version -Descending | Select-Object -First 1 })
  $keep = @($keepBase + $keepHw)
  $keepKeys = @($keep | ForEach-Object { "$($_.id)@$($_.version)" })
  $stacks = @($catalogue.stacks | Where-Object { $s = $_; (-not $s.trial) -and @($s.cartridges | Where-Object { $keepKeys -notcontains $_ }).Count -eq 0 } | Group-Object id | ForEach-Object { $_.Group | Sort-Object version -Descending | Select-Object -First 1 })
  $stackKeys = @($stacks | ForEach-Object { "$($_.id)@$($_.version)" })
  $active = New-Object PSObject
  if ($catalogue.active) {
    foreach ($p in $catalogue.active.PSObject.Properties) {
      if ($stackKeys -contains $p.Value) { $active | Add-Member -NotePropertyName $p.Name -NotePropertyValue $p.Value }
    }
  }
  $release = [ordered]@{ active = $active; adapters = @(); cartridges = $keep; note = $catalogue.note; schema = $catalogue.schema; stacks = $stacks }
  New-Item -ItemType Directory -Force -Path (Join-Path $libDst 'cartridges') | Out-Null
  $json = $release | ConvertTo-Json -Depth 32
  # Shipped as release-catalogue.json: the installer merges it into the user's library
  # (`neuromorph-nano library merge`), which creates catalogue.json on a first install and adds
  # only the new rows on an upgrade.
  [IO.File]::WriteAllText((Join-Path $libDst 'cartridges\release-catalogue.json'), $json, (New-Object System.Text.UTF8Encoding($false)))
  $fetch = @()
  foreach ($c in $keep) {
    $rel = 'cartridges/' + $c.file
    $src = Join-Path $libSrc ($rel -replace '/', '\')
    if ($c.file -like 'base/*') {
      # `source`: where the file sits in the public Hugging Face cartridge library the installer
      # reads (base/<role>/<id>.cartridge).
      $role = ($c.file -split '/')[1]
      $fetch += [ordered]@{ path = $rel; source = "base/$role/$($c.id).cartridge"; sha256 = $c.sha256; bytes = [int64]$c.bytes }
      $manifest = Join-Path (Split-Path $src -Parent) 'MANIFEST.md'
      if (Test-Path -LiteralPath $manifest) {
        $md = Join-Path $libDst ((Split-Path $rel -Parent) -replace '/', '\')
        New-Item -ItemType Directory -Force -Path $md | Out-Null
        Copy-Item -Force -LiteralPath $manifest -Destination (Join-Path $md 'MANIFEST.md')
      }
    } else {
      $dest = Join-Path $libDst ($rel -replace '/', '\')
      New-Item -ItemType Directory -Force -Path (Split-Path $dest -Parent) | Out-Null
      Copy-Item -Force -LiteralPath $src -Destination $dest
    }
  }
  $filesDoc = [ordered]@{
    note = 'Base cartridge files the installer fetches into library\ (tools\install.ps1), each checked against its SHA-256 before it is placed.'
    files = $fetch
  }
  [IO.File]::WriteAllText((Join-Path $libDst 'cartridges\release-files.json'), ($filesDoc | ConvertTo-Json -Depth 8), (New-Object System.Text.UTF8Encoding($false)))
  Get-ChildItem -LiteralPath (Join-Path $libSrc 'parameter-maps') -Directory | ForEach-Object {
    $role = $_.Name
    $lineagePath = Join-Path $_.FullName 'lineage.json'
    $lin = $null
    $usedNames = @()
    if (Test-Path -LiteralPath $lineagePath) {
      $lin = Get-Content -Raw -LiteralPath $lineagePath | ConvertFrom-Json
      foreach ($prop in $lin.maps.PSObject.Properties) {
        if ([string]$prop.Value.outcome -eq 'used') { $usedNames += $prop.Name }
      }
    }
    $copied = @()
    Get-ChildItem -LiteralPath $_.FullName -File | Where-Object {
      (($_.Name -like 'job-filter-*.json') -or ($_.Name -eq 'super-weights.json')) -and
      ($usedNames.Count -eq 0 -or $usedNames -contains $_.Name)
    } | Sort-Object Name | ForEach-Object {
      $dest = Join-Path $libDst "parameter-maps\$role"
      New-Item -ItemType Directory -Force -Path $dest | Out-Null
      Copy-Item -Force -LiteralPath $_.FullName -Destination (Join-Path $dest $_.Name)
      $sha = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
      $purpose = if ($_.Name -like 'job-filter-*') { 'serving' } else { 'base-build' }
      $desc = if ($_.Name -eq 'job-filter-read.json') { 'Serving job filter for Reader: calibrated per-layer gain and bias tuning for workspace reading and cited span extraction.' }
        elseif ($_.Name -eq 'job-filter-write.json') { 'Serving job filter for Writer: per-layer gain and bias tuning for idle L1-L4 world-model note writing.' }
        elseif ($_.Name -eq 'job-filter-judge-code.json') { 'Serving job filter for Reranker: sharpens close-candidate code judging and decoy withholding.' }
        elseif ($_.Name -eq 'job-filter-code-slice.json') { "Serving job filter for $($role): focuses $role layers on code-slice representation and retrieval." }
        elseif ($_.Name -eq 'super-weights.json') { "Precision surgery baseline for $($role): critical high-leverage weight coordinates restored by cascade surgery." }
        else { "Verified parameter map for $role." }
      $copied += "| ``$($_.Name)`` | $purpose | $('{0:N0}' -f $_.Length) | ``$sha`` | $desc |"
    }
    if ($copied.Count -gt 0) {
      $mDest = Join-Path $libDst "parameter-maps\$role\MANIFEST.md"
      $mText = "# $role parameter maps & job filters`n`n| File | Purpose | Bytes | SHA-256 | What it does |`n|---|---|---|---|---|`n" + ($copied -join "`n") + "`n"
      [IO.File]::WriteAllText($mDest, $mText, (New-Object System.Text.UTF8Encoding($false)))
    }
  }

  $uiOut = Join-Path $Root 'interface\out'
  if (Test-Path -LiteralPath $uiOut) {
    New-Item -ItemType Directory -Force -Path (Join-Path $stageBundle 'interface') | Out-Null
    Copy-Item -Recurse -Force -LiteralPath $uiOut -Destination (Join-Path $stageBundle 'interface\out')
  } else {
    Write-Warning 'interface\out is missing; static UI bundle will not be included in this zip.'
  }

  # Hygiene check 1: ensure no forbidden directory names leaked into staging.
  Get-ChildItem -LiteralPath $stageBundle -Directory | ForEach-Object {
    if ($forbiddenRootDirs -contains $_.Name.ToLowerInvariant()) {
      throw "Hygiene failure: forbidden root directory staged: $($_.FullName)"
    }
  }
  Get-ChildItem -LiteralPath $stageBundle -Recurse -Directory | ForEach-Object {
    if ($forbiddenAnyDirs -contains $_.Name.ToLowerInvariant()) {
      throw "Hygiene failure: forbidden directory staged: $($_.FullName)"
    }
  }

  Get-ChildItem -LiteralPath $stageBundle -Recurse -File | ForEach-Object {
    if ($forbiddenFileNames -contains $_.Name.ToLowerInvariant()) {
      throw "Hygiene failure: private file staged: $($_.FullName)"
    }
  }

  # Hygiene check 3: weights\ holds weight cards only (no model file ships), and library\ holds only
  # the release catalogue and release-files.json (in cartridges\), MANIFEST.md files, serving job
  # filters, active super-weights baselines, and hardware cartridges.
  Get-ChildItem -LiteralPath (Join-Path $stageBundle 'weights') -Recurse -File | ForEach-Object {
    if (@('part.json', 'README.md') -notcontains $_.Name) {
      throw "Hygiene failure: only weight cards may ship in weights\: $($_.FullName)"
    }
  }
  $libStaged = Join-Path $stageBundle 'library'
  Get-ChildItem -LiteralPath $libStaged -Recurse -File | ForEach-Object {
    $rel = $_.FullName.Substring($libStaged.Length + 1)
    $ok = ($rel -eq 'cartridges\release-catalogue.json') -or ($rel -eq 'cartridges\release-files.json') -or ($_.Name -eq 'MANIFEST.md') -or
      ($rel -like 'parameter-maps\*\job-filter-*.json') -or ($rel -like 'parameter-maps\*\super-weights.json') -or
      ($rel -like 'cartridges\hardware\*')
    if (-not $ok) {
      throw "Hygiene failure: not on the library allowlist: $($_.FullName)"
    }
  }

  # Hygiene check 2: ensure no private local paths or internal repo names appear in text payloads.
  $textExts = @('.md', '.txt', '.json', '.ps1', '.toml', '.yml', '.yaml', '.html', '.js', '.css')
  Get-ChildItem -LiteralPath $stageBundle -Recurse -File | Where-Object { $textExts -contains $_.Extension.ToLowerInvariant() } | ForEach-Object {
    $content = Get-Content -Raw -LiteralPath $_.FullName
    foreach ($pat in $forbiddenPathPatterns) {
      if ($content -match $pat) {
        throw "Hygiene failure: pattern '$pat' matched in $($_.FullName)"
      }
    }
  }

  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
  if (Test-Path -LiteralPath $zipPath) { Remove-Item -Force -LiteralPath $zipPath }
  # 2026-11-25: written entry by entry with forward-slash names. Compress-Archive in Windows
  # PowerShell 5.1 writes backslash entry names, which other unzip tools extract as flat files
  # named with backslashes instead of folders.
  Add-Type -AssemblyName System.IO.Compression
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  $zip = [IO.Compression.ZipFile]::Open($zipPath, [IO.Compression.ZipArchiveMode]::Create)
  try {
    Get-ChildItem -LiteralPath $stageBundle -Recurse -File | ForEach-Object {
      $entry = $bundleName + '/' + ($_.FullName.Substring($stageBundle.Length + 1) -replace '\\', '/')
      [void][IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $_.FullName, $entry, [IO.Compression.CompressionLevel]::Optimal)
    }
  } finally {
    $zip.Dispose()
  }

  $hash = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
  $sumLine = "$hash  $bundleName.zip`n"
  [IO.File]::WriteAllText($sumPath, $sumLine, (New-Object System.Text.UTF8Encoding($false)))

  $sizeMb = [Math]::Round((Get-Item -LiteralPath $zipPath).Length / 1MB, 2)
  Write-Host "Packaged : $zipPath ($sizeMb MiB)"
  Write-Host "SHA-256  : $hash"
  Write-Host "Checksums: $sumPath"
  Write-Host 'Hygiene  : PASS (0 forbidden folders, 0 private path matches)'
} finally {
  Remove-Item -Recurse -Force -LiteralPath $stageRoot -ErrorAction SilentlyContinue
}
