#Requires -Version 5.1
<#
.SYNOPSIS
  Install Neuromorph Nano on Windows (x64) with SHA-256 verification and User PATH setup.
.DESCRIPTION
  Downloads the official release archive and SHA256SUMS.txt from GitHub Releases (or installs
  offline from -LocalZip for air-gapped environments), verifies the archive hash, extracts into
  %LOCALAPPDATA%\NeuromorphNano (or -InstallDir), adds bin\ to the current user's PATH, and
  checks `neuromorph-nano --version`.

  The four released models (about 8 GB) are then fetched from the Hugging Face revisions their
  weight cards pin, and the base cartridges (about 6.6 GB) from the public cartridge library;
  every file is checked against its SHA-256 (INTEGRITY.md, library\cartridges\release-files.json) before it
  is placed, files already present with the right hash are kept, and a partial download resumes.
  Air-gapped machines copy them from a folder instead (-ModelsFrom, -CartridgesFrom). An upgrade
  keeps weights\ and library\ (models, cartridges and the user's own learned cartridges); new
  cartridge versions are merged into the cartridge library, and the Cartridges page's upgrade choice (keep
  both, the default, or replace old) decides what happens to the older base cartridge versions.
.EXAMPLE
  irm https://raw.githubusercontent.com/gavriel-tech/neuromorph-nano/main/tools/install.ps1 | iex
.EXAMPLE
  .\tools\install.ps1 -LocalZip .\dist\neuromorph-nano-v0.1.0-win-x64.zip
.EXAMPLE
  .\tools\install.ps1 -LocalZip .\neuromorph-nano-v0.1.0-win-x64.zip -ModelsFrom E:\nano\weights -CartridgesFrom E:\nano\library
#>
[CmdletBinding()]
param(
  [string]$Version = 'v0.1.0',
  [string]$Repo = 'gavriel-tech/neuromorph-nano',
  [string]$InstallDir = (Join-Path $env:LOCALAPPDATA 'NeuromorphNano'),
  [string]$LocalZip = '',
  [string]$LocalChecksums = '',
  [switch]$NoPath,
  [switch]$SkipVerify,
  # 2026-11-27: by default, install fetches the four base cartridges (~6.5 GB) plus only the small
  # tokenizer/config JSONs (~44 MB) so Neuromorph Nano runs standalone. Pass -WithOpenWeights to
  # also download the optional ~8 GB upstream Qwen3 .safetensors open weights (for fallback,
  # reference parity checks, parameter mapping, and cartridge surgery).
  [switch]$NoModels,
  [switch]$WithOpenWeights,
  [string]$ModelsFrom = '',
  [string]$ModelFilter = '',
  [switch]$NoCartridges,
  [string]$CartridgesFrom = '',
  [string]$CartridgeRepo = 'gavriel-tech/neuromorph-nano-cartridges',
  [string]$CartridgeRevision = 'main'
)

$ErrorActionPreference = 'Stop'

if ([Environment]::Is64BitOperatingSystem -eq $false) {
  throw 'Neuromorph Nano requires 64-bit Windows (x64).'
}

$assetName = "neuromorph-nano-$Version-win-x64.zip"
$stageDir = Join-Path ([IO.Path]::GetTempPath()) ("neuromorph-nano-install-" + [Guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory -Force -Path $stageDir | Out-Null

function Get-ExpectedHash([string]$checksumFile, [string]$fileName) {
  foreach ($line in (Get-Content -LiteralPath $checksumFile)) {
    $trimmed = $line.Trim()
    if (-not $trimmed -or $trimmed.StartsWith('#')) { continue }
    if ($trimmed -match '^([0-9a-fA-F]{64})\s+\*?(.+)$') {
      $hash = $Matches[1].ToLowerInvariant()
      $name = [IO.Path]::GetFileName($Matches[2].Trim())
      if ($name -ieq $fileName) { return $hash }
    }
  }
  return ''
}

function Get-FileSha256([string]$path) {
  (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
}

# One file placed at $dest only once its SHA-256 equals $sha: from the folder $from when given
# (air-gapped installs), else downloaded from $url (curl.exe resumes a partial download; Windows
# PowerShell's Invoke-WebRequest otherwise). A file already present with the right hash is kept.
function Install-VerifiedFile([string]$url, [string]$fromFile, [string]$dest, [string]$sha, [string]$label, [string]$headerFile = '') {
  if ((Test-Path -LiteralPath $dest) -and ((Get-FileSha256 $dest) -eq $sha)) {
    Write-Host "  ok       $label (already present)"
    return
  }
  New-Item -ItemType Directory -Force -Path (Split-Path $dest -Parent) | Out-Null
  $part = "$dest.part"
  if ($fromFile) {
    if (-not (Test-Path -LiteralPath $fromFile)) { throw "$label`: not found in the local folder: $fromFile" }
    Copy-Item -Force -LiteralPath $fromFile -Destination $part
  } else {
    Write-Host "  fetch    $label"
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($curl) {
      $curlArgs = @('-L', '--fail', '--silent', '--show-error', '--retry', '3', '--retry-delay', '2', '-C', '-', '-o', $part)
      if ($headerFile) { $curlArgs += @('-H', "@$headerFile") }
      & $curl.Source @curlArgs $url
      if ($LASTEXITCODE -ne 0) { throw "$label`: download failed (curl exit $LASTEXITCODE): $url" }
    } else {
      $old = $ProgressPreference
      $ProgressPreference = 'SilentlyContinue'
      $hdr = @{}
      if ($headerFile) { $hdr['Authorization'] = ((Get-Content -Raw -LiteralPath $headerFile) -replace '^Authorization:\s*', '').Trim() }
      try { Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $part -Headers $hdr } finally { $ProgressPreference = $old }
    }
  }
  $got = Get-FileSha256 $part
  if ($got -ne $sha) {
    Remove-Item -Force -LiteralPath $part
    throw "$label`: SHA-256 mismatch (expected $sha, got $got); nothing was placed."
  }
  if (Test-Path -LiteralPath $dest) {
    Set-ItemProperty -LiteralPath $dest -Name IsReadOnly -Value $false
    Remove-Item -Force -LiteralPath $dest
  }
  Move-Item -LiteralPath $part -Destination $dest
  Set-ItemProperty -LiteralPath $dest -Name IsReadOnly -Value $true
  Write-Host "  verified $label"
}

function Assert-FreeSpace([string]$dir, [int64]$need, [string]$what) {
  $root = [IO.Path]::GetPathRoot((Resolve-Path -LiteralPath $dir).Path)
  $free = (New-Object IO.DriveInfo($root)).AvailableFreeSpace
  if ($free -lt ($need + 1GB)) {
    throw ("{0} need about {1:N1} GB on {2} and {3:N1} GB is free; free some space or pass -InstallDir on another drive." -f $what, ($need / 1GB), $root, ($free / 1GB))
  }
}

# The four released models, file by file as INTEGRITY.md lists them, from the Hugging Face
# repository and revision each weight card pins (weights\<model>\part.json: origin, revision).
function Install-Models([string]$Root, [string]$Filter, [string]$From) {
  $rows = @()
  foreach ($line in (Get-Content -LiteralPath (Join-Path $Root 'INTEGRITY.md'))) {
    if ($line -match '^\| `models/([^/`]+)/files/([^`]+)` \| `([0-9a-f]{64})` \|') {
      $rows += [pscustomobject]@{ Model = $Matches[1]; File = $Matches[2]; Sha = $Matches[3] }
    }
  }
  if ($rows.Count -eq 0) { throw 'INTEGRITY.md lists no model files.' }
  $rows = @($rows | Where-Object { $_.File -like $Filter })
  $need = [int64]0
  $cards = @{}
  foreach ($m in ($rows | Select-Object -ExpandProperty Model -Unique)) {
    $card = Get-Content -Raw -LiteralPath (Join-Path $Root "weights\$m\part.json") | ConvertFrom-Json
    $cards[$m] = $card.weights
    if ($Filter -eq '*' -and -not (Test-Path -LiteralPath (Join-Path $Root "weights\$m\files\$($card.weights.main_file -replace '^files/', '')"))) {
      $need += [int64]$card.weights.bytes
    }
  }
  if ($need -gt 0) { Assert-FreeSpace $Root $need 'The four models' }
  Write-Host ("Models: {0} files from the pinned Hugging Face revisions, each checked against INTEGRITY.md" -f $rows.Count)
  foreach ($r in $rows) {
    $w = $cards[$r.Model]
    $repo = $w.origin -replace '^https://huggingface\.co/', ''
    $url = "https://huggingface.co/$repo/resolve/$($w.revision)/$($r.File)"
    $fromFile = ''
    if ($From) { $fromFile = Join-Path $From ("$($r.Model)\files\" + ($r.File -replace '/', '\')) }
    $dest = Join-Path $Root ("weights\$($r.Model)\files\" + ($r.File -replace '/', '\'))
    Install-VerifiedFile $url $fromFile $dest $r.Sha "$($r.Model)/$($r.File)"
  }
}

# The base cartridges (library\cartridges\release-files.json), from the public cartridge library on Hugging
# Face. A cartridge that cannot be fetched is a warning, not a failure: the Nano then serves the
# open weights for that MicroModel and says so at start.
function Install-Cartridges([string]$Root, [string]$Repo, [string]$Revision, [string]$From) {
  $listFile = Join-Path $Root 'library\cartridges\release-files.json'
  if (-not (Test-Path -LiteralPath $listFile)) { Write-Warning 'library\cartridges\release-files.json is missing; no base cartridges to fetch.'; return }
  $list = (Get-Content -Raw -LiteralPath $listFile | ConvertFrom-Json).files
  $need = [int64]0
  foreach ($f in $list) {
    if (-not (Test-Path -LiteralPath (Join-Path $Root ("library\" + ($f.path -replace '/', '\'))))) { $need += [int64]$f.bytes }
  }
  if ($need -gt 0) {
    try {
      Assert-FreeSpace $Root $need 'The base cartridges'
    } catch {
      Write-Warning ("{0} The Nano serves the open weights until the cartridges are installed (run this installer again)." -f $_.Exception.Message)
      return
    }
  }
  Write-Host ("Base cartridges: {0} files from {1}, each checked against its SHA-256" -f @($list).Count, $Repo)
  # The official library is public; a private cartridge repository (-CartridgeRepo) needs the user's
  # own Hugging Face token: HF_TOKEN, else the file
  # the hf command line writes at login. It is sent only to huggingface.co, from a header file in
  # the installer's temporary folder (not on a command line), and the file is removed afterwards.
  $headerFile = ''
  $token = $env:HF_TOKEN
  if (-not $token) { $token = $env:HUGGING_FACE_HUB_TOKEN }
  $tokenFile = Join-Path $env:USERPROFILE '.cache\huggingface\token'
  if (-not $token -and (Test-Path -LiteralPath $tokenFile)) { $token = (Get-Content -Raw -LiteralPath $tokenFile).Trim() }
  if ($token -and -not $From) {
    $headerFile = Join-Path $stageDir 'hf-auth.txt'
    [IO.File]::WriteAllText($headerFile, "Authorization: Bearer $token", (New-Object System.Text.UTF8Encoding($false)))
  }
  $missing = @()
  foreach ($f in $list) {
    $dest = Join-Path $Root ("library\" + ($f.path -replace '/', '\'))
    $fromFile = ''
    if ($From) { $fromFile = Join-Path $From ($f.path -replace '/', '\') }
    try {
      $src = $f.path
      if ($f.source) { $src = $f.source }
      Install-VerifiedFile "https://huggingface.co/$Repo/resolve/$Revision/$src" $fromFile $dest $f.sha256 $f.path $headerFile
    } catch {
      $missing += $f.path
      Write-Warning "$($f.path): $($_.Exception.Message)"
    }
  }
  if ($headerFile) { Remove-Item -Force -LiteralPath $headerFile -ErrorAction SilentlyContinue }
  if ($missing.Count -gt 0) {
    $hint = ''
    if (-not $token -and -not $From) { $hint = ' The cartridge repository may need your Hugging Face token: set HF_TOKEN (or log in with the hf command line) and run the installer again.' }
    Write-Warning ("{0} base cartridge(s) not installed; those MicroModels serve their open weights until you run this installer again.{1}" -f $missing.Count, $hint)
  }
}

try {
  $zipPath = ''
  $checksumPath = ''

  if ($LocalZip) {
    $zipPath = (Resolve-Path -LiteralPath $LocalZip).Path
    $assetName = [IO.Path]::GetFileName($zipPath)
    if ($LocalChecksums) {
      $checksumPath = (Resolve-Path -LiteralPath $LocalChecksums).Path
    } else {
      $sibling = Join-Path (Split-Path $zipPath -Parent) 'SHA256SUMS.txt'
      if (Test-Path -LiteralPath $sibling) { $checksumPath = $sibling }
    }
    Write-Host "Installing Neuromorph Nano offline from $zipPath"
  } else {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $baseUrl = "https://github.com/$Repo/releases/download/$Version"
    $zipPath = Join-Path $stageDir $assetName
    $checksumPath = Join-Path $stageDir 'SHA256SUMS.txt'

    Write-Host "Downloading Neuromorph Nano $Version from https://github.com/$Repo ..."
    Invoke-WebRequest -UseBasicParsing -Uri "$baseUrl/$assetName" -OutFile $zipPath
    Invoke-WebRequest -UseBasicParsing -Uri "$baseUrl/SHA256SUMS.txt" -OutFile $checksumPath
  }

  $actualHash = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($checksumPath) {
    $expectedHash = Get-ExpectedHash $checksumPath $assetName
    if (-not $expectedHash) {
      throw "SHA256SUMS.txt does not contain an entry for $assetName."
    }
    if ($actualHash -ne $expectedHash) {
      throw "SHA-256 mismatch for $assetName (expected $expectedHash, got $actualHash). Installation aborted."
    }
    Write-Host ("SHA-256 verified: {0}" -f $actualHash.Substring(0, 16))
  } elseif (-not $LocalZip -or -not $SkipVerify) {
    throw 'SHA256SUMS.txt not found. Pass -LocalChecksums <path> or -SkipVerify for unsigned local builds.'
  } else {
    Write-Warning "Skipping SHA-256 verification (-SkipVerify on local zip): $actualHash"
  }

  $extractDir = Join-Path $stageDir 'unpacked'
  Expand-Archive -LiteralPath $zipPath -DestinationPath $extractDir -Force

  $payloadRoot = $extractDir
  $topItems = @(Get-ChildItem -LiteralPath $extractDir)
  if ($topItems.Count -eq 1 -and $topItems[0].PSIsContainer -and (Test-Path -LiteralPath (Join-Path $topItems[0].FullName 'bin\neuromorph-nano.exe'))) {
    $payloadRoot = $topItems[0].FullName
  }
  if (-not (Test-Path -LiteralPath (Join-Path $payloadRoot 'bin\neuromorph-nano.exe'))) {
    throw 'Archive is missing bin\neuromorph-nano.exe.'
  }

  New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
  # 2026-11-25: program folders are replaced; weights\ and library\ are merged file by file, so an
  # upgrade keeps the downloaded models, the fetched cartridges and the user's own learned
  # cartridges, and an existing library catalogue is never overwritten.
  $replaced = @('bin', 'designs', 'interface', 'tools', 'crates')
  Get-ChildItem -LiteralPath $payloadRoot | ForEach-Object {
    $dest = Join-Path $InstallDir $_.Name
    if ($_.PSIsContainer -and ($replaced -contains $_.Name)) {
      if (Test-Path -LiteralPath $dest) { Remove-Item -Recurse -Force -LiteralPath $dest }
      Copy-Item -Recurse -Force -LiteralPath $_.FullName -Destination $dest
    } elseif ($_.PSIsContainer) {
      $srcRoot = $_.FullName
      Get-ChildItem -LiteralPath $srcRoot -Recurse -File | ForEach-Object {
        $rel = $_.FullName.Substring($srcRoot.Length + 1)
        $target = Join-Path $dest $rel
        New-Item -ItemType Directory -Force -Path (Split-Path $target -Parent) | Out-Null
        if (Test-Path -LiteralPath $target) { Set-ItemProperty -LiteralPath $target -Name IsReadOnly -Value $false }
        Copy-Item -Force -LiteralPath $_.FullName -Destination $target
      }
    } else {
      Copy-Item -Force -LiteralPath $_.FullName -Destination $dest
    }
  }

  if (-not $NoModels) {
    $effectiveFilter = $ModelFilter
    if (-not $effectiveFilter) {
      $effectiveFilter = if ($WithOpenWeights) { '*' } else { '*.json' }
    }
    Install-Models -Root $InstallDir -Filter $effectiveFilter -From $ModelsFrom
    if ($effectiveFilter -eq '*.json') {
      Write-Host 'Note: Installed tokenizer/config JSONs for standalone base-cartridge mode. Pass -WithOpenWeights (or use Models -> Downloads) if you also want the optional ~8 GB upstream open weights for fallback, reference checks, parameter mapping, or surgery.'
    }
  } else {
    Write-Warning 'Model tokenizer/config files skipped (-NoModels): put config.json and tokenizer.json (or full open weights) in weights\ before serving.'
  }
  if (-not $NoCartridges) {
    Install-Cartridges -Root $InstallDir -Repo $CartridgeRepo -Revision $CartridgeRevision -From $CartridgesFrom
  } else {
    Write-Warning 'Base cartridges skipped (-NoCartridges): the Nano serves the open weights until they are installed.'
  }
  # The release's cartridges and stacks join the library: a first install gets its catalogue, an
  # upgrade adds only the new rows; the Cartridges page's upgrade choice (keep both or replace old)
  # decides what happens to older base cartridge versions.
  $releaseCatalogue = Join-Path $InstallDir 'library\cartridges\release-catalogue.json'
  if (Test-Path -LiteralPath $releaseCatalogue) {
    $exeForMerge = Join-Path $InstallDir 'bin\neuromorph-nano.exe'
    & $exeForMerge library merge $releaseCatalogue
    if ($LASTEXITCODE -ne 0) { Write-Warning "The library merge did not finish (exit $LASTEXITCODE); run: neuromorph-nano library merge `"$releaseCatalogue`"" }
  }

  $binDir = Join-Path $InstallDir 'bin'
  $exe = Join-Path $binDir 'neuromorph-nano.exe'

  if (-not $NoPath) {
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $parts = @()
    if ($userPath) { $parts = @($userPath -split ';' | Where-Object { $_ -ne '' }) }
    $already = $false
    foreach ($p in $parts) {
      if ($p.TrimEnd('\') -ieq $binDir.TrimEnd('\')) { $already = $true; break }
    }
    if (-not $already) {
      $newPath = ($parts + $binDir) -join ';'
      [Environment]::SetEnvironmentVariable('Path', $newPath, 'User')
      if ($env:Path -notlike "*$binDir*") {
        $env:Path = "$binDir;$env:Path"
      }
      Write-Host "Added $binDir to your User PATH."
    }
  }

  Write-Host ''
  Write-Host 'Installed Neuromorph Nano:'
  & $exe --version
  Write-Host ''
  Write-Host "Location : $InstallDir"
  Write-Host 'Next steps (open a new terminal if PATH was just updated):'
  Write-Host '  1. Connect your MCP client : neuromorph-nano connect'
  Write-Host '  2. Open the local UI       : neuromorph-nano launch ui'
  Write-Host '  3. Optional air-gap lock   : $env:NANO_AIR_GAP = "on"'
} finally {
  Remove-Item -Recurse -Force -LiteralPath $stageDir -ErrorAction SilentlyContinue
}
