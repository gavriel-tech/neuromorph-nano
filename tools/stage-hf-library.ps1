#Requires -Version 5.1
<#
.SYNOPSIS
  Stages the public Hugging Face cartridge library of the Neuromorph Nano
  (gavriel-tech/neuromorph-nano-cartridges) and, only on -Publish -Yes, uploads it.
.DESCRIPTION
  2026-11-26. The public library holds the base cartridge each MicroModel serves, at the path the
  engine fetches it from (base/<role>/<role>-base.cartridge, nmcore fetch.rs), with a model card, the
  upstream notices and one manifest per role. It is built from designs\base-cartridges.json (the
  serving versions and their SHA-256 sums) and the files on the shelf in library\cartridges\base\.

  Stage (the default): checks every serving cartridge against its SHA-256 (skip with -NoHash), then
  writes into -Out the small files only: README.md (the card), NOTICE.md, base\<role>\MANIFEST.md and
  upload-plan.json. The cartridges (several GB) are never copied; the plan names where each one
  goes in the repository.
  Publish (-Publish -Yes, the owner's call): uploads the small files and each cartridge straight
  from the shelf with the hf command line, one commit per file.
  Licence (Luis Jake Gabriel III, 2026-11-26: "We should be source available non commercial"): the
  library is source-available under the PolyForm Noncommercial License 1.0.0, the Nano's own
  LICENSE, with commercial use by individual agreement (COMMERCIAL.md); the upstream Apache-2.0
  notices and a copy of that licence ship beside it.
.PARAMETER Out
  The staging folder (default: neuromorph-nano-cartridges-public beside this repository).
.PARAMETER Repo
  The Hugging Face repository (default gavriel-tech/neuromorph-nano-cartridges).
#>
[CmdletBinding()]
param(
  [string]$Out = '',
  [string]$Repo = 'gavriel-tech/neuromorph-nano-cartridges',
  [switch]$NoHash,
  [switch]$Force,
  [switch]$Publish,
  [switch]$Yes
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition)
$Root = (Resolve-Path -LiteralPath $Root).Path
if (-not $Out) { $Out = Join-Path (Split-Path -Parent $Root) 'neuromorph-nano-cartridges-public' }
$Out = [IO.Path]::GetFullPath($Out)
$Upstream = @{
  'qwen3-0.6b'           = @{ Repo = 'Qwen/Qwen3-0.6B'; Role = 'Reader' }
  'qwen3-1.7b'           = @{ Repo = 'Qwen/Qwen3-1.7B'; Role = 'Writer' }
  'qwen3-embedding-0.6b' = @{ Repo = 'Qwen/Qwen3-Embedding-0.6B'; Role = 'Embedder' }
  'qwen3-reranker-0.6b'  = @{ Repo = 'Qwen/Qwen3-Reranker-0.6B'; Role = 'Reranker' }
}

function Read-Json([string]$Path) { return (Get-Content -LiteralPath $Path -Raw -Encoding utf8 | ConvertFrom-Json) }
function Write-Text([string]$Path, [string]$Text) {
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
  [IO.File]::WriteAllText($Path, $Text.Replace("`r`n", "`n"), (New-Object Text.UTF8Encoding($false)))
}

if ($Publish) {
  if (-not $Yes) { throw 'Publishing needs -Yes as well: it uploads to Hugging Face.' }
  $planPath = Join-Path $Out 'upload-plan.json'
  $plan = Read-Json $planPath
  foreach ($f in $plan.files) {
    if ($f.sha256) {
      $h = (Get-FileHash -LiteralPath $f.local -Algorithm SHA256).Hash.ToLowerInvariant()
      if ($h -ne $f.sha256) { throw "SHA-256 of $($f.local) is $h, not $($f.sha256); nothing uploaded." }
    }
  }
  $hfPy = Join-Path $env:USERPROFILE '.hf-cli\venv\Scripts\python.exe'
  if (-not (Test-Path -LiteralPath $hfPy)) {
    throw "hf CLI Python runtime not found at $hfPy; install hf CLI or upload with HfApi.create_commit."
  }
  $pyCode = @'
import json, pathlib, sys
from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete

repo = sys.argv[1]
plan = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding='utf-8'))
api = HfApi()
existing = {s.rfilename for s in api.repo_info(repo, repo_type='model').siblings}
ops = [CommitOperationAdd(path_in_repo=f['path'], path_or_fileobj=f['local']) for f in plan['files']]
if 'LICENSE-APACHE-2.0' in existing and not any(f['path'] == 'LICENSE-APACHE-2.0' for f in plan['files']):
    ops.append(CommitOperationDelete(path_in_repo='LICENSE-APACHE-2.0'))
info = api.create_commit(
    repo_id=repo,
    repo_type='model',
    operations=ops,
    commit_message='Update cartridge library: add parameter maps, hardware stack, and third-party license layout',
)
print('Published single atomic commit:', info.commit_url)
'@
  $pyCode | & $hfPy - $Repo $planPath
  if ($LASTEXITCODE -ne 0) { throw "atomic Hugging Face commit failed (exit $LASTEXITCODE)" }
  Write-Host "Published $($plan.files.Count) files in 1 commit to https://huggingface.co/$Repo"
  return
}

# Stage.
if (Test-Path -LiteralPath $Out) {
  if (-not $Force) { throw "$Out exists; run again with -Force to replace it." }
  if (-not (Test-Path -LiteralPath (Join-Path $Out 'upload-plan.json'))) { throw "$Out was not made by this tool; not replacing it." }
  Remove-Item -LiteralPath $Out -Recurse -Force
}
$design = Read-Json (Join-Path $Root 'designs\base-cartridges.json')
$rows = @()
$planFiles = @()
foreach ($p in $design.serving.PSObject.Properties) {
  $model = $p.Name
  $e = $p.Value
  $id = [string]$e.cartridge
  $version = [int]$e.version
  $sha = ([string]$e.sha256).ToLowerInvariant()
  $role = $id -replace '-base$', ''
  $local = Join-Path $Root "library\cartridges\base\$role\$id-v$version.cartridge"
  if (-not (Test-Path -LiteralPath $local)) { throw "Missing on the shelf: $local" }
  $bytes = (Get-Item -LiteralPath $local).Length
  if (-not $NoHash) {
    Write-Host "checking $id@$version ($([math]::Round($bytes / 1GB, 2)) GB)"
    $h = (Get-FileHash -LiteralPath $local -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($h -ne $sha) { throw "SHA-256 of $local is $h, but designs\base-cartridges.json says $sha" }
  }
  $up = $Upstream[$model]
  if (-not $up) { throw "No upstream record for $model" }
  $card = Read-Json (Join-Path $Root "weights\$model\part.json")
  $rev = ($card | ConvertTo-Json -Depth 20 | Select-String -Pattern '"revision":\s*"([0-9a-f]{40})"').Matches[0].Groups[1].Value
  $rows += [pscustomobject]@{ Role = $up.Role; Model = $model; Repo = $up.Repo; Revision = $rev; Id = $id; Version = $version; Bytes = $bytes; Sha = $sha; RoleDir = $role }
  $planFiles += [ordered]@{ local = $local; path = "base/$role/$id.cartridge"; sha256 = $sha; bytes = $bytes }
  Write-Text (Join-Path $Out "base\$role\MANIFEST.md") @"
# $($up.Role) base cartridge

| | |
|---|---|
| file | ``$id.cartridge`` |
| version | $version (``$id@$version``) |
| size | $('{0:N0}' -f $bytes) bytes |
| SHA-256 | ``$sha`` |
| MicroModel | $($up.Role) |
| built from | [$($up.Repo)](https://huggingface.co/$($up.Repo)) at revision ``$rev`` (Apache-2.0) |

The Neuromorph Nano downloads this file when the user asks, checks it against the SHA-256 above
before keeping it, and serves the $($up.Role) MicroModel from it. The upstream open weights are not
changed; this cartridge is a separate file derived from them (see NOTICE.md).
"@
}
$rows = $rows | Sort-Object Role
$table = ($rows | ForEach-Object { "| $($_.Role) | ``base/$($_.RoleDir)/$($_.Id).cartridge`` | $($_.Version) | $('{0:N2}' -f ($_.Bytes / 1GB)) GB | [$($_.Repo)](https://huggingface.co/$($_.Repo)) | ``$($_.Sha.Substring(0, 16))...`` |" }) -join "`n"
$baseModels = ($rows | ForEach-Object { "- $($_.Repo)" }) -join "`n"

function Describe-Artifact([string]$Role, [string]$Name) {
  if ($Name -eq 'job-filter-read.json') { return 'Maps per-layer affine gain and bias sockets across Reader layers for workspace reading and cited span extraction (active while plugged into reader-base).' }
  if ($Name -eq 'job-filter-write.json') { return 'Maps per-layer affine gain and bias sockets across Writer layers for idle L1-L4 world-model note and outline writing (active while plugged into writer-base).' }
  if ($Name -eq 'job-filter-judge-code.json') { return 'Maps per-layer affine gain and bias sockets across Reranker layers for close-candidate code judging and decoy withholding (active while plugged into reranker-base).' }
  if ($Name -eq 'job-filter-code-slice.json') { return "Maps per-layer affine gain and bias sockets across $Role layers for code-slice representation and retrieval (active while plugged into $Role-base)." }
  if ($Name -like 'job-filter-*.json') {
    $job = $Name -replace '^job-filter-', '' -replace '\.json$', ''
    return "Maps per-layer affine gain and bias sockets across $Role layers for the '$job' job (active while plugged in without altering base weights)."
  }
  if ($Name -eq 'super-weights.json') { return "Maps exact [row, col] high-leverage super-weight coordinates across $Role attention and MLP projections pinned so surgery and pruning preserve them." }
  if ($Name -match '^pmap-owl-(\d+)\.json$') {
    $pct = $Matches[1]
    return "Maps per-weight Wanda sensitivity (|w| * ||x||) and OWL outlier-weighted layer budgets across $Role attention and MLP projections ($pct% target sparsity), used to prune and build $Role-base."
  }
  if ($Name -like 'pmap-*.json') { return "Maps per-weight sensitivity and layer sparsity budgets across $Role attention and MLP projections, used to prune and build $Role-base from upstream open weights." }
  return "Maps verified parameter measurements ($Name) across $Role layers."
}

# Parameter maps across the four MicroModel roles: ONLY active/good maps marked outcome = "used"
# in each role's lineage.json (serving job-filter-*.json, active super-weights.json, and current
# pmap-owl-*.json base-build maps). Superseded/experimental files are never staged publicly.
$mapFiles = @()
$mapManifests = @()
$mapRoles = @('embedder', 'reader', 'reranker', 'writer')
# Build privacy patterns at runtime from this machine and checkout remote so this script never
# embeds local paths, usernames, or private repository names in public source.
$privacyParts = New-Object System.Collections.Generic.List[string]
foreach ($lit in @('held-out', 'held_out')) { $privacyParts.Add([regex]::Escape($lit)) }
foreach ($p in @($Root, $env:USERPROFILE)) {
  if ($p) {
    $privacyParts.Add([regex]::Escape($p.TrimEnd('\') + '\'))
    $privacyParts.Add([regex]::Escape(($p.TrimEnd('\') -replace '\\', '/') + '/'))
  }
}
if ($env:USERNAME -and $env:USERNAME.Length -ge 3) {
  $privacyParts.Add('\b' + [regex]::Escape($env:USERNAME) + '\b')
}
try {
  $privUrl = (& git -c safe.directory=* -C $Root remote get-url origin 2>$null | Select-Object -First 1)
  if ($privUrl) {
    $privName = [IO.Path]::GetFileNameWithoutExtension((($privUrl -split '[/:]') | Select-Object -Last 1))
    if ($privName) { $privacyParts.Add([regex]::Escape($privName)) }
  }
} catch { }
$privacyRe = '(?i)(' + ($privacyParts -join '|') + ')'
foreach ($r in $mapRoles) {
  $rDir = Join-Path $Root "library\parameter-maps\$r"
  $lineagePath = Join-Path $rDir 'lineage.json'
  if (-not (Test-Path -LiteralPath $lineagePath)) { continue }
  $lineage = Read-Json $lineagePath
  $goodNames = @()
  foreach ($prop in $lineage.maps.PSObject.Properties) {
    if ([string]$prop.Value.outcome -eq 'used' -and $prop.Name.EndsWith('.json')) {
      $goodNames += $prop.Name
    }
  }
  $roleRows = @()
  $goodNames | Sort-Object | ForEach-Object {
    $name = $_
    $file = Get-Item -LiteralPath (Join-Path $rDir $name) -ErrorAction SilentlyContinue
    if (-not $file -or -not $file.Exists) { return }
    $raw = Get-Content -Raw -LiteralPath $file.FullName
    if ($raw -match $privacyRe) {
      throw "Privacy check failed on parameter map $($file.FullName)"
    }
    $mSha = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    $meta = $lineage.maps.$name
    $desc = Describe-Artifact $r $name
    $purpose = if ($meta -and $meta.purpose) { [string]$meta.purpose } else { 'parameter-map' }
    $madeOn = if ($meta -and $meta.made_on) { [string]$meta.made_on } else { '' }
    $madeBy = if ($meta -and $meta.made_by) { [string]$meta.made_by } else { '' }
    $mapFiles += [pscustomobject]@{ local = $file.FullName; path = "parameter-maps/$r/$($file.Name)"; sha256 = $mSha; bytes = [int64]$file.Length; role = $r; name = $file.Name }
    $roleRows += [pscustomobject]@{ name = $file.Name; purpose = $purpose; made_on = $madeOn; made_by = $madeBy; bytes = [int64]$file.Length; sha = $mSha; desc = $desc }
  }
  if ($roleRows.Count -gt 0) {
    $mPath = Join-Path $Out "parameter-maps\$r\MANIFEST.md"
    $mTable = ($roleRows | ForEach-Object {
      "| ``$($_.name)`` | $($_.purpose) | ``$($_.made_on)`` | $('{0:N0}' -f $_.bytes) | ``$($_.sha)`` |"
    }) -join "`n"
    $mDetails = ($roleRows | ForEach-Object {
      "### ``$($_.name)```n`n- **What it does:** $($_.desc)`n- **Purpose:** $($_.purpose)`n- **Built on:** ``$($_.made_on)```n- **Build command:** ``$($_.made_by)```n- **SHA-256:** ``$($_.sha)``"
    }) -join "`n`n"
    Write-Text $mPath @"
# $r parameter maps & job filters

Active, verified parameter maps and serving job filters for the **$r** MicroModel. Only maps marked ``outcome = "used"`` for the current serving base cartridge are published here.

| File | Purpose | Built on | Bytes | SHA-256 |
|---|---|---|---|---|
$mTable

## Descriptions

$mDetails
"@
    $mapManifests += [ordered]@{ local = $mPath; path = "parameter-maps/$r/MANIFEST.md"; sha256 = $null; bytes = $null }
  }
}
$mapSummary = ($mapRoles | ForEach-Object {
  $r = $_
  $items = @($mapFiles | Where-Object { $_.role -eq $r })
  $sumBytes = [int64]0
  foreach ($it in $items) { $sumBytes += [int64]$it.bytes }
  $mb = [math]::Round(($sumBytes / 1MB), 1)
  "| $r | $($items.Count) | $mb MB | [``parameter-maps/$r/MANIFEST.md``](parameter-maps/$r/MANIFEST.md) |"
}) -join "`n"

# Shareable hardware stack exported via neuromorph-nano library export (non-base cartridges only).
$stackFiles = @()
$nanoExe = Join-Path $Root 'bin\neuromorph-nano.exe'
if (-not (Test-Path -LiteralPath $nanoExe)) { $nanoExe = Join-Path $Root 'target\release\neuromorph-nano.exe' }
if (Test-Path -LiteralPath $nanoExe) {
  $stackOut = Join-Path $Out 'stacks\gpu-nvidia-geforce-gtx-1650-4096mib-v3.stack.json'
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $stackOut) | Out-Null
  & $nanoExe library export 'gpu-nvidia-geforce-gtx-1650-4096mib@3' --out $stackOut | Out-Null
  if ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $stackOut)) {
    $stSha = (Get-FileHash -LiteralPath $stackOut -Algorithm SHA256).Hash.ToLowerInvariant()
    $stBytes = (Get-Item -LiteralPath $stackOut).Length
    $stackFiles += [ordered]@{ local = $stackOut; path = 'stacks/gpu-nvidia-geforce-gtx-1650-4096mib-v3.stack.json'; sha256 = $stSha; bytes = $stBytes }
    $stackManifest = Join-Path $Out 'stacks\MANIFEST.md'
    Write-Text $stackManifest @"
# Shared cartridge stacks

Verified, non-base cartridge stacks that can be fetched from **Models -> Libraries** and imported on trial.

| File | Stack | Target | Bytes | SHA-256 |
|---|---|---|---|---|
| ``gpu-nvidia-geforce-gtx-1650-4096mib-v3.stack.json`` | ``gpu-nvidia-geforce-gtx-1650-4096mib@3`` | NVIDIA GeForce GTX 1650 (4 GB VRAM) | $('{0:N0}' -f $stBytes) | ``$stSha`` |

## Descriptions

### ``gpu-nvidia-geforce-gtx-1650-4096mib-v3.stack.json``

- **What it does:** Hardware tuning stack carrying ``gtx-1650-4gb@3`` settings measured on a 4 GB Turing GPU so all four MicroModels stay co-resident smoothly within 4 GB VRAM. Imports on trial so you can evaluate before activating.
- **SHA-256:** ``$stSha``
"@
    $stackFiles += [ordered]@{ local = $stackManifest; path = 'stacks/MANIFEST.md'; sha256 = $null; bytes = $null }
  }
}

Write-Text (Join-Path $Out 'README.md') @"
---
license: other
license_name: polyform-noncommercial-1.0.0
license_link: LICENSE
thumbnail: https://raw.githubusercontent.com/gavriel-tech/neuromorph-nano/main/docs/images/banner.jpg
base_model:
$baseModels
tags:
- neuromorph-nano
- cartridge
- parameter-maps
- mcp
---

<div align="center">

<img src="banner.jpg" alt="Neuromorph Nano" width="600" />

</div>

# Neuromorph Nano cartridges

The official cartridge and parameter-map library of the [Neuromorph Nano](https://github.com/gavriel-tech/neuromorph-nano),
a local, source-available digital organism that works beside your coding agent, by Luis Jake Gabriel
III (Gavriel Technologies): it reads and remembers your project, serves the context your agent needs,
builds and checks with it, maps parameters and makes its own cartridges, and learns from what
follows, with four MicroModels on your own GPU. Each base cartridge is the set of parameters one
MicroModel serves from, derived from the upstream open weights named below. The Nano downloads them
only when the user asks and checks every file against its SHA-256 before keeping it.

## Base cartridges (``base/``)

| MicroModel | file | version | size | built from | SHA-256 |
|---|---|---|---|---|---|
$table

Each ``base/<role>/MANIFEST.md`` gives the full SHA-256 and the upstream revision.

## Parameter maps & job filters (``parameter-maps/``)

| MicroModel role | files | size | manifest |
|---|---|---|---|
$mapSummary

Includes only the active, verified maps for the serving base cartridges: the 6 serving
``job-filter-*.json`` filters, 4 ``super-weights.json`` surgery baselines, and 4 ``pmap-owl-*.json``
base-build parameter maps. Each role's ``MANIFEST.md`` logs every file's SHA-256, build command,
and description. Download them in one click from **Models -> Downloads** or browse individual files
in **Models -> Libraries**.

## Shared stacks (``stacks/``)

- ``stacks/gpu-nvidia-geforce-gtx-1650-4096mib-v3.stack.json`` - hardware tuning stack for NVIDIA GeForce GTX 1650 (4 GB VRAM). See [``stacks/MANIFEST.md``](stacks/MANIFEST.md).

## Archive layout (``archived/``)

Only active, current, verified files sit under ``base/``, ``parameter-maps/``, and ``stacks/``.
When newer versions replace older ones in future releases, decommissioned files are moved into
``archived/`` mirroring the exact same folder layout (``archived/base/<role>/``,
``archived/parameter-maps/<role>/``, ``archived/stacks/``). The Neuromorph Nano downloader
automatically skips ``archived/`` so users only ever see healthy, up-to-date downloadable content.

## Use & Contributing

Install [Neuromorph Nano](https://github.com/gavriel-tech/neuromorph-nano#quickstart), then download
base cartridges, parameter maps, job filters, and shared stacks directly from its interface
(**Models -> Downloads** and **Models -> Libraries**) or installer. Neuromorph Nano runs directly on
these four base cartridges (about 6.5 GB) plus the small tokenizer/config JSONs (about 44 MB). The
full upstream Qwen3 open weights (about 8 GB) are optional and only needed for fallback,
output-parity reference checks, parameter mapping, or cartridge surgery. You can also publish your
own parameter maps and stacks directly from **Models -> Libraries -> Publish to Hugging Face** (to
your own repository or as a Pull Request here). The files are not meant to be loaded by other runtimes.

## Documentation & Contributing

- **Source & Releases:** [github.com/gavriel-tech/neuromorph-nano](https://github.com/gavriel-tech/neuromorph-nano)
- **User Guide:** [docs/user-guide.md](https://github.com/gavriel-tech/neuromorph-nano/blob/main/docs/user-guide.md)
- **Cartridges & Parameter Maps:** [docs/cartridges-and-parameters.md](https://github.com/gavriel-tech/neuromorph-nano/blob/main/docs/cartridges-and-parameters.md)
- **Contributing Cartridges, Maps & Code:** [CONTRIBUTING.md](https://github.com/gavriel-tech/neuromorph-nano/blob/main/CONTRIBUTING.md)
- **Contributor Reward Distribution Pool:** [CONTRIBUTOR_POOL.md](https://github.com/gavriel-tech/neuromorph-nano/blob/main/CONTRIBUTOR_POOL.md)

## License

Source-available, non-commercial. The cartridges, parameter maps, and stacks in this repository are
licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE), the same license as Neuromorph
Nano: free for personal, academic and other non-commercial use. Commercial use of any kind requires
an individually negotiated commercial license from Luis Jake Gabriel III (Gavriel Technologies); see
[COMMERCIAL.md](https://github.com/gavriel-tech/neuromorph-nano/blob/main/COMMERCIAL.md).

The upstream models are Qwen3 open weights by the Qwen team under the Apache License 2.0. Their
notices are in [NOTICE.md](NOTICE.md) and a copy of that upstream license is kept under
[third_party/APACHE-2.0.txt](third_party/APACHE-2.0.txt).
"@
$notice = ($rows | ForEach-Object { "- [$($_.Repo)](https://huggingface.co/$($_.Repo)), revision ``$($_.Revision)``: Apache License 2.0." }) -join "`n"
Write-Text (Join-Path $Out 'NOTICE.md') @"
# Notices

The base cartridges in this repository are derived from the following open weights, each licensed
under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0; full text in
third_party/APACHE-2.0.txt):

$notice

Changes made: each cartridge is a separate file built by the Neuromorph Nano's tools from the
upstream weights (parameter selection and restoration per layer, stored in the cartridge format);
the upstream weights themselves are not modified. Copyright of the upstream weights remains with
their authors.
"@
# The license files and banner: the Nano's own LICENSE at root, upstream Apache-2.0 text under
# third_party/APACHE-2.0.txt, and banner.jpg.
Copy-Item -LiteralPath (Join-Path $Root 'LICENSE') -Destination (Join-Path $Out 'LICENSE')
Copy-Item -LiteralPath (Join-Path $Root 'docs\images\banner.jpg') -Destination (Join-Path $Out 'banner.jpg')
$apache = Get-ChildItem -LiteralPath (Join-Path $Root 'weights') -Recurse -File -Filter 'LICENSE' | Where-Object { (Get-Content -LiteralPath $_.FullName -TotalCount 3) -match 'Apache License' } | Select-Object -First 1
if (-not $apache) { throw 'No upstream Apache-2.0 LICENSE found under weights\ (download the open weights first).' }
New-Item -ItemType Directory -Force -Path (Join-Path $Out 'third_party') | Out-Null
Copy-Item -LiteralPath $apache.FullName -Destination (Join-Path $Out 'third_party\APACHE-2.0.txt')
$planFiles = @(
  [ordered]@{ local = (Join-Path $Out 'README.md'); path = 'README.md'; sha256 = $null; bytes = $null },
  [ordered]@{ local = (Join-Path $Out 'banner.jpg'); path = 'banner.jpg'; sha256 = $null; bytes = $null },
  [ordered]@{ local = (Join-Path $Out 'LICENSE'); path = 'LICENSE'; sha256 = $null; bytes = $null },
  [ordered]@{ local = (Join-Path $Out 'third_party\APACHE-2.0.txt'); path = 'third_party/APACHE-2.0.txt'; sha256 = $null; bytes = $null },
  [ordered]@{ local = (Join-Path $Out 'NOTICE.md'); path = 'NOTICE.md'; sha256 = $null; bytes = $null }
) + ($rows | ForEach-Object { [ordered]@{ local = (Join-Path $Out "base\$($_.RoleDir)\MANIFEST.md"); path = "base/$($_.RoleDir)/MANIFEST.md"; sha256 = $null; bytes = $null } }) + $planFiles + $mapManifests + ($mapFiles | ForEach-Object { [ordered]@{ local = $_.local; path = $_.path; sha256 = $_.sha256; bytes = $_.bytes } }) + $stackFiles
Write-Text (Join-Path $Out 'upload-plan.json') ((@{ repo = $Repo; files = $planFiles } | ConvertTo-Json -Depth 5))

$total = ($rows | Measure-Object -Property Bytes -Sum).Sum
$mapTotal = [int64]0
foreach ($mf in $mapFiles) { $mapTotal += [int64]$mf.bytes }
Write-Host ''
Write-Host "Staged      : $Out"
Write-Host "Repository  : https://huggingface.co/$Repo (staged locally; nothing uploaded)"
Write-Host ("Cartridges  : {0} ({1:N2} GB, uploaded from the shelf, not copied)" -f $rows.Count, ($total / 1GB))
Write-Host ("Param Maps  : {0} ({1:N1} MB, uploaded from library\parameter-maps\, not copied)" -f $mapFiles.Count, ($mapTotal / 1MB))
Write-Host ("Stacks      : {0}" -f $stackFiles.Count)
Write-Host "Hashes      : $(if ($NoHash) { 'not checked (-NoHash)' } else { 'every cartridge matches designs\base-cartridges.json' })"
Write-Host 'License     : PolyForm Noncommercial 1.0.0 (LICENSE), upstream Apache-2.0 notices (NOTICE.md, third_party/APACHE-2.0.txt)'
Write-Host ''
Write-Host 'To publish, on the owner''s word only: tools\stage-hf-library.ps1 -Publish -Yes'
