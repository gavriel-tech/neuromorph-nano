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
  $plan = Read-Json (Join-Path $Out 'upload-plan.json')
  foreach ($f in $plan.files) {
    if ($f.sha256) {
      $h = (Get-FileHash -LiteralPath $f.local -Algorithm SHA256).Hash.ToLowerInvariant()
      if ($h -ne $f.sha256) { throw "SHA-256 of $($f.local) is $h, not $($f.sha256); nothing more uploaded." }
    }
    & hf upload $Repo $f.local $f.path --repo-type model --commit-message "Add $($f.path)"
    if ($LASTEXITCODE -ne 0) { throw "upload of $($f.path) failed" }
  }
  Write-Host "Published $($plan.files.Count) files to https://huggingface.co/$Repo"
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
Write-Text (Join-Path $Out 'README.md') @"
---
license: other
license_name: polyform-noncommercial-1.0.0
license_link: LICENSE
base_model:
$baseModels
tags:
- neuromorph-nano
- cartridge
- mcp
---

<div align="center">

<img src="banner.jpg" alt="Neuromorph Nano" width="600" />

</div>

# Neuromorph Nano cartridges

The official cartridge library of the [Neuromorph Nano](https://github.com/gavriel-tech/neuromorph-nano),
a local, source-available digital organism that works beside your coding agent, by Luis Jake Gabriel
III (Gavriel Technologies): it reads and remembers your project, serves the context your agent needs,
builds and checks with it, maps parameters and makes its own cartridges, and learns from what
follows, with four MicroModels on your own GPU. Each base cartridge is the set of parameters one MicroModel serves from, derived
from the upstream open weights named below. The Nano downloads them only when the user asks and
checks every file against its SHA-256 before keeping it.

| MicroModel | file | version | size | built from | SHA-256 |
|---|---|---|---|---|---|
$table

Each ``base/<role>/MANIFEST.md`` gives the full SHA-256 and the upstream revision.

## Use

Install [Neuromorph Nano](https://github.com/gavriel-tech/neuromorph-nano#quickstart), then download the base cartridges from its interface (Models -> Downloads) or installer. Neuromorph Nano runs directly on these four base cartridges (~6.5 GB) plus the small tokenizer/config JSONs (~44 MB). The full upstream Qwen3 open weights (~8 GB) are optional and only needed for fallback, output-parity reference checks, parameter mapping, or cartridge surgery. The files are not meant to be loaded by other runtimes.

## Documentation & Contributing

- **Source & Releases:** [github.com/gavriel-tech/neuromorph-nano](https://github.com/gavriel-tech/neuromorph-nano)
- **User Guide:** [docs/user-guide.md](https://github.com/gavriel-tech/neuromorph-nano/blob/main/docs/user-guide.md)
- **Cartridges & Parameter Maps:** [docs/cartridges-and-parameters.md](https://github.com/gavriel-tech/neuromorph-nano/blob/main/docs/cartridges-and-parameters.md)
- **Contributing Cartridges, Maps & Code:** [CONTRIBUTING.md](https://github.com/gavriel-tech/neuromorph-nano/blob/main/CONTRIBUTING.md)
- **Contributor Reward Distribution Pool:** [CONTRIBUTOR_POOL.md](https://github.com/gavriel-tech/neuromorph-nano/blob/main/CONTRIBUTOR_POOL.md)

## License

Source-available, non-commercial. The cartridges in this repository are licensed under the
[PolyForm Noncommercial License 1.0.0](LICENSE), the same license as Neuromorph Nano: free for
personal, academic and other non-commercial use. Commercial use of any kind requires an individually
negotiated commercial license from Luis Jake Gabriel III (Gavriel Technologies); see
[COMMERCIAL.md](https://github.com/gavriel-tech/neuromorph-nano/blob/main/COMMERCIAL.md).

The upstream models are Qwen3 open weights by the Qwen team under the Apache License 2.0. Their
notices are in [NOTICE.md](NOTICE.md) and a copy of that licence is in
[LICENSE-APACHE-2.0](LICENSE-APACHE-2.0).
"@
$notice = ($rows | ForEach-Object { "- [$($_.Repo)](https://huggingface.co/$($_.Repo)), revision ``$($_.Revision)``: Apache License 2.0." }) -join "`n"
Write-Text (Join-Path $Out 'NOTICE.md') @"
# Notices

The base cartridges in this repository are derived from the following open weights, each licensed
under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0):

$notice

Changes made: each cartridge is a separate file built by the Neuromorph Nano's tools from the
upstream weights (parameter selection and restoration per layer, stored in the cartridge format);
the upstream weights themselves are not modified. Copyright of the upstream weights remains with
their authors.
"@
# The license files and banner: the Nano's own LICENSE, the upstream Apache-2.0 text, and banner.jpg.
Copy-Item -LiteralPath (Join-Path $Root 'LICENSE') -Destination (Join-Path $Out 'LICENSE')
Copy-Item -LiteralPath (Join-Path $Root 'docs\images\banner.jpg') -Destination (Join-Path $Out 'banner.jpg')
$apache = Get-ChildItem -LiteralPath (Join-Path $Root 'weights') -Recurse -File -Filter 'LICENSE' | Where-Object { (Get-Content -LiteralPath $_.FullName -TotalCount 3) -match 'Apache License' } | Select-Object -First 1
if (-not $apache) { throw 'No upstream Apache-2.0 LICENSE found under weights\ (download the open weights first).' }
Copy-Item -LiteralPath $apache.FullName -Destination (Join-Path $Out 'LICENSE-APACHE-2.0')
$planFiles = @(
  [ordered]@{ local = (Join-Path $Out 'README.md'); path = 'README.md'; sha256 = $null; bytes = $null },
  [ordered]@{ local = (Join-Path $Out 'banner.jpg'); path = 'banner.jpg'; sha256 = $null; bytes = $null },
  [ordered]@{ local = (Join-Path $Out 'LICENSE'); path = 'LICENSE'; sha256 = $null; bytes = $null },
  [ordered]@{ local = (Join-Path $Out 'LICENSE-APACHE-2.0'); path = 'LICENSE-APACHE-2.0'; sha256 = $null; bytes = $null },
  [ordered]@{ local = (Join-Path $Out 'NOTICE.md'); path = 'NOTICE.md'; sha256 = $null; bytes = $null }
) + ($rows | ForEach-Object { [ordered]@{ local = (Join-Path $Out "base\$($_.RoleDir)\MANIFEST.md"); path = "base/$($_.RoleDir)/MANIFEST.md"; sha256 = $null; bytes = $null } }) + $planFiles
Write-Text (Join-Path $Out 'upload-plan.json') ((@{ repo = $Repo; files = $planFiles } | ConvertTo-Json -Depth 5))

$total = ($rows | Measure-Object -Property Bytes -Sum).Sum
Write-Host ''
Write-Host "Staged      : $Out"
Write-Host "Repository  : https://huggingface.co/$Repo (private until launch; nothing uploaded)"
Write-Host ("Cartridges  : {0} ({1:N2} GB, uploaded from the shelf, not copied)" -f $rows.Count, ($total / 1GB))
Write-Host "Hashes      : $(if ($NoHash) { 'not checked (-NoHash)' } else { 'every cartridge matches designs\base-cartridges.json' })"
Write-Host 'Licence     : PolyForm Noncommercial 1.0.0 (LICENSE), upstream Apache-2.0 notices (NOTICE.md, LICENSE-APACHE-2.0)'
Write-Host ''
Write-Host 'To publish, on the owner''s word only: tools\stage-hf-library.ps1 -Publish -Yes'
