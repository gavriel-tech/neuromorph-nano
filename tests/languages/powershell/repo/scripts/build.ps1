param(
  [string]$Configuration = 'Release',
  [switch]$Clean
)
<#
  Builds the site into ./site and writes a manifest of what was built.
  function Not-Real { } stays inside this block comment.
#>
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\lib\common.ps1"

# Pages larger than this many kilobytes fail the build.
$MaxPageKb = 512

function Remove-SiteOutput {
  param([string]$Path)
  if (Test-Path $Path) {
    Remove-Item -Recurse -Force $Path
    Write-Log "removed $Path"
  }
}

function Build-Page([string]$Source, [string]$Target) {
  $html = Convert-Markdown -Text (Get-Content -Raw $Source)
  $sizeKb = [math]::Ceiling($html.Length / 1024)
  if ($sizeKb -gt $MaxPageKb) {
    throw "page $Source is $sizeKb KB, over the $MaxPageKb KB limit"
  }
  Set-Content -Path $Target -Value $html -Encoding utf8
  return $sizeKb
}

function Write-Manifest {
  param([object[]]$Pages, [string]$Path)
  $body = @"
built: $(Get-Date -Format o)
configuration: $Configuration
pages: $($Pages.Count)
"@
  Set-Content -Path $Path -Value $body -Encoding utf8
}

if ($Clean) { Remove-SiteOutput -Path 'site' }
New-Item -ItemType Directory -Force 'site' | Out-Null
$pages = foreach ($md in Get-ChildItem 'pages' -Filter *.md) {
  $target = Join-Path 'site' ($md.BaseName + '.html')
  [pscustomobject]@{ Page = $md.Name; Kb = (Build-Page $md.FullName $target) }
}
Write-Manifest -Pages $pages -Path 'site/manifest.txt'
