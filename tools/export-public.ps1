#Requires -Version 5.1
<#
.SYNOPSIS
  Builds the public source tree of the Neuromorph Nano from a commit of the private repository,
  checks it for private material, and prepares the release in the public repository's own folder.
  It never commits and never pushes.
.DESCRIPTION
  2026-11-26 (Luis Jake Gabriel III: "public and private commits and pushes should be active for
  different reasons. The private is the testing and experimental ground and public is public").
  Two repositories, two purposes:
    private  the owner's private repository: every working commit and push, experiments and
             tests; its pre-push hook refuses the public repository.
    public   gavriel-tech/neuromorph-nano: release commits only, one per version, made from this
             tool's folder; its pre-push hook refuses a push unless NANO_PUBLIC_RELEASE names the
             release the newest commit carries.
  Steps, in order; the first failure stops the run:
    1. the public folder (-Out): a clone of the public repository when it has releases, a new
       repository when it has none, or the existing folder for the next release (its files are
       replaced, its history kept, so git shows exactly what the release changes);
    2. git archive of -Ref (committed files only, so nothing ignored or uncommitted can leak),
       leaving out the held-out and real-use test sets (-ExcludeTests) and private workflow tools
       (-ExcludePaths), extracted into the folder;
    3. hygiene: no private folder, no private file, no machine path and no private repository name
       (a failure leaves the folder with no release files in it);
    4. the public pre-push guard, and a summary with the commands for the owner.
.PARAMETER Release
  The release name the commit will carry (for example v0.1.0).
.PARAMETER Out
  The public folder (default: neuromorph-nano-public beside this repository).
.PARAMETER Ref
  The private commit to export (default HEAD).
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)][ValidatePattern('^v\d+\.\d+\.\d+([.-][0-9A-Za-z.-]+)?$')][string]$Release,
  [string]$Out = '',
  [string]$Ref = 'HEAD',
  [string[]]$ExcludeTests = @('dev-large', 'dev-large-2', 'real-use', 'real-use-2', 'real-use-harness'),
  # Private workflow tools: the cartridge backup helper works on the private backup repository.
  # docs/objectives.md is the internal objectives record (targets, measurement sources); public
  # text is curated (Luis Jake Gabriel III, 2026-11-26), so it stays private.
  # The private folder and the private rules loader are tracked in the private repository only.
  [string[]]$ExcludePaths = @('internal', 'CLAUDE.local.md', 'tools/hf-cartridge-backup.ps1', 'docs/objectives.md')
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition)
$Root = (Resolve-Path -LiteralPath $Root).Path
if (-not $Out) { $Out = Join-Path (Split-Path -Parent $Root) 'neuromorph-nano-public' }
$Out = [IO.Path]::GetFullPath($Out)
$PublicRemote = 'https://github.com/gavriel-tech/neuromorph-nano.git'
$Marker = 'neuromorph-public-export'

function Invoke-Git([string[]]$GitArgs, [string]$Dir = $Root) {
  $out = & git -c safe.directory=* -C $Dir @GitArgs 2>&1
  if ($LASTEXITCODE -ne 0) { throw "git $($GitArgs -join ' ') failed: $out" }
  return $out
}
# A git read whose stderr is noise (a credential helper's notice): Windows PowerShell 5.1 would
# otherwise stop on it under ErrorActionPreference Stop.
function Get-GitQuiet([string[]]$GitArgs, [string]$Dir = $Root) {
  $ErrorActionPreference = 'Continue'
  $out = & git -c safe.directory=* -C $Dir @GitArgs 2>$null
  return $out
}
function Clear-Tree([string]$Dir) {
  Get-ChildItem -LiteralPath $Dir -Force | Where-Object { $_.Name -ne '.git' } | Remove-Item -Recurse -Force
}

# 1. The public folder.
$mode = ''
if (Test-Path -LiteralPath $Out) {
  $isPublic = (Test-Path -LiteralPath (Join-Path $Out '.git')) -and ((Get-GitQuiet @('remote', 'get-url', 'origin') $Out) -eq $PublicRemote)
  if (-not $isPublic) { throw "$Out exists and is not the public repository's folder; not touching it." }
  $dirty = Get-GitQuiet @('status', '--porcelain') $Out
  $hasCommit = (Get-GitQuiet @('rev-parse', '--verify', '-q', 'HEAD') $Out)
  if ($hasCommit -and $dirty) { throw "$Out has changes that are not a commit yet; commit them as a release or discard them first." }
  $mode = if ($hasCommit) { 'next release' } else { 'first release' }
} else {
  $remoteHeads = Get-GitQuiet @('ls-remote', '--heads', $PublicRemote)
  if ($remoteHeads) {
    Invoke-Git @('clone', '-q', $PublicRemote, $Out) (Split-Path -Parent $Out) | Out-Null
    $mode = 'next release'
  } else {
    New-Item -ItemType Directory -Force -Path $Out | Out-Null
    Invoke-Git @('init', '-q', '-b', 'main') $Out | Out-Null
    Invoke-Git @('remote', 'add', 'origin', $PublicRemote) $Out | Out-Null
    $mode = 'first release'
  }
}
Clear-Tree $Out

# 2. Archive the private commit and extract it.
$commit = (Invoke-Git @('rev-parse', $Ref)).ToString().Trim()
$tmpDir = Join-Path $Root 'scratch\tmp'
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
$tar = Join-Path $tmpDir "public-export-$($commit.Substring(0, 8)).tar"
$spec = @('archive', '--format=tar', '-o', $tar, $commit, '--', '.') + ($ExcludeTests | ForEach-Object { ":(exclude)tests/$_" }) + ($ExcludePaths | ForEach-Object { ":(exclude)$_" })
Invoke-Git $spec | Out-Null
# Windows' own tar (bsdtar): a GNU tar earlier on the PATH reads 'D:' as a remote host.
& (Join-Path (Join-Path $env:SystemRoot 'System32') 'tar.exe') -xf $tar -C $Out
if ($LASTEXITCODE -ne 0) { throw "tar could not extract $tar" }
Remove-Item -LiteralPath $tar -Force

# 3. Hygiene.
# The private folders of this repository sit at its root; a folder of the same name deeper down (the
# interface's memory page, a test fixture's node_modules) is product or fixture content. internal,
# scratch, private and .claude are refused at any depth.
$forbiddenRootDirs = @('internal', 'scratch', 'runs', 'memory', 'private', 'units', 'muscle', 'target', 'node_modules', '.claude', 'bin', 'library', 'exports')
$forbiddenAnyDirs = @('internal', 'scratch', 'private', '.claude')
$forbiddenFiles = @('claude.local.md', 'interface-agents.md')
$forbiddenRootFiles = @('settings.json', 'cartridge-sources.json')
# Machine paths and the private repository's name, read from this machine at run time (so this
# tool names none of them): the private checkout's folder, the user's profile folder, and the name
# of the private checkout's remote.
$forbiddenText = New-Object System.Collections.Generic.List[string]
foreach ($p in @($Root, $env:USERPROFILE)) {
  if ($p) {
    $forbiddenText.Add($p.TrimEnd('\') + '\')
    $forbiddenText.Add(($p.TrimEnd('\') -replace '\\', '/') + '/')
  }
}
$privateUrl = ''
try { $privateUrl = (& git -C $Root remote get-url origin 2>$null | Select-Object -First 1) } catch { }
if ($privateUrl) {
  $privateName = [IO.Path]::GetFileNameWithoutExtension((($privateUrl -split '[/:]') | Select-Object -Last 1))
  if ($privateName) { $forbiddenText.Add($privateName) }
}
$problems = New-Object System.Collections.Generic.List[string]
Get-ChildItem -LiteralPath $Out -Force -Directory | Where-Object { $_.Name -ne '.git' } | ForEach-Object {
  if ($forbiddenRootDirs -contains $_.Name.ToLowerInvariant()) { $problems.Add("private folder: $($_.Name)") }
}
Get-ChildItem -LiteralPath $Out -Force -File | ForEach-Object {
  if ($forbiddenRootFiles -contains $_.Name.ToLowerInvariant()) { $problems.Add("private file: $($_.Name)") }
}
$tree = Get-ChildItem -LiteralPath $Out -Force | Where-Object { $_.Name -ne '.git' }
$dirs = $tree | Where-Object { $_.PSIsContainer } | ForEach-Object { $_; Get-ChildItem -LiteralPath $_.FullName -Recurse -Force -Directory }
foreach ($d in $dirs) {
  if ($forbiddenAnyDirs -contains $d.Name.ToLowerInvariant()) { $problems.Add("private folder: $($d.FullName.Substring($Out.Length + 1))") }
}
foreach ($t in $ExcludeTests) {
  if (Test-Path -LiteralPath (Join-Path $Out "tests\$t")) { $problems.Add("left-out test set present: tests\$t") }
}
foreach ($p in $ExcludePaths) {
  if (Test-Path -LiteralPath (Join-Path $Out $p)) { $problems.Add("left-out path present: $p") }
}
$textExt = @('.md', '.rs', '.ts', '.tsx', '.mjs', '.js', '.json', '.toml', '.ps1', '.py', '.txt', '.css', '.yml', '.yaml', '.html', '.cmd', '.cu', '.h')
$files = $tree | ForEach-Object { if ($_.PSIsContainer) { Get-ChildItem -LiteralPath $_.FullName -Recurse -Force -File } else { $_ } }
foreach ($f in $files) {
  $rel = $f.FullName.Substring($Out.Length + 1)
  if ($forbiddenFiles -contains $f.Name.ToLowerInvariant()) { $problems.Add("private file: $rel"); continue }
  if ($textExt -notcontains $f.Extension.ToLowerInvariant() -or $f.Length -gt 8MB) { continue }
  $text = [IO.File]::ReadAllText($f.FullName)
  foreach ($p in $forbiddenText) {
    if ($text.IndexOf($p, [StringComparison]::OrdinalIgnoreCase) -ge 0) { $problems.Add("'$p' in $rel") }
  }
}
if ($problems.Count) {
  $problems | ForEach-Object { Write-Host "  $_" }
  # A failed export never sits ready to publish: the release files are taken out again.
  Clear-Tree $Out
  throw "Hygiene failure: $($problems.Count) problem(s); no release files were left in $Out."
}

# 4. The public pre-push guard: a push goes out only for the release the newest commit names.
$hook = @'
#!/bin/sh
# Written by tools/export-public.ps1 (2026-11-26). The public repository receives release commits
# only: set NANO_PUBLIC_RELEASE to the release name the newest commit carries to push it.
msg=$(git log -1 --format=%s 2>/dev/null)
if [ -z "$NANO_PUBLIC_RELEASE" ]; then
  echo "pre-push: refused. Public pushes are releases only: set NANO_PUBLIC_RELEASE to the release name." >&2
  exit 1
fi
case "$msg" in
  *"$NANO_PUBLIC_RELEASE"*) exit 0 ;;
esac
echo "pre-push: refused. The newest commit ('$msg') does not name the release $NANO_PUBLIC_RELEASE." >&2
exit 1
'@
[IO.File]::WriteAllText((Join-Path $Out '.git\hooks\pre-push'), $hook.Replace("`r`n", "`n"), (New-Object Text.UTF8Encoding($false)))
Set-Content -LiteralPath (Join-Path $Out ".git\$Marker") -Value "$Release from $commit at $((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'))" -Encoding ascii

$bytes = ($files | Measure-Object -Property Length -Sum).Sum
$changes = @(Get-GitQuiet @('status', '--porcelain') $Out).Count
Write-Host ''
Write-Host "Release     : $Release ($mode)"
Write-Host "Public tree : $Out"
Write-Host "From commit : $commit (private)"
Write-Host ("Files       : {0} ({1:N1} MiB); {2} paths change against the last public release" -f $files.Count, ($bytes / 1MB), $changes)
Write-Host "Left out    : tests\$($ExcludeTests -join ', tests\'), $($ExcludePaths -join ', ')"
Write-Host 'Hygiene     : PASS (no private folder, file, machine path or private repository name)'
Write-Host "Remote      : $PublicRemote (nothing committed, nothing pushed)"
Write-Host ''
Write-Host 'To publish, on the owner''s word only (review first: git -C <folder> status, diff):'
Write-Host "  git -C `"$Out`" add -A"
Write-Host "  git -C `"$Out`" commit -m `"Release $Release`""
Write-Host "  `$env:NANO_PUBLIC_RELEASE = '$Release'; git -C `"$Out`" push -u origin main; git -C `"$Out`" tag $Release; git -C `"$Out`" push origin $Release"
