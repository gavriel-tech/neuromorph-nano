#Requires -Version 5.1
<#
.SYNOPSIS
  Audit surviving production functional code and period payouts for the Neuromorph Nano
  30% Commercial Licensing Contributor Pool (contributors-ledger.json).
.DESCRIPTION
  Uses 100% deterministic folder criticality weights (zero subjective quality multipliers) and
  strictly excludes tests/, .md/.txt files, docs/, code comments, blank/whitespace lines, and
  auto-generated files (part.json, INTEGRITY.md, lockfiles):

    crates/        (Core Rust Engine)            : 2.0x
    kernels/       (CUDA / Compute Kernels)      : 2.0x
    designs/, lib  (Cartridges & Parameter Maps) : 2.0x
    interface/src/ (Functional Local Interface)  : 1.0x
    tools/         (Build & Release Tooling)     : 0.5x
    everything else / docs / comments / tests    : 0.0x

  - With -Range: audits one PR/commit range (exact functional lines, weighted units, ignored
    comment lines, and exact % of the active weighted codebase).
  - With -CodeShare: runs a live surviving git-blame audit across every active production code
    file at HEAD, merges any period-scoped refinement-credit / breakthrough-credit entries from
    contributors-ledger.json (filtered by -Period when specified), and prints ONE unified period
    payout table summing to exactly 100.00% of the 30% Neuromorph Nano licensing profit pool
    (with optional -NetLicensingProfit).
  - With -SyncBaseline: updates v0.1.0-baseline in contributors-ledger.json to live HEAD metrics.
  - With -Record: records a period-scoped refinement-credit, breakthrough-credit, or merged PR
    entry in contributors-ledger.json.
.EXAMPLE
  .\tools\audit-pr-credit.ps1 -Range HEAD~1..HEAD
.EXAMPLE
  .\tools\audit-pr-credit.ps1 -CodeShare -Period 2026-Q4 -NetLicensingProfit 100000
#>
[CmdletBinding()]
param(
  [string]$Root = (Split-Path $PSScriptRoot -Parent),
  [string]$Range = '',
  [switch]$CodeShare,
  [switch]$SyncBaseline,
  [switch]$Record,
  [string]$Period = '',
  [double]$NetLicensingProfit = 0,
  [Nullable[int]]$Pr = $null,
  [string]$Commit = '',
  [string]$Contributor = '',
  [string]$GitHub = '',
  [ValidateSet('rust-engine', 'cuda-kernels', 'cartridge-and-pmap', 'interface-ux', 'tooling', 'refinement-credit', 'breakthrough-credit', 'founding-organism-baseline')]
  [string]$Category = 'rust-engine',
  [int]$WeightedUnits = 0,
  [string]$Summary = '',
  [string]$AuditedBy = 'Luis Jake Gabriel III'
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path -LiteralPath $Root).Path
$ledgerPath = Join-Path $Root 'contributors-ledger.json'

$WEIGHTS = [ordered]@{
  'rust-engine'        = 2.0
  'cuda-kernels'       = 2.0
  'cartridge-and-pmap' = 2.0
  'interface-ux'       = 1.0
  'tooling'            = 0.5
}

function Test-ExcludedPath([string]$p) {
  $norm = $p.Replace('\', '/').ToLowerInvariant()
  if ($norm.EndsWith('.md') -or $norm.EndsWith('.txt')) { return $true }
  if ($norm -eq 'integrity.md' -or $norm -eq 'cargo.lock' -or $norm.EndsWith('package-lock.json')) { return $true }
  if ($norm -match '/part\.json$') { return $true }
  if ($norm.StartsWith('docs/') -or $norm.StartsWith('internal/') -or $norm.StartsWith('tests/') -or $norm.StartsWith('interface/tests/') -or $norm.StartsWith('interface/scripts/')) { return $true }
  return $false
}

function Get-Layer([string]$p) {
  $norm = $p.Replace('\', '/').ToLowerInvariant()
  if ($norm.StartsWith('crates/')) { return 'rust-engine' }
  if ($norm.StartsWith('kernels/')) { return 'cuda-kernels' }
  if ($norm.StartsWith('designs/') -or $norm.StartsWith('library/') -or $norm.EndsWith('.cartridge')) { return 'cartridge-and-pmap' }
  if ($norm.StartsWith('interface/src/')) { return 'interface-ux' }
  if ($norm.StartsWith('tools/')) { return 'tooling' }
  return ''
}

function Test-CommentOnlyLine([string]$line) {
  $t = $line.Trim()
  if (-not $t) { return $true }
  if ($t.StartsWith('//') -or $t.StartsWith('/*') -or $t.StartsWith('*/') -or $t.StartsWith('* ') -or $t -eq '*') { return $true }
  if ($t.StartsWith('#') -and -not $t.StartsWith('#[') -and -not $t.StartsWith('#!') -and -not $t.StartsWith('#include') -and -not $t.StartsWith('#define') -and -not $t.StartsWith('#if') -and -not $t.StartsWith('#endif') -and -not $t.StartsWith('#pragma')) { return $true }
  if ($t.StartsWith('<!--') -or $t.StartsWith('<#') -or $t.StartsWith('#>')) { return $true }
  return $false
}

function Get-ActiveProductionCodebase {
  $tracked = & git -c safe.directory=* -C $Root ls-files
  if ($LASTEXITCODE -ne 0) { throw 'git ls-files failed' }
  $codeExts = @('.rs', '.cu', '.cuh', '.ts', '.tsx', '.js', '.mjs', '.css', '.ps1', '.cmd', '.py')
  $linesByLayer = [ordered]@{}
  $weightedByLayer = [ordered]@{}
  foreach ($k in $WEIGHTS.Keys) { $linesByLayer[$k] = 0; $weightedByLayer[$k] = 0.0 }

  $files = New-Object System.Collections.Generic.List[string]
  foreach ($rel in $tracked) {
    if (-not $rel.Trim()) { continue }
    if (Test-ExcludedPath $rel) { continue }
    $layer = Get-Layer $rel
    if (-not $layer) { continue }
    $ext = [IO.Path]::GetExtension($rel).ToLowerInvariant()
    if ($codeExts -notcontains $ext) { continue }
    $full = Join-Path $Root $rel
    if (-not (Test-Path -LiteralPath $full)) { continue }
    $cnt = 0
    foreach ($ln in [IO.File]::ReadLines($full)) {
      if (-not (Test-CommentOnlyLine $ln)) { $cnt++ }
    }
    $linesByLayer[$layer] += $cnt
    $weightedByLayer[$layer] += ($cnt * $WEIGHTS[$layer])
    $files.Add($rel)
  }
  $totalLines = 0
  $totalWeighted = 0.0
  foreach ($k in $WEIGHTS.Keys) {
    $totalLines += $linesByLayer[$k]
    $totalWeighted += $weightedByLayer[$k]
  }
  return [PSCustomObject]@{
    TotalLines      = $totalLines
    TotalWeighted   = [int][Math]::Round($totalWeighted)
    LinesByLayer    = $linesByLayer
    WeightedByLayer = $weightedByLayer
    Files           = $files
  }
}

$baseCode = $null
$prWeightedAdded = 0

if ($Range) {
  Write-Host "=== Production Functional-Code Diff Audit ($Range) ==="
  $baseCode = Get-ActiveProductionCodebase
  $diffLines = & git -c safe.directory=* -C $Root diff -U0 -w --ignore-blank-lines $Range
  if ($LASTEXITCODE -ne 0) { throw "git diff failed for range $Range" }

  $byLayer = [ordered]@{}
  foreach ($k in $WEIGHTS.Keys) {
    $byLayer[$k] = @{ files = (New-Object System.Collections.Generic.HashSet[string]); add = 0; del = 0; comments = 0 }
  }
  $excludedSet = New-Object System.Collections.Generic.HashSet[string]
  $curPath = ''
  $curLayer = ''

  foreach ($line in $diffLines) {
    if ($line.StartsWith('+++ b/')) {
      $curPath = $line.Substring(6).Trim()
      if (Test-ExcludedPath $curPath) {
        [void]$excludedSet.Add($curPath)
        $curLayer = ''
      } else {
        $curLayer = Get-Layer $curPath
        if (-not $curLayer) { [void]$excludedSet.Add($curPath) }
      }
      continue
    }
    if ($line.StartsWith('+++ ') -or $line.StartsWith('--- ') -or $line.StartsWith('@@') -or $line.StartsWith('diff --git')) { continue }
    if (-not $curLayer) { continue }
    if ($line.StartsWith('+') -or $line.StartsWith('-')) {
      $body = $line.Substring(1)
      if (Test-CommentOnlyLine $body) {
        $byLayer[$curLayer].comments++
      } else {
        [void]$byLayer[$curLayer].files.Add($curPath)
        if ($line.StartsWith('+')) { $byLayer[$curLayer].add++ } else { $byLayer[$curLayer].del++ }
      }
    }
  }

  $substantiveFiles = 0
  $totalAdd = 0
  $totalDel = 0
  $totalIgnoredComments = 0
  $weightedAdd = 0.0
  foreach ($k in $byLayer.Keys) {
    $substantiveFiles += $byLayer[$k].files.Count
    $totalAdd += $byLayer[$k].add
    $totalDel += $byLayer[$k].del
    $totalIgnoredComments += $byLayer[$k].comments
    $weightedAdd += ($byLayer[$k].add * $WEIGHTS[$k])
  }

  $prWeightedAdded = [int][Math]::Round($weightedAdd)
  $poolPct = if ($baseCode.TotalWeighted -gt 0) { ($weightedAdd / $baseCode.TotalWeighted) * 100.0 } else { 0.0 }
  $netDelta = $totalAdd - $totalDel

  Write-Host ("Production files: {0}  |  Excluded docs/tests/generated: {1}  |  Ignored comments: {2}" -f $substantiveFiles, $excludedSet.Count, $totalIgnoredComments)
  Write-Host ("Active production codebase: {0:N0} functional lines ({1:N0} weighted units)" -f $baseCode.TotalLines, $baseCode.TotalWeighted)
  Write-Host ("PR functional delta:        +{0} / -{1} (net {2:+#;-#;0}) = {3:N0} weighted units ({4:N3}% of 30% pool)" -f $totalAdd, $totalDel, $netDelta, $prWeightedAdded, $poolPct)
  foreach ($k in $byLayer.Keys) {
    $row = $byLayer[$k]
    if ($row.files.Count -gt 0) {
      $layerTotal = $baseCode.LinesByLayer[$k]
      $layerPct = if ($layerTotal -gt 0) { ($row.add / $layerTotal) * 100.0 } else { 0.0 }
      Write-Host ("  {0,-20} [{1:N1}x] {2,3} file(s)  +{3} / -{4} lines ({5:N2}% of {0}; {6} comments ignored)" -f $k, $WEIGHTS[$k], $row.files.Count, $row.add, $row.del, $layerPct, $row.comments)
    }
  }
  if ($substantiveFiles -eq 0) {
    Write-Host 'Result: 0 production functional code lines changed (0.000% of 30% pool).'
  }
  Write-Host ''
}

if (-not (Test-Path -LiteralPath $ledgerPath)) {
  throw "Missing ledger: $ledgerPath"
}

$ledger = Get-Content -Raw -LiteralPath $ledgerPath | ConvertFrom-Json

if ($SyncBaseline) {
  if (-not $baseCode) { $baseCode = Get-ActiveProductionCodebase }
  $baseEntry = $ledger.entries | Where-Object { $_.id -eq 'v0.1.0-baseline' } | Select-Object -First 1
  if (-not $baseEntry) { throw 'v0.1.0-baseline entry not found in ledger.' }
  $baseEntry | Add-Member -NotePropertyName 'functional_lines' -NotePropertyValue $baseCode.TotalLines -Force
  $baseEntry | Add-Member -NotePropertyName 'weighted_units' -NotePropertyValue $baseCode.TotalWeighted -Force
  $ledger | Add-Member -NotePropertyName 'metric' -NotePropertyValue 'folder-weighted-surviving-production-code' -Force
  $ledger | Add-Member -NotePropertyName 'folder_weights' -NotePropertyValue ([PSCustomObject]$WEIGHTS) -Force
  $json = $ledger | ConvertTo-Json -Depth 10
  [IO.File]::WriteAllText($ledgerPath, $json.Replace("`r`n", "`n") + "`n", (New-Object System.Text.UTF8Encoding($false)))
  Write-Host ("Synced v0.1.0-baseline to live production codebase: {0:N0} functional lines = {1:N0} weighted units." -f $baseCode.TotalLines, $baseCode.TotalWeighted)
  Write-Host ''
}

if ($Record) {
  if (-not $Contributor.Trim() -or -not $GitHub.Trim()) {
    throw '-Record requires -Contributor "<Full Name>" and -GitHub "<handle>" (human controller only).'
  }
  $finalUnits = if ($WeightedUnits -gt 0) { $WeightedUnits } else { $prWeightedAdded }
  if ($finalUnits -le 0) {
    throw '-Record requires -WeightedUnits > 0 or a -Range with merged production code.'
  }
  if (-not $Summary.Trim()) {
    throw '-Record requires -Summary "<concise description of verified merged work>".'
  }
  $resolvedCommit = if ($Commit) { $Commit } elseif ($Range) { (& git -c safe.directory=* -C $Root rev-parse --short ($Range -split '\.\.')[-1]).Trim() } else { (& git -c safe.directory=* -C $Root rev-parse --short HEAD).Trim() }
  $entryId = if ($null -ne $Pr) { "pr-$Pr-$resolvedCommit" } else { "commit-$resolvedCommit" }
  if (@($ledger.entries | Where-Object { $_.id -eq $entryId }).Count -gt 0) {
    throw "Ledger entry '$entryId' already exists."
  }
  $newEntry = [ordered]@{
    id                 = $entryId
    date               = (Get-Date).ToUniversalTime().ToString('yyyy-MM-dd')
    period             = if ($Period.Trim()) { $Period.Trim() } else { $null }
    pr                 = $Pr
    commit             = $resolvedCommit
    contributor_name   = $Contributor.Trim()
    contributor_github = $GitHub.Trim()
    category           = $Category
    summary            = $Summary.Trim()
    weighted_units     = $finalUnits
    active             = $true
    audited_by         = $AuditedBy
  }
  $entries = New-Object System.Collections.Generic.List[object]
  foreach ($e in $ledger.entries) { $entries.Add($e) }
  $entries.Add([PSCustomObject]$newEntry)
  $ledger.entries = $entries.ToArray()
  $json = $ledger | ConvertTo-Json -Depth 10
  [IO.File]::WriteAllText($ledgerPath, $json.Replace("`r`n", "`n") + "`n", (New-Object System.Text.UTF8Encoding($false)))
  Write-Host "Recorded audited entry: $entryId ($GitHub, +$finalUnits weighted units, $Category)"
  Write-Host ''
}

$poolPct = [double]$ledger.allocation_proposed.contributors_pool_pct
$minThresh = [double]$ledger.minimum_pool_threshold_pct

if ($CodeShare) {
  $periodLabel = if ($Period.Trim()) { "Period $Period" } else { 'Live HEAD' }
  Write-Host "=== Unified Per-Period Payout Audit (${periodLabel}: Surviving Weighted Code + Period Credits) ==="
  if (-not $baseCode) { $baseCode = Get-ActiveProductionCodebase }
  $linesByAuthor = @{}
  $weightedByAuthor = @{}
  $creditsByAuthor = @{}
  $countedLines = 0

  foreach ($rel in $baseCode.Files) {
    $w = $WEIGHTS[(Get-Layer $rel)]
    $blame = & git -c safe.directory=* -C $Root blame -w --line-porcelain HEAD -- $rel 2>$null
    if ($LASTEXITCODE -ne 0) { continue }
    $author = ''
    foreach ($b in $blame) {
      if ($b.StartsWith('author ')) {
        $author = $b.Substring(7).Trim()
      } elseif ($b.StartsWith("`t")) {
        $code = $b.Substring(1)
        if (-not (Test-CommentOnlyLine $code)) {
          if (-not $linesByAuthor.ContainsKey($author)) { $linesByAuthor[$author] = 0; $weightedByAuthor[$author] = 0.0; $creditsByAuthor[$author] = 0 }
          $linesByAuthor[$author]++
          $weightedByAuthor[$author] += $w
          $countedLines++
        }
      }
    }
  }

  # Merge period-scoped refinement-credit and breakthrough-credit entries from contributors-ledger.json.
  $periodCredits = @($ledger.entries | Where-Object {
    $_.active -eq $true -and
    ($_.category -eq 'refinement-credit' -or $_.category -eq 'breakthrough-credit') -and
    ((-not $Period.Trim()) -or ($_.period -eq $Period.Trim()))
  })
  foreach ($c in $periodCredits) {
    $who = $c.contributor_name
    if (-not $linesByAuthor.ContainsKey($who)) { $linesByAuthor[$who] = 0; $weightedByAuthor[$who] = 0.0; $creditsByAuthor[$who] = 0 }
    $u = if ($null -ne $c.weighted_units) { [int]$c.weighted_units } else { [int]$c.functional_lines }
    $creditsByAuthor[$who] += $u
  }

  $grandTotal = 0.0
  foreach ($k in $linesByAuthor.Keys) { $grandTotal += ($weightedByAuthor[$k] + $creditsByAuthor[$k]) }
  if ($grandTotal -le 0) { $grandTotal = 1.0 }
  $eligibleTotal = 0.0
  foreach ($k in $linesByAuthor.Keys) {
    $t = $weightedByAuthor[$k] + $creditsByAuthor[$k]
    if ((($t / $grandTotal) * 100.0) -ge $minThresh) { $eligibleTotal += $t }
  }
  if ($eligibleTotal -le 0) { $eligibleTotal = $grandTotal }
  $poolDollars = $NetLicensingProfit * ($poolPct / 100.0)

  $rows = foreach ($k in $linesByAuthor.Keys) {
    $wu = [int][Math]::Round($weightedByAuthor[$k])
    $cr = [int]$creditsByAuthor[$k]
    $tot = $wu + $cr
    $rawShare = ($tot / $grandTotal) * 100.0
    $isElig = $rawShare -ge $minThresh
    $payoutShare = if ($isElig) { ($tot / $eligibleTotal) * 100.0 } else { 0.0 }
    $shareOfProfit = if ($isElig) { ($tot / $eligibleTotal) * $poolPct } else { 0.0 }
    $obj = [ordered]@{
      Contributor    = $k
      SurvivingLines = [int]$linesByAuthor[$k]
      WeightedCode   = $wu
      PeriodCredits  = $cr
      TotalUnits     = $tot
      PoolShare      = ('{0:N2}%' -f $payoutShare)
      ProfitShare    = ('{0:N2}%' -f $shareOfProfit)
      Eligible       = if ($isElig) { 'yes' } else { "below $minThresh%" }
    }
    if ($NetLicensingProfit -gt 0) {
      $obj['PeriodPayout'] = if ($isElig) { ('${0:N2}' -f ($poolDollars * ($tot / $eligibleTotal))) } else { '$0.00' }
    }
    [PSCustomObject]$obj
  }
  $rows | Sort-Object -Property TotalUnits -Descending | Format-Table -AutoSize
  Write-Host ("Total surviving production code: {0:N0} lines across {1} files  |  Total period units: {2:N0} (always = 100.00% of the {3}% pool)" -f $countedLines, $baseCode.Files.Count, [int][Math]::Round($grandTotal), $poolPct)
  if ($NetLicensingProfit -gt 0) {
    Write-Host ("Net Neuromorph Nano licensing profit: `${0:N2}  |  30% Contributor Pool: `${1:N2}" -f $NetLicensingProfit, $poolDollars)
  }
  Write-Host ''
  return
}

Write-Host '=== Published Ledger Snapshot (30% Neuromorph Nano Licensing Pool) ==='
$active = @($ledger.entries | Where-Object { $_.active -eq $true })
$totalUnits = 0
foreach ($e in $active) {
  $totalUnits += if ($null -ne $e.weighted_units) { [int]$e.weighted_units } else { [int]$e.functional_lines }
}
if ($totalUnits -le 0) { $totalUnits = 1 }

$grouped = $active | Group-Object -Property contributor_github | ForEach-Object {
  $u = 0
  foreach ($e in $_.Group) { $u += if ($null -ne $e.weighted_units) { [int]$e.weighted_units } else { [int]$e.functional_lines } }
  $name = $_.Group[0].contributor_name
  $shareOfPool = ($u / $totalUnits) * 100.0
  $shareOfTotal = ($u / $totalUnits) * $poolPct
  [PSCustomObject]@{
    GitHub        = $_.Name
    Contributor   = $name
    WeightedUnits = $u
    PoolShare     = ('{0:N2}%' -f $shareOfPool)
    OverallEst    = ('{0:N2}%' -f $shareOfTotal)
    Eligible      = if ($shareOfPool -ge $minThresh) { 'yes' } else { "below $minThresh%" }
  }
} | Sort-Object -Property WeightedUnits -Descending

$grouped | Format-Table -AutoSize
Write-Host ("Total active weighted units: {0:N0}  |  Active entries: {1}  |  Policy: {2}" -f $totalUnits, $active.Count, $ledger.policy)
