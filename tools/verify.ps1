param(
  [string]$Root = (Split-Path $PSScriptRoot -Parent),
  [switch]$SkipInterface,
  [switch]$Card
)
# Release checks of the Neuromorph Nano, in order; stops at the first failure and prints a summary.
#  1. cargo build --release with zero warnings
#  2. clippy on the product crates (every target, warnings denied)
#  3. product tests (cargo test --release, the lab tool excluded)
#  4. section checks of every crate (nmengn and nmffi open the graphics card: card steps)
#  5. check cards, anatomy, designs, integrity
#  6. MCP smoke test over standard input and output (tools/mcp-smoke.ps1)
#  7. engine self-test against tests/engine-reference (selftest engine; a card step)
#  8. interface build and lint (npm run build, npx eslint src)
# Everything is built into its own target folder, scratch\gate\target, so no running program of
# another build locks it, and no process is stopped. Card steps run only with -Card and only when
# no other process holds the graphics card; otherwise they are reported as skipped, never passed.
$ErrorActionPreference = 'Continue'
Set-Location $Root
$target = Join-Path $Root 'scratch\gate\target'
$tmp = Join-Path $Root 'scratch\gate\tmp'
New-Item -ItemType Directory -Force $target, $tmp | Out-Null
$env:CARGO_TARGET_DIR = $target
$env:TEMP = $tmp
$env:TMP = $tmp
$exe = Join-Path $target 'release\neuromorph-nano.exe'
$crates = 'nmhash nmjson nmtype nmnte nmlex nmsubs nmpod nmbrain nmmem nmnpcs nmffi nmsafe nmtokz nmengn nmlearn nmtool nmmusc nmapi nmcore'.Split(' ')
$cardCrates = @('nmengn', 'nmffi')
$results = New-Object System.Collections.Generic.List[object]
$started = Get-Date

function Summary {
  Write-Output ''
  Write-Output 'step                         result  seconds'
  foreach ($r in $results) { Write-Output ('{0,-28} {1,-7} {2,7:N1}' -f $r.Step, $r.Result, $r.Seconds) }
  Write-Output ('total {0:N1} s' -f ((Get-Date) - $started).TotalSeconds)
}
function Step([string]$name, [scriptblock]$body) {
  Write-Output ''
  Write-Output "== $name"
  $t = Get-Date
  # The body prints its details and returns its verdict last; only a literal $true passes.
  $res = @(& $body)
  $ok = ($res.Count -gt 0) -and ($res[-1] -is [bool]) -and $res[-1]
  if ($res.Count -gt 1) { $res[0..($res.Count - 2)] | ForEach-Object { Write-Output $_ } }
  $results.Add([pscustomobject]@{ Step = $name; Result = $(if ($ok) { 'ok' } else { 'FAIL' }); Seconds = ((Get-Date) - $t).TotalSeconds })
  if (-not $ok) {
    Summary
    Write-Output "verify: FAIL at '$name'"
    exit 1
  }
}
function Skip([string]$name, [string]$why) {
  Write-Output ''
  Write-Output "== $name"
  Write-Output "skipped: $why"
  $results.Add([pscustomobject]@{ Step = $name; Result = 'skipped'; Seconds = 0.0 })
}
function Native([string]$file, [string[]]$argv) {
  $out = & $file @argv 2>&1 | ForEach-Object { "$_" }
  [pscustomobject]@{ Code = $LASTEXITCODE; Lines = @($out) }
}
# Why the card steps cannot run now, or '' when they can: -Card given and no process on the card.
function CardBlocked {
  if (-not $Card) { return 'no -Card switch' }
  $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
  if (-not $smi) { return 'nvidia-smi not found, the card cannot be checked' }
  $apps = @(& $smi.Source '--query-compute-apps=pid,process_name' '--format=csv,noheader' 2>$null | Where-Object { $_.Trim() })
  if ($LASTEXITCODE -ne 0) { return 'nvidia-smi failed, the card cannot be checked' }
  if ($apps.Count -gt 0) { return "the card is in use by: $($apps -join '; ')" }
  return ''
}

# A program left running from this gate's own folder would lock its build; it is reported, not stopped.
$held = @(Get-Process neuromorph-nano -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $exe })
if ($held.Count -gt 0) {
  Write-Output "verify: FAIL: neuromorph-nano from $exe is running (pid $(($held | ForEach-Object { $_.Id }) -join ', ')); verify stops no process"
  exit 1
}

Step 'build' {
  $r = Native 'cargo' @('build', '--release', '--workspace', '--message-format', 'short')
  $warn = @($r.Lines | Where-Object { $_ -match '(^|: )(warning|error)(\[|:)' })
  $r.Lines | Select-Object -Last 3 | ForEach-Object { Write-Output $_ }
  $warn | ForEach-Object { Write-Output $_ }
  Write-Output "warnings and errors: $($warn.Count)"
  ($r.Code -eq 0) -and ($warn.Count -eq 0)
}
Step 'clippy (product crates)' {
  $r = Native 'cargo' @('clippy', '--release', '--workspace', '--exclude', 'nmlab', '--all-targets', '--message-format', 'short', '--', '-D', 'warnings')
  $r.Lines | Where-Object { $_ -match 'warning|error' } | ForEach-Object { Write-Output $_ }
  $r.Lines | Select-Object -Last 1 | ForEach-Object { Write-Output $_ }
  $r.Code -eq 0
}
Step 'tests (product crates)' {
  $r = Native 'cargo' @('test', '--release', '--workspace', '--exclude', 'nmlab')
  $sum = @($r.Lines | Where-Object { $_ -match '^test result:' })
  $failed = @($r.Lines | Where-Object { $_ -cmatch '(FAILED|panicked at)' })
  $failed | ForEach-Object { Write-Output $_ }
  $passed = ($sum | ForEach-Object { if ($_ -match '(\d+) passed') { [int]$Matches[1] } else { 0 } } | Measure-Object -Sum).Sum
  $ignored = ($sum | ForEach-Object { if ($_ -match '(\d+) ignored') { [int]$Matches[1] } else { 0 } } | Measure-Object -Sum).Sum
  Write-Output "$($sum.Count) test binaries: $passed passed, $ignored ignored, $($failed.Count) failure lines"
  $r.Code -eq 0
}
$blocked = CardBlocked
$checked = if ($blocked) { @($crates | Where-Object { $cardCrates -notcontains $_ }) } else { $crates }
Step 'section checks' {
  $r = Native $exe (@('check') + $checked)
  $okc = @($r.Lines | Where-Object { $_ -like 'ok   *' }).Count
  $r.Lines | Where-Object { $_ -like 'FAIL*' } | ForEach-Object { Write-Output $_ }
  $r.Lines | Select-Object -Last 1 | ForEach-Object { Write-Output $_ }
  Write-Output "section checks ok: $okc ($($checked.Count) crates)"
  $r.Code -eq 0
}
if ($blocked) { Skip 'section checks (card crates)' "$($cardCrates -join ', '): $blocked" }
Step 'cards, anatomy, designs, integrity, languages' {
  $r = Native $exe @('check', 'cards', 'anatomy', 'designs', 'integrity', 'languages')
  $r.Lines | ForEach-Object { Write-Output $_ }
  $r.Code -eq 0
}
Step 'MCP smoke test' {
  $r = Native 'powershell' @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $Root 'tools\mcp-smoke.ps1'), '-Root', $Root, '-Exe', $exe)
  $r.Lines | ForEach-Object { Write-Output $_ }
  $r.Code -eq 0
}
# Checked again: the card may have been taken while the steps above ran.
$blocked = CardBlocked
if ($blocked) {
  Skip 'selftest engine' $blocked
} else {
  Step 'selftest engine' {
    $r = Native $exe @('selftest', 'engine')
    $r.Lines | Where-Object { $_ -match 'PASS|FAIL|MISS|differs|recorded|graphics card' -and $_ -notmatch '^\s+(cpu|gpu) ' } | ForEach-Object { Write-Output $_ }
    $r.Code -eq 0
  }
}
if (-not $SkipInterface) {
  Step 'interface build' {
    Push-Location (Join-Path $Root 'interface')
    $r = Native 'npm' @('run', 'build')
    Pop-Location
    $r.Lines | Select-Object -Last 4 | ForEach-Object { Write-Output $_ }
    $r.Code -eq 0
  }
  Step 'interface lint' {
    Push-Location (Join-Path $Root 'interface')
    $r = Native 'npx' @('eslint', 'src')
    Pop-Location
    $r.Lines | ForEach-Object { Write-Output $_ }
    Write-Output "eslint exit code $($r.Code)"
    $r.Code -eq 0
  }
}
Summary
$skipped = @($results | Where-Object { $_.Result -eq 'skipped' } | ForEach-Object { $_.Step })
if ($skipped.Count -gt 0) {
  Write-Output "verify: PASS with steps skipped, not passed: $($skipped -join '; ')"
} else {
  Write-Output 'verify: PASS'
}
exit 0
