param(
  [string]$Root = (Split-Path $PSScriptRoot -Parent),
  [string]$Repo = '',
  [string]$Exe = '',
  [int]$TimeoutSeconds = 120
)
# MCP smoke test over standard input and output: starts `neuromorph-nano serve --root <repo>
# --memory off`, sends initialize, tools/list and one nm_context_pack call, closes the input and
# checks the three replies: the server names itself neuromorph-nano, nm_context_pack is listed,
# and the context pack cites a file of the fixture repository. Exit code 0 on pass, 1 on fail.
# -Exe runs another build of the program (default: target\release under -Root).
$ErrorActionPreference = 'Stop'
if (-not $Repo) { $Repo = Join-Path $Root 'tests' | Join-Path -ChildPath 'fixtures' | Join-Path -ChildPath 'small-repo' }
$exe = if ($Exe) { $Exe } else { Join-Path $Root 'target\release\neuromorph-nano.exe' }
function Fail([string]$m) { Write-Output "mcp smoke: FAIL $m"; exit 1 }
if (-not (Test-Path $exe)) { Fail "no build at $exe (run cargo build --release)" }
if (-not (Test-Path (Join-Path $Repo 'Cargo.toml'))) { Fail "fixture repository not found: $Repo" }

$question = 'Where is the function take_from_shelf defined?'
$requests = @(
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"mcp-smoke","version":"1"}}}',
  '{"jsonrpc":"2.0","method":"notifications/initialized"}',
  '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}',
  ('{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"nm_context_pack","arguments":{"question":"' + $question + '"}}}')
)

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $exe
$psi.Arguments = 'serve --root "' + $Repo + '" --memory off'
$psi.WorkingDirectory = $Root
$psi.UseShellExecute = $false
$psi.RedirectStandardInput = $true
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.StandardOutputEncoding = [Text.Encoding]::UTF8
$p = [Diagnostics.Process]::Start($psi)
$errTask = $p.StandardError.ReadToEndAsync()
$outTask = $p.StandardOutput.ReadToEndAsync()
foreach ($r in $requests) { $p.StandardInput.WriteLine($r) }
$p.StandardInput.Close()
if (-not $p.WaitForExit($TimeoutSeconds * 1000)) { $p.Kill(); Fail "no exit within $TimeoutSeconds s" }
$out = $outTask.Result
$replies = @{}
foreach ($line in ($out -split "`n")) {
  $t = $line.Trim()
  if (-not $t) { continue }
  try { $j = $t | ConvertFrom-Json } catch { Fail "standard output holds a line that is not JSON: $t" }
  if ($null -ne $j.id) { $replies[[int]$j.id] = $j }
}
foreach ($id in 1, 2, 3) { if (-not $replies.ContainsKey($id)) { Fail "no reply to request $id (stderr: $($errTask.Result.Trim()))" } }
if ($replies[1].result.serverInfo.name -ne 'neuromorph-nano') { Fail "initialize: serverInfo.name is '$($replies[1].result.serverInfo.name)'" }
$names = @($replies[2].result.tools | ForEach-Object { $_.name })
if ($names -notcontains 'nm_context_pack') { Fail "tools/list: nm_context_pack missing ($($names.Count) tools)" }
$text = (@($replies[3].result.content) | ForEach-Object { $_.text }) -join "`n"
if ($replies[3].result.isError) { Fail "nm_context_pack returned an error: $text" }
if ($text -notmatch 'crates/inventory/src/lib\.rs') { Fail "nm_context_pack does not cite crates/inventory/src/lib.rs:`n$text" }
$cite = ([regex]::Match($text, 'crates/inventory/src/lib\.rs:\d+(-\d+)?')).Value
Write-Output "mcp smoke: PASS (serverInfo neuromorph-nano $($replies[1].result.serverInfo.version); $($names.Count) tools; nm_context_pack cites $cite)"
exit 0
