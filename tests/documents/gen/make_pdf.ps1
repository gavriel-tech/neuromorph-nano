# Makes tests/documents/pdf/: the test manual printed to PDF by programs that ship with Windows 11.
#  1. python make_pdf.py html: the HTML pages, the plain lines and the expected text of every page
#  2. Microsoft Edge headless (--print-to-pdf): the manual in one column and in two columns, with a
#     running header and footer, and a three-page record whose middle page is a picture only
#  3. Microsoft Print to PDF, driven by .NET System.Drawing.Printing with PrintToFile (no dialog):
#     the same pages as plain lines
#  4. a page count of every printed file from its bytes (each /Type /Page object), against the
#     plan in expected/pdf/pages.tsv
#  5. python make_pdf.py craft: the encrypted copies and the crafted files (marked as crafted)
# Edge runs with its own profile and temporary folder under scratch/, and every process started
# with that profile is ended after each print. Windows PowerShell 5.1.
# Run: powershell -NoProfile -File tests/documents/gen/make_pdf.ps1
$ErrorActionPreference = 'Stop'
$docs = Split-Path $PSScriptRoot -Parent
$root = Split-Path (Split-Path $docs -Parent) -Parent
$out = Join-Path $docs 'pdf'
$src = Join-Path $docs 'expected\pdf\src'
$work = Join-Path $root 'scratch\pdf-make'
$profileDir = Join-Path $work 'edge-profile'
New-Item -ItemType Directory -Force $out, $work, $profileDir | Out-Null
$env:TEMP = $work
$env:TMP = $work

& python (Join-Path $PSScriptRoot 'make_pdf.py') html
if ($LASTEXITCODE -ne 0) { throw 'make_pdf.py html failed' }

$edge = @("${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe", "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe") |
  Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $edge) { throw 'Microsoft Edge not found' }

function Stop-EdgeTree {
  $mark = $profileDir.ToLower()
  Get-CimInstance Win32_Process -Filter "Name = 'msedge.exe'" |
    Where-Object { $_.CommandLine -and $_.CommandLine.ToLower().Contains($mark) } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

function Print-Html([string]$html, [string]$pdf) {
  $target = Join-Path $out $pdf
  if (Test-Path $target) { Remove-Item -Force $target }
  $url = ([System.Uri](Join-Path $src $html)).AbsoluteUri
  $argv = @('--headless', '--disable-gpu', '--no-first-run', '--no-default-browser-check', '--disable-extensions',
    '--disable-sync', '--no-pdf-header-footer', '--generate-pdf-document-outline', "--user-data-dir=`"$profileDir`"",
    "--print-to-pdf=`"$target`"", $url)
  $p = Start-Process -FilePath $edge -ArgumentList $argv -PassThru -WindowStyle Hidden
  if (-not $p.WaitForExit(120000)) { Write-Output "edge: still running after 120 s for $pdf" }
  Stop-EdgeTree
  if (-not (Test-Path $target)) { throw "edge did not write $pdf" }
  Write-Output ("edge: {0} ({1} bytes)" -f $pdf, (Get-Item $target).Length)
}

Print-Html 'manual-1col.html' 'manual-1col.pdf'
Print-Html 'manual-2col.html' 'manual-2col.pdf'
Print-Html 'inspection-scan.html' 'inspection-scan.pdf'

# Microsoft Print to PDF: one page per '--- page N ---' section of the lines file, no dialog.
Add-Type -AssemblyName System.Drawing
$lines = [System.IO.File]::ReadAllLines((Join-Path $src 'manual-msprint.txt'), [System.Text.Encoding]::UTF8)
$pages = New-Object System.Collections.Generic.List[object]
foreach ($l in $lines) {
  if ($l -match '^--- page \d+ ---$') { $pages.Add((New-Object System.Collections.Generic.List[string])) }
  elseif ($pages.Count -gt 0) { $pages[$pages.Count - 1].Add($l) }
}
$msTarget = Join-Path $out 'manual-msprint.pdf'
if (Test-Path $msTarget) { Remove-Item -Force $msTarget }
$doc = New-Object System.Drawing.Printing.PrintDocument
$doc.PrinterSettings.PrinterName = 'Microsoft Print to PDF'
if (-not $doc.PrinterSettings.IsValid) { throw 'Microsoft Print to PDF is not available' }
$doc.PrinterSettings.PrintToFile = $true
$doc.PrinterSettings.PrintFileName = $msTarget
$doc.DocumentName = 'Halvern AQ-40 manual'
$doc.DefaultPageSettings.Margins = New-Object System.Drawing.Printing.Margins(60, 60, 60, 60)
$script:pageAt = 0
$font = New-Object System.Drawing.Font('Arial', 8.5)
$doc.add_PrintPage({
    param($sender, $e)
    $y = [single]$e.MarginBounds.Top
    $h = $font.GetHeight($e.Graphics)
    foreach ($l in $pages[$script:pageAt]) {
      $e.Graphics.DrawString($l, $font, [System.Drawing.Brushes]::Black, [single]$e.MarginBounds.Left, $y)
      $y += $h
    }
    $script:pageAt++
    $e.HasMorePages = $script:pageAt -lt $pages.Count
  })
$doc.Print()
for ($i = 0; $i -lt 60 -and -not (Test-Path $msTarget); $i++) { Start-Sleep -Milliseconds 500 }
Start-Sleep -Seconds 2
if (-not (Test-Path $msTarget)) { throw 'Microsoft Print to PDF did not write its file' }
Write-Output ("print to pdf: manual-msprint.pdf ({0} bytes)" -f (Get-Item $msTarget).Length)

# Page counts from the bytes, against the plan.
$plan = Get-Content (Join-Path $docs 'expected\pdf\pages.tsv') | Select-Object -Skip 1
$bad = 0
foreach ($row in $plan) {
  $f, $want = $row -split "`t"
  $bytes = [System.IO.File]::ReadAllBytes((Join-Path $out $f))
  $text = [System.Text.Encoding]::GetEncoding(28591).GetString($bytes)
  $n = ([regex]::Matches($text, '/Type\s*/Page(?![a-zA-Z])')).Count
  $state = if ($n -eq [int]$want) { 'ok' } else { $bad++; 'DIFFERS' }
  Write-Output ("pages: {0} has {1}, planned {2}: {3}" -f $f, $n, $want, $state)
}
if ($bad -gt 0) { throw "$bad files do not have the planned number of pages" }

& python (Join-Path $PSScriptRoot 'make_pdf.py') craft
if ($LASTEXITCODE -ne 0) { throw 'make_pdf.py craft failed' }
Remove-Item -Recurse -Force (Join-Path $PSScriptRoot '__pycache__') -ErrorAction SilentlyContinue
Write-Output 'done'
