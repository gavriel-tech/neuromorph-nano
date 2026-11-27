# Makes tests/documents/general/: containers and binary files holding text written for the test,
# made with programs that ship with Windows: Compress-Archive and .NET System.IO.Compression
# (zip, gzip), tar.exe (tar, tar with gzip), .NET ResourceWriter (a binary resources file). The
# .docx, .xlsx and .pptx are zips of minimal XML assembled here by hand. Two files are crafted
# byte by byte and say so: legacy.doc (a Compound File Binary signature followed by UTF-16 and
# Windows-1252 text among binary bytes; not a real Word file) and locked.zip (a zip whose
# members are marked encrypted). Everything is invented. Windows PowerShell 5.1.
# Run: powershell -NoProfile -File tests/documents/gen/make_general.ps1
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$root = Split-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) -Parent
$dest = Join-Path (Split-Path $PSScriptRoot -Parent) 'general'
$stage = Join-Path $root 'scratch\general-stage'
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force $dest, $stage | Out-Null
$utf8 = New-Object System.Text.UTF8Encoding($false)

function Put([string]$rel, [string]$text) {
  $p = Join-Path $stage $rel
  New-Item -ItemType Directory -Force (Split-Path $p -Parent) | Out-Null
  [System.IO.File]::WriteAllText($p, $text.Replace("`r`n", "`n"), $utf8)
}
function ZipDir([string]$dir, [string]$name) {
  $tmp = Join-Path $stage ($name + '.zip')
  Compress-Archive -Path (Join-Path $stage "$dir\*") -DestinationPath $tmp -Force
  Copy-Item $tmp (Join-Path $dest $name) -Force
}

# A Word document: the parts a minimal .docx has.
$ct = '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>'
Put 'docx\[Content_Types].xml' $ct
Put 'docx\_rels\.rels' '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'
Put 'docx\docProps\core.xml' '<?xml version="1.0" encoding="UTF-8"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Lantern Ridge Field Station Handbook</dc:title><dc:creator>Station office</dc:creator></cp:coreProperties>'
Put 'docx\word\document.xml' @'
<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
<w:p><w:pPr><w:pStyle w:val="Title"/></w:pPr><w:r><w:t>Lantern Ridge Field Station Handbook</w:t></w:r></w:p>
<w:p><w:r><w:t>Visitors sign the paper register in the boot room before going past the second gate.</w:t></w:r></w:p>
<w:p><w:r><w:t>The generator shed key hangs on the green hook behind the radio desk.</w:t></w:r></w:p>
<w:p><w:r><w:t>Quiet hours begin at ten in the evening and end at six in the morning.</w:t></w:r></w:p>
</w:body></w:document>
'@
ZipDir 'docx' 'handbook.docx'

# An Excel workbook: shared strings and one sheet.
Put 'xlsx\[Content_Types].xml' $ct.Replace('/word/document.xml', '/xl/workbook.xml')
Put 'xlsx\xl\workbook.xml' '<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheets><sheet name="Budget" sheetId="1"/></sheets></workbook>'
Put 'xlsx\xl\sharedStrings.xml' @'
<?xml version="1.0" encoding="UTF-8"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="6" uniqueCount="6">
<si><t>Item</t></si><si><t>Cost</t></si>
<si><t>Replacement anemometer for the north mast</t></si>
<si><t>Annual radio licence renewal</t></si>
<si><t>Snow tyres for the station truck</t></si>
<si><t>Budget note: the anemometer quote from Halden Instruments is valid until 30 April.</t></si>
</sst>
'@
Put 'xlsx\xl\worksheets\sheet1.xml' @'
<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>
<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>
<row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2"><v>1840</v></c></row>
<row r="3"><c r="A3" t="s"><v>3</v></c><c r="B3"><v>310</v></c></row>
<row r="4"><c r="A4" t="s"><v>4</v></c><c r="B4"><v>920</v></c></row>
<row r="6"><c r="A6" t="s"><v>5</v></c></row>
</sheetData></worksheet>
'@
ZipDir 'xlsx' 'budget.xlsx'

# A PowerPoint deck: two slides.
Put 'pptx\[Content_Types].xml' $ct.Replace('/word/document.xml', '/ppt/presentation.xml')
Put 'pptx\ppt\presentation.xml' '<?xml version="1.0" encoding="UTF-8"?><p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:sldIdLst><p:sldId id="256"/><p:sldId id="257"/></p:sldIdLst></p:presentation>'
$slide = '<?xml version="1.0" encoding="UTF-8"?><p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>{0}</a:t></a:r></a:p></p:txBody></p:sp><p:sp><p:txBody><a:p><a:r><a:t>{1}</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>'
Put 'pptx\ppt\slides\slide1.xml' ($slide -f 'Winter Briefing', 'Prepared by the station office for the season ahead.')
Put 'pptx\ppt\slides\slide2.xml' ($slide -f 'Road closures', 'The upper track is closed from the first snowfall until the thaw survey is signed off.')
ZipDir 'pptx' 'briefing.pptx'

# A plain zip of notes, a table and a picture.
Put 'notes\notes\2026-10-03-owl-survey.md' @'
# Owl survey, 3 October 2026

We walked the beech transect from the stile to the old quarry after dusk.
We counted seven tawny owls along the beech transect, two more than last year.
One barn owl was heard near the quarry but not seen.
'@
Put 'notes\data\rain-gauge.csv' @'
date,gauge,mm,note
2026-10-09,G1,3.0,
2026-10-10,G2,12.5,
2026-10-11,G3,41.5,overflow funnel fitted after this reading
2026-10-12,G1,0.0,
'@
New-Item -ItemType Directory -Force (Join-Path $stage 'notes\photos') | Out-Null
[System.IO.File]::WriteAllBytes((Join-Path $stage 'notes\photos\mast.jpg'), [byte[]](0xFF, 0xD8, 0xFF, 0xE0, 0, 16, 74, 70, 73, 70, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0xFF, 0xD9))
ZipDir 'notes' 'field-notes.zip'

# Tar archives made by Windows' tar.exe: a plain tar and a gzip-compressed one.
Put 'tar\logs\generator.log' @'
2026-11-02T06:00:00Z INFO generator started for the morning run
2026-11-02T07:15:00Z WARN generator fuel below 20 percent, refill booked with the valley depot
2026-11-02T09:30:00Z INFO generator stopped, mains restored
'@
Put 'tar\config\station.json' '{"station": "Lantern Ridge", "elevation_m": 1240, "radio_channel": 4}'
$tarexe = Join-Path $env:SystemRoot 'System32\tar.exe'
Push-Location (Join-Path $stage 'tar')
& $tarexe -cf (Join-Path $dest 'logs.tar') logs config
Pop-Location
Put 'tgz\config\sensors.yaml' @'
wind_mast:
  height_m: 12
  calibrated: 2026-09-14
  serial: WM-4471
rain_gauges:
  - G1
  - G2
  - G3
'@
Push-Location (Join-Path $stage 'tgz')
& $tarexe -czf (Join-Path $dest 'station-config.tgz') config
Pop-Location

# A gzip file made by .NET GZipStream.
$memo = $utf8.GetBytes("Memo to all volunteers`n`nThe water filter cartridges are changed every 90 days; the next change is due on 5 January.`nSpare cartridges are on the top shelf of the store.`n")
$ms = New-Object System.IO.MemoryStream
$gz = New-Object System.IO.Compression.GZipStream($ms, [System.IO.Compression.CompressionLevel]::Optimal, $true)
$gz.Write($memo, 0, $memo.Length); $gz.Close()
[System.IO.File]::WriteAllBytes((Join-Path $dest 'memo.txt.gz'), $ms.ToArray())

# A zip inside a zip, made with System.IO.Compression.
function ZipBytes([hashtable]$members) {
  $ms = New-Object System.IO.MemoryStream
  $z = New-Object System.IO.Compression.ZipArchive($ms, [System.IO.Compression.ZipArchiveMode]::Create, $true)
  foreach ($k in ($members.Keys | Sort-Object)) {
    $e = $z.CreateEntry($k, [System.IO.Compression.CompressionLevel]::Optimal)
    $s = $e.Open(); $b = [byte[]]$members[$k]; $s.Write($b, 0, $b.Length); $s.Close()
  }
  $z.Dispose()
  return , $ms.ToArray()
}
$inner = ZipBytes @{ 'inventory.txt' = $utf8.GetBytes("Spare parts held at the station: two fan belts, one water pump impeller, four headlamp bulbs.`n") }
[System.IO.File]::WriteAllBytes((Join-Path $dest 'nested.zip'), (ZipBytes @{ 'readme.txt' = $utf8.GetBytes("The inner archive lists the spare parts.`n"); 'parts/inner.zip' = $inner }))

# A zip whose members are marked encrypted (general purpose flag bit 0 set in the local and
# central headers): crafted, since no program shipped with Windows makes encrypted zips.
$lk = [byte[]](ZipBytes @{ 'passwords-not-here.txt' = $utf8.GetBytes("This text must not be read.`n") })
for ($i = 0; $i -lt $lk.Length - 4; $i++) {
  if ($lk[$i] -eq 0x50 -and $lk[$i + 1] -eq 0x4B -and $lk[$i + 2] -eq 3 -and $lk[$i + 3] -eq 4) { $lk[$i + 6] = $lk[$i + 6] -bor 1 }
  if ($lk[$i] -eq 0x50 -and $lk[$i + 1] -eq 0x4B -and $lk[$i + 2] -eq 1 -and $lk[$i + 3] -eq 2) { $lk[$i + 8] = $lk[$i + 8] -bor 1 }
}
[System.IO.File]::WriteAllBytes((Join-Path $dest 'locked.zip'), $lk)

# A binary .NET resources file written by ResourceWriter (UTF-16 names, UTF-8 values).
$resPath = Join-Path $dest 'strings.resources'
$rw = New-Object System.Resources.ResourceWriter($resPath)
$rw.AddResource('FirstAidLocation', 'The first aid kit is in the red cabinet beside the drying room door.')
$rw.AddResource('StationName', 'Lantern Ridge Field Station')
$rw.Generate(); $rw.Close()

# Crafted: a Compound File Binary signature, binary bytes, then UTF-16 LE and Windows-1252 text.
$rng = New-Object System.Random(7)
$doc = New-Object System.Collections.Generic.List[byte]
$doc.AddRange([byte[]](0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1))
$noise = New-Object byte[] 1500; $rng.NextBytes($noise); $doc.AddRange($noise)
$doc.AddRange([System.Text.Encoding]::Unicode.GetBytes('Radio check schedule: call the valley base at 08:00 and 18:00 on channel 4.'))
$noise = New-Object byte[] 900; $rng.NextBytes($noise); $doc.AddRange($noise)
$doc.AddRange([System.Text.Encoding]::GetEncoding(1252).GetBytes("Caf$([char]0xE9) supplies are ordered on the first Friday of each month."))
$noise = New-Object byte[] 600; $rng.NextBytes($noise); $doc.AddRange($noise)
[System.IO.File]::WriteAllBytes((Join-Path $dest 'legacy.doc'), $doc.ToArray())

# Random bytes with one sentence inside, under a name no reader knows.
$bin = New-Object byte[] 4096; $rng.NextBytes($bin)
$s = [System.Text.Encoding]::ASCII.GetBytes('Calibration constant for sensor T7 is 0.9821.')
[Array]::Copy($s, 0, $bin, 2000, $s.Length)
[System.IO.File]::WriteAllBytes((Join-Path $dest 'unknown.bin'), $bin)
Write-Output "made $dest"
