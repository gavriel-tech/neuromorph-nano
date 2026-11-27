# Makes tests/documents/inflate/: original files whose text was written for the test, each
# compressed by Windows' own Deflate (.NET System.IO.Compression.DeflateStream and GZipStream) at
# every level .NET offers (Optimal, Fastest, NoCompression, the last giving stored blocks), a
# zlib stream (the Optimal deflate data with a zlib header and Adler-32 trailer added here),
# corrupt streams and two decompression bombs. Windows PowerShell 5.1, no other program.
# Run: powershell -NoProfile -File tests/documents/gen/make_inflate.ps1
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$base = Join-Path (Split-Path $PSScriptRoot -Parent) 'inflate'
$orig = Join-Path $base 'original'
$out = Join-Path $base 'compressed'
New-Item -ItemType Directory -Force $orig, $out | Out-Null

function Put([string]$path, [byte[]]$bytes) { [System.IO.File]::WriteAllBytes($path, $bytes) }

# Originals. The prose was written for this test; it is repeated with a counter to pass 64 KB, so
# the compressor emits several blocks.
$para = @'
The Harrow Creek pumping station lifts water from the lower basin to the ridge reservoir. Two
pumps run in turn, each for one week, so that wear is shared. The duty pump starts when the
reservoir falls below 3.2 metres and stops at 4.6 metres. Every Monday the operator records the
running hours, the suction and discharge pressures and the motor current in the station log.
A seal that weeps more than ten drops a minute is replaced at the next changeover.
'@
$sb = New-Object System.Text.StringBuilder
for ($i = 1; $i -le 220; $i++) { [void]$sb.AppendLine("Section $i. " + $para.Replace("week", "week ($i)")) }
$prose = [System.Text.Encoding]::UTF8.GetBytes($sb.ToString().Replace("`r`n", "`n"))
Put (Join-Path $orig 'prose.txt') $prose
$rows = New-Object System.Text.StringBuilder
[void]$rows.AppendLine('day,pump,hours,suction_kpa,discharge_kpa,current_a')
for ($d = 1; $d -le 400; $d++) { [void]$rows.AppendLine(('{0},{1},{2},{3},{4},{5}' -f $d, (1 + ($d % 2)), (6 + ($d * 7) % 11), (40 + ($d * 13) % 9), (610 + ($d * 17) % 23), (11 + ($d * 5) % 4))) }
Put (Join-Path $orig 'table.csv') ([System.Text.Encoding]::UTF8.GetBytes($rows.ToString().Replace("`r`n", "`n")))
$bin = New-Object byte[] 70000
for ($i = 0; $i -lt $bin.Length; $i++) { $bin[$i] = [byte](($i * 31 + ($i -shr 7)) % 256) }
Put (Join-Path $orig 'bytes.bin') $bin
Put (Join-Path $orig 'short.txt') ([System.Text.Encoding]::UTF8.GetBytes("Valve V-12 closed at 06:40.`n"))
# No empty file: .NET Framework writes nothing at all for empty input, which is no DEFLATE stream.

function Deflate([byte[]]$data, [System.IO.Compression.CompressionLevel]$level) {
  $ms = New-Object System.IO.MemoryStream
  $ds = New-Object System.IO.Compression.DeflateStream($ms, $level, $true)
  $ds.Write($data, 0, $data.Length); $ds.Close()
  return , $ms.ToArray()
}
function Gzip([byte[]]$data) {
  $ms = New-Object System.IO.MemoryStream
  $gs = New-Object System.IO.Compression.GZipStream($ms, [System.IO.Compression.CompressionLevel]::Optimal, $true)
  $gs.Write($data, 0, $data.Length); $gs.Close()
  return , $ms.ToArray()
}
function Adler([byte[]]$data) {
  [uint32]$a = 1; [uint32]$b = 0
  foreach ($x in $data) { $a = ($a + $x) % 65521; $b = ($b + $a) % 65521 }
  return [uint32](($b * 65536) + $a)
}

foreach ($f in Get-ChildItem $orig -File) {
  $data = [System.IO.File]::ReadAllBytes($f.FullName)
  foreach ($lv in 'Optimal', 'Fastest', 'NoCompression') {
    Put (Join-Path $out ("{0}.{1}.deflate" -f $f.Name, $lv.ToLower())) (Deflate $data ([System.IO.Compression.CompressionLevel]::$lv))
  }
  Put (Join-Path $out ("{0}.gz" -f $f.Name)) (Gzip $data)
  $raw = Deflate $data ([System.IO.Compression.CompressionLevel]::Optimal)
  $ad = Adler $data
  $z = New-Object System.Collections.Generic.List[byte]
  $z.Add(0x78); $z.Add(0x9C); $z.AddRange([byte[]]$raw)
  $z.Add([byte](($ad -shr 24) -band 255)); $z.Add([byte](($ad -shr 16) -band 255)); $z.Add([byte](($ad -shr 8) -band 255)); $z.Add([byte]($ad -band 255))
  Put (Join-Path $out ("{0}.zlib" -f $f.Name)) $z.ToArray()
}

# Corrupt data, each expected to give a named fault.
$bad = Join-Path $base 'corrupt'
New-Item -ItemType Directory -Force $bad | Out-Null
$good = [System.IO.File]::ReadAllBytes((Join-Path $out 'prose.txt.optimal.deflate'))
Put (Join-Path $bad 'truncated.deflate') ($good[0..([int]($good.Length / 2))])
Put (Join-Path $bad 'block-type.deflate') ([byte[]](0x07, 0x00, 0x00))
Put (Join-Path $bad 'stored-length.deflate') ([byte[]](0x01, 0x05, 0x00, 0x05, 0x00, 0x61, 0x62, 0x63, 0x64, 0x65))
Put (Join-Path $bad 'distance.deflate') ([byte[]](0x03, 0x02))
$gz = [System.IO.File]::ReadAllBytes((Join-Path $out 'prose.txt.gz'))
$gz[$gz.Length - 6] = $gz[$gz.Length - 6] -bxor 0x55
Put (Join-Path $bad 'checksum.gz') $gz
$zl = [System.IO.File]::ReadAllBytes((Join-Path $out 'table.csv.zlib'))
$zl[$zl.Length - 1] = $zl[$zl.Length - 1] -bxor 1
Put (Join-Path $bad 'checksum.zlib') $zl
Put (Join-Path $bad 'header.zlib') ([byte[]](0x78, 0x9D, 0x03, 0x00))
Put (Join-Path $bad 'garbage.deflate') ([System.Text.Encoding]::ASCII.GetBytes('this is not deflate data at all, just words'))

# Decompression bombs: 50 MB of zero bytes deflated, and a zip whose one member is 7 MB of zeros.
$bomb = Join-Path $base 'bomb'
New-Item -ItemType Directory -Force $bomb | Out-Null
Put (Join-Path $bomb 'zeros.deflate') (Deflate (New-Object byte[] (50MB)) ([System.IO.Compression.CompressionLevel]::Optimal))
$zp = Join-Path $bomb 'zeros.zip'
if (Test-Path $zp) { Remove-Item $zp }
$zip = [System.IO.Compression.ZipFile]::Open($zp, 'Create')
$e = $zip.CreateEntry('zeros.txt', [System.IO.Compression.CompressionLevel]::Optimal)
$s = $e.Open(); $z7 = New-Object byte[] (7MB); $s.Write($z7, 0, $z7.Length); $s.Close()
$zip.Dispose()
Write-Output "made $base"
