param([string]$Root = (Split-Path $PSScriptRoot -Parent))
# Writes INTEGRITY.md: the SHA-256 of every fixed file of the Neuromorph Nano, one table row per
# file, in the form `neuromorph-nano check integrity` reads (a back-quoted path relative to the
# Nano folder, then a back-quoted 64-digit hash). Fixed files: the released model files under
# <models>/<model>/files/ (download-tool metadata in .cache is left out), written as
# `models/<model>/files/<file>`: a path in the models folder, which the check finds the way the
# engine does (NANO_MODELS, else "dir" in designs/models.json, else weights/); the compiled kernels in
# kernels/build/, the engine reference in tests/engine-reference/ (the 0.6B recording through
# its MANIFEST.sha256, which the check expands line by line), and the locked test sets: the code
# lock record tests/dev/code-lock.txt and every file of tests/documents/ (the any-data material,
# fixtures, development and held-out question sets) and of tests/languages/ (the language
# fixtures and their questions). Run it only on purpose, after a
# deliberate change to one of these files; then run `neuromorph-nano check integrity`.
$ErrorActionPreference = 'Stop'
Set-Location $Root
function Row([string]$rel) {
  $h = (Get-FileHash -LiteralPath (Join-Path $Root $rel) -Algorithm SHA256).Hash.ToLower()
  '| `' + ($rel -replace '\\', '/') + '` | `' + $h + '` |'
}
function Rel([string]$full) { $full.Substring($Root.Length).TrimStart('\', '/') }
function ModelsDir {
  if ($env:NANO_MODELS) { return $env:NANO_MODELS }
  $design = Join-Path $Root 'designs\models.json'
  if (Test-Path $design) {
    $d = (Get-Content -Raw $design | ConvertFrom-Json).dir
    if ($d) { if ([IO.Path]::IsPathRooted($d)) { return $d } else { return (Join-Path $Root $d) } }
  }
  Join-Path $Root 'weights'
}
function ModelRow([string]$models, [string]$full) {
  $h = (Get-FileHash -LiteralPath $full -Algorithm SHA256).Hash.ToLower()
  $rel = $full.Substring($models.Length).TrimStart('\', '/') -replace '\\', '/'
  '| `models/' + $rel + '` | `' + $h + '` |'
}

$out = New-Object System.Collections.Generic.List[string]
$out.Add('# INTEGRITY: hashes of the fixed files')
$out.Add('')
$out.Add('SHA-256 of every file the Neuromorph Nano must find unchanged: the released model files, the')
$out.Add('compiled kernels, the engine reference and the locked test sets. `neuromorph-nano check integrity` recomputes each')
$out.Add('hash and reports every difference; a `MANIFEST.sha256` row is also expanded line by line.')
$out.Add('Written by `tools/gen-integrity.ps1`; regenerate only after a deliberate change to one of these')
$out.Add('files. Model rows start with `models/`: the path is in the models folder (`NANO_MODELS`, else')
$out.Add('`dir` in `designs/models.json`), so the check follows the models wherever they are kept. The')
$out.Add('model rows also give `nmnpcs::acquire` the hash of every shard after the first.')

$out.Add('')
$out.Add('## Released model files')
$out.Add('')
$out.Add('| File | SHA-256 |')
$out.Add('|---|---|')
$models = (Resolve-Path (ModelsDir)).Path
Get-ChildItem -LiteralPath $models -Directory | Sort-Object Name | ForEach-Object {
  $files = Join-Path $_.FullName 'files'
  if (Test-Path $files) {
    Get-ChildItem -LiteralPath $files -Recurse -File | Where-Object { $_.FullName -notmatch '[\\/]\.cache[\\/]' } |
      Sort-Object FullName | ForEach-Object { $out.Add((ModelRow $models $_.FullName)) }
  }
}

$out.Add('')
$out.Add('## Compiled kernels')
$out.Add('')
$out.Add('| File | SHA-256 |')
$out.Add('|---|---|')
Get-ChildItem -LiteralPath (Join-Path $Root 'kernels\build') -Filter '*.fatbin' -File | Sort-Object Name |
  ForEach-Object { $out.Add((Row (Rel $_.FullName))) }

$out.Add('')
$out.Add('## Engine reference')
$out.Add('')
$out.Add('| File | SHA-256 |')
$out.Add('|---|---|')
foreach ($rel in @(
    'tests/engine-reference/qwen3-0.6b/MANIFEST.sha256',
    'tests/engine-reference/qwen3-0.6b/tolerance.json',
    'tests/engine-reference/qwen3-1.7b/reference.json',
    'tests/engine-reference/qwen3-1.7b/prompt0-last.f32',
    'tests/engine-reference/qwen3-1.7b/prompt1-last.f32',
    'tests/engine-reference/qwen3-1.7b/prompt2-last.f32',
    'tests/engine-reference/qwen3-1.7b/prompt3-last.f32',
    'tests/engine-reference/role-models/inputs.json',
    'tests/engine-reference/role-models/outputs.json',
    'tests/engine-reference/fingerprint.txt')) {
  $out.Add((Row $rel))
}
$out.Add('')
$out.Add('## Locked test sets')
$out.Add('')
$out.Add('| File | SHA-256 |')
$out.Add('|---|---|')
$out.Add((Row 'tests/dev/code-lock.txt'))
Get-ChildItem -LiteralPath (Join-Path $Root 'tests\documents') -Recurse -File -Force | Sort-Object FullName |
  ForEach-Object { $out.Add((Row (Rel $_.FullName))) }
Get-ChildItem -LiteralPath (Join-Path $Root 'tests\languages') -Recurse -File -Force | Sort-Object FullName |
  ForEach-Object { $out.Add((Row (Rel $_.FullName))) }
[IO.File]::WriteAllText((Join-Path $Root 'INTEGRITY.md'), (($out -join "`n") + "`n"))
"INTEGRITY.md: $(($out | Where-Object { $_ -like '| ``*' }).Count) rows"
