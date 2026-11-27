param([Alias('Mini')][string]$Root = (Split-Path $PSScriptRoot -Parent))
# -Mini is the parameter's earlier name, still accepted.
# Writes crates/<crate>/part.json for every library crate of the Neuromorph Nano. The program
# crate nmcore and the lab tool nmlab carry no card. Run with no argument from anywhere: the
# Nano folder is the one above this script. Then `neuromorph-nano seal crates/*/part.json` and
# `neuromorph-nano check cards`.
# Built sub-components carry the SHA-256 of their source file and version 1; planned ones
# carry an empty hash and version 0. The card's own "hash" is left empty for `seal`.
$ErrorActionPreference = 'Stop'
function P($name, $kind, $req = $true) { @{ name = $name; kind = $kind; required = $req } }
$crates = [ordered]@{
  nmhash  = @{ kind='other'; parent=$null; deps=@(); subs=@('sha256'); src='in-house engine, Gavriel Technologies'; in=@(); out=@(); cfg='{}' }
  nmjson  = @{ kind='other'; parent=$null; deps=@(); subs=@('json','encode','error','pretty','check'); src='in-house engine, Gavriel Technologies'; in=@(); out=@(); cfg='{}' }
  nmlex   = @{ kind='other'; parent=$null; deps=@(); subs=@('lex','lang','lang_lex'); src='in-house engine, Gavriel Technologies'; in=@(); out=@(); cfg='{}' }
  nmtype  = @{ kind='other'; parent=$null; deps=@('nmjson','nmhash'); subs=@('packet','component','port','card','hop','tensor','task','memory','view','class','message','settings','order','project','doc','budget','examples','wire'); src='in-house engine, Gavriel Technologies'; in=@(); out=@(); cfg='{}' }
  nmdoc   = @{ kind='other'; parent='nmsubs'; deps=@('nmjson','nmtype'); subs=@('format','text','md','csv','markup','html','xml','json','yaml','log','pdf/mod','pdf/lex','pdf/file','pdf/crypt','pdf/filters','pdf/font','pdf/tables','pdf/content','pdf/layout','pdf/structure','pdf/fixtures','general','inflate','raw','tar','zip'); src='in-house engine, Gavriel Technologies'; in=@((P 'file' 'path')); out=@((P 'doc' 'documents')); cfg='{"max_text_bytes":524288}' }
  nmengn  = @{ kind='other'; parent='nmapi'; deps=@('nmjson','nmtype','nmnpcs','nmsafe','nmtokz','nmffi','nmhash'); subs=@('half','cpu','gpu','placement','assembly','model','components','roles','reference','adapter','pmap','decode','kvstore','residency','base','chanrec','drone','filter','tentacle','capacity','device','limits','autotune','speculate'); src='in-house engine, Gavriel Technologies'; in=@((P 'x' 'tensor')); out=@((P 'y' 'tensor')); cfg='{"assembly":"qwen3-reference","population":"qwen3-0.6b"}' }
  nmffi   = @{ kind='other'; parent='nmengn'; deps=@(); subs=@('dynlib','cuda','prof'); src='in-house engine, Gavriel Technologies'; in=@(); out=@(); cfg='{}' }
  nmsafe  = @{ kind='other'; parent='nmengn'; deps=@(); subs=@('tensors','numeric','affine','experts','json','error','check'); src='in-house engine, Gavriel Technologies'; in=@(); out=@(); cfg='{}' }
  nmtokz  = @{ kind='other'; parent='nmengn'; deps=@(); subs=@('model','normalize','split','added','unicode_data','unicode_marks','json','error','check'); src='in-house engine, Gavriel Technologies'; in=@(); out=@(); cfg='{}' }
  nmnpcs  = @{ kind='other'; parent='nmapi'; deps=@('nmjson','nmhash','nmtype'); subs=@('acquire','decompose','compat','provision','lifecycle','parmap','inspect','atlas','compare','cartridge'); src='in-house engine, Gavriel Technologies'; in=@((P 'card' 'json')); out=@((P 'clusters' 'refs')); cfg='{"recording":false}' }
  nmlearn = @{ kind='other'; parent='nmapi'; deps=@('nmjson','nmhash','nmsafe','nmtype','nmbrain'); subs=@('unit','registry','isolation','outcomes','stepspec','forward','rls','readout','adapt','cartridge','preference','route','ledger','calibrate','tuner','learned','engine','hardware','idle','library','pref_fit'); src='in-house engine, Gavriel Technologies'; in=@((P 'outcomes' 'packets')); out=@((P 'report' 'json')); cfg='{"max_run_seconds":60,"quant":"q8"}' }
  nmsubs  = @{ kind='other'; parent='nmapi'; deps=@('nmjson','nmhash','nmlex','nmtype','nmdoc'); subs=@('repo','doc','lock','dense','lens','refgraph','views','general','zoom','sync','semantic','history','execution','interface','constraint','verification','uncertainty','cache','imports','items','piece','tags','tree_shape','units'); src='in-house engine, Gavriel Technologies'; in=@((P 'query' 'text')); out=@((P 'packets' 'packets')); cfg='{"skip":["target","9-weight","golden"]}' }
  nmpod   = @{ kind='neuropod'; parent='nmapi'; deps=@('nmjson','nmlex','nmtype'); subs=@('unit','tip','pipeline','record','recursion','territory','attention','pod','core','evidence','fixture','intake','window'); src='in-house engine, Gavriel Technologies'; in=@((P 'plan' 'json')); out=@((P 'packets' 'packets')); cfg='{"tentacles":8,"tips":8,"max_depth":1}' }
  nmbrain = @{ kind='brain'; parent='nmapi'; deps=@('nmjson','nmlex','nmtype'); subs=@('slot','router','writer','embedder','reranker','arbiter','contradiction','synthesis','rivers','objectives','homeostasis','development','stack','gmap','g0','evidence','g1','g2','chain','instruct','reflex','allocation','motor'); src='in-house engine, Gavriel Technologies'; in=@((P 'packets' 'packets')); out=@((P 'pack' 'packets')); cfg='{"slots":["router","writer","embedder","reranker"]}' }
  nmnte   = @{ kind='other'; parent='nmapi'; deps=@('nmjson','nmtype'); subs=@('graph','snapshot','frame','scheduler','runtime','registry','compose','topology','binding','lifecycle','plasticity','integrity','homeo'); src='in-house engine, Gavriel Technologies'; in=@((P 'task' 'json')); out=@((P 'result' 'any' $false)); cfg='{"workers":0}' }
  nmmem   = @{ kind='memory'; parent='nmapi'; deps=@('nmjson','nmhash','nmlex','nmtype'); subs=@('working','episodic','structural','identity','gate','consolidation','tiers','temp','hierarchy','runday','journal','record','book','retention','classes','pass','recall2','migrate','mem2','evidence','keep','onewriter','outcomes','session_open','store'); src='in-house engine, Gavriel Technologies'; in=@((P 'write' 'packets')); out=@((P 'refs' 'refs')); cfg='{"working_cap":32}' }
  nmtool  = @{ kind='tool'; parent='nmapi'; deps=@('nmjson','nmlex','nmtype'); subs=@('run','lex','build','explain','stepcheck','improve','approval','verify','conformance','scoreboard','conformance/anatomy','conformance/stores','conformance/text'); src='in-house engine, Gavriel Technologies'; in=@((P 'request' 'json')); out=@((P 'packets' 'packets')); cfg='{"scratch":"scratch"}' }
  nmmusc  = @{ kind='other'; parent='nmapi'; deps=@('nmjson','nmhash','nmtype'); subs=@('symbol','log','miner','gain','pack','store','reload','withdraw','measure','priors','config','g3','sites','synth'); src='in-house engine, Gavriel Technologies'; in=@((P 'symbols' 'json')); out=@((P 'control' 'json')); cfg='{"store":"muscle"}' }
  nmapi   = @{ kind='organism'; parent=$null; deps=@('nmjson','nmhash','nmlex','nmtype','nmengn','nmnpcs','nmlearn','nmsubs','nmpod','nmbrain','nmnte','nmmem','nmtool'); subs=@('contract','mcp','tools','forage','anatomy','neural','learn','build','session','delta','assemble','improve','fault','telemetry','coordinator','tentacle_pass','tentacle_core','core_read','joint_read','decide','drone_windows','drone_path','whole','ask','adapter','adapter_fit','distil','brain_side','cache_reads','drone_dendrite','g3','key_scan','reader','request_read','scan_dump','tentacle_body','tentacle_read','tentacle_unit','writer_idle'); src='in-house engine, Gavriel Technologies'; in=@((P 'call' 'json')); out=@((P 'result' 'json')); cfg='{}' }
}
function J([string]$t) { '"' + ($t -replace '\\','\\' -replace '"','\"') + '"' }
function Ports($list) {
  if ($list.Count -eq 0) { return '[]' }
  '[' + (($list | ForEach-Object { '{"name": ' + (J $_.name) + ', "kind": ' + (J $_.kind) + ', "required": ' + ($(if ($_.required) {'true'} else {'false'})) + '}' }) -join ', ') + ']'
}
# The modules a crate's src/lib.rs declares (`mod x;` or `pub mod x;`, test modules left out).
function LibMods([string]$dir) {
  $lib = Join-Path $dir 'src\lib.rs'
  if (-not (Test-Path $lib)) { return @() }
  @(Select-String -LiteralPath $lib -Pattern '^\s*(pub(\([a-z]+\))?\s+)?mod\s+([a-z0-9_]+)\s*;' |
    ForEach-Object { $_.Matches[0].Groups[3].Value } | Where-Object { $_ -ne 'tests' })
}
foreach ($name in $crates.Keys) {
  $c = $crates[$name]
  $dir = Join-Path $Root "crates\$name"
  New-Item -ItemType Directory -Force $dir | Out-Null
  # A module lib.rs declares that the list above does not name yet is added at the end, so a card
  # never misses a built module; the list keeps its order and its planned (not yet built) entries.
  $listed = @($c.subs | ForEach-Object { ($_ -split '/')[0] })
  $new = @(LibMods $dir | Where-Object { $listed -notcontains $_ })
  if ($new.Count -gt 0) {
    Write-Warning ("$name : lib.rs modules not in this script's list, added: " + ($new -join ', ') + ' (add them to the list above)')
    $c.subs = @($c.subs) + $new
  }
  $subs = foreach ($s in $c.subs) {
    # A sub-component is src/<s>.rs; `a/b` names src/a/b.rs (a module folder) and keeps that id.
    $file = Join-Path $dir ('src\' + $s.Replace('/', '\') + '.rs')
    if (Test-Path $file) { $h = (Get-FileHash $file -Algorithm SHA256).Hash.ToLower(); $v = 1 } else { $h = ''; $v = 0 }
    '    {"id": ' + (J "$name.$s") + ', "version": ' + $v + ', "hash": ' + (J $h) + '}'
  }
  $deps = '[' + (($c.deps | ForEach-Object { J $_ }) -join ', ') + ']'
  $parent = if ($c.parent) { J $c.parent } else { 'null' }
  $text = @"
{
  "id": $(J $name),
  "kind": $(J $c.kind),
  "version": 1,
  "parent": $parent,
  "ports_in": $(Ports $c.in),
  "ports_out": $(Ports $c.out),
  "deps": $deps,
  "config_schema": {},
  "config_default": $($c.cfg),
  "subcomponents": [
$($subs -join ",`n")
  ],
  "lineage": [],
  "source": $(J $c.src),
  "hash": ""
}
"@
  [IO.File]::WriteAllText((Join-Path $dir 'part.json'), $text.Replace("`r`n", "`n") + "`n")
  "$name : $($c.subs.Count) sub-components"
}
