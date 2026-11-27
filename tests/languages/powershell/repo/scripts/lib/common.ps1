# Helpers shared by the build and deploy scripts.

$script:LogPath = Join-Path $env:TEMP 'site-build.log'

function Write-Log {
  param([Parameter(Mandatory)][string]$Message)
  $line = '{0:u} {1}' -f (Get-Date), $Message
  Add-Content -Path $script:LogPath -Value $line
  Write-Host $line
}

function Convert-Markdown {
  param([string]$Text)
  # Headings and paragraphs only; enough for the help pages.
  $out = foreach ($l in $Text -split "`n") {
    if ($l -match '^(#{1,3})\s+(.*)$') {
      $n = $Matches[1].Length
      "<h$n>$($Matches[2])</h$n>"
    } elseif ($l.Trim()) {
      "<p>$l</p>"
    }
  }
  return ($out -join "`n")
}

class DeployTarget {
  [string]$Name
  [string]$Url

  DeployTarget([string]$name, [string]$url) {
    $this.Name = $name
    $this.Url = $url
  }

  [string] Describe() {
    return "$($this.Name) at $($this.Url)"
  }
}

enum DeployMode {
  Preview
  Production
}

function global:Get-DeployTarget([string]$Name) {
  $targets = @{ preview = 'https://preview.example.test'; production = 'https://www.example.test' }
  return [DeployTarget]::new($Name, $targets[$Name])
}
