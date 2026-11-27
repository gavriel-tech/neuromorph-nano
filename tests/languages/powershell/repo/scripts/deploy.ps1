param([ValidateSet('preview', 'production')][string]$Mode = 'preview')
. "$PSScriptRoot\lib\common.ps1"

# Seconds to wait for the target to answer after the upload.
$HealthTimeoutSec = 30

function Test-Health([string]$Url) {
  $deadline = (Get-Date).AddSeconds($HealthTimeoutSec)
  while ((Get-Date) -lt $deadline) {
    try {
      $r = Invoke-WebRequest -UseBasicParsing -Uri "$Url/health" -TimeoutSec 5
      if ($r.StatusCode -eq 200) { return $true }
    } catch {
      Start-Sleep -Seconds 2
    }
  }
  return $false
}

function Publish-Site([DeployTarget]$Target) {
  Write-Log "publishing to $($Target.Describe())"
  Copy-Item -Recurse -Force 'site/*' (Join-Path 'deploy' $Target.Name)
  if (-not (Test-Health $Target.Url)) {
    throw "target $($Target.Name) did not answer within $HealthTimeoutSec s"
  }
}

Publish-Site (Get-DeployTarget $Mode)
