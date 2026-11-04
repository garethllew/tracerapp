<#
.SYNOPSIS
  Regenerates the winget manifests for a Tracer release from the templates.

.DESCRIPTION
  Downloads Tracer_<version>_x64-setup.exe from the GitHub release v<version>,
  computes its SHA-256 and writes the three manifest files into
  packaging/winget/manifests/<version>/.

  The release must be PUBLISHED (not a draft), because winget needs the public
  URL. Use -InstallerPath to hash a local file instead of downloading (for
  testing the script or for a not-yet-published build).

  This does not submit anything. Submission is a manual step, see PACKAGING.md.

.EXAMPLE
  ./update-manifest.ps1 -Version 0.1.0
#>
param(
  [Parameter(Mandatory = $true)][ValidatePattern('^\d+\.\d+\.\d+$')][string]$Version,
  [string]$InstallerPath
)
$ErrorActionPreference = 'Stop'

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$templates = Join-Path $here 'templates'
$outDir = Join-Path $here "manifests/$Version"
$name = "Tracer_${Version}_x64-setup.exe"
$url = "https://github.com/garethllew/tracerapp/releases/download/v$Version/$name"

$tmp = $null
if ($InstallerPath) {
  $file = (Resolve-Path $InstallerPath).Path
} else {
  $tmp = Join-Path ([IO.Path]::GetTempPath()) "tracer-winget-$([guid]::NewGuid())"
  New-Item -ItemType Directory -Path $tmp | Out-Null
  $file = Join-Path $tmp $name
  Write-Host "Downloading $url"
  Invoke-WebRequest -Uri $url -OutFile $file -UseBasicParsing
}

try {
  $sha = (Get-FileHash $file -Algorithm SHA256).Hash.ToUpper()
  Write-Host "SHA256: $sha"

  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
  foreach ($t in Get-ChildItem $templates -Filter *.yaml) {
    $text = (Get-Content $t.FullName -Raw).Replace('{{VERSION}}', $Version).Replace('{{SHA256}}', $sha)
    # LF endings, UTF-8 without BOM.
    [IO.File]::WriteAllText((Join-Path $outDir $t.Name), $text.Replace("`r`n", "`n"), (New-Object Text.UTF8Encoding $false))
  }
  Write-Host "Wrote manifests to $outDir"
  Write-Host "Next: winget validate --manifest `"$outDir`", then (owner) wingetcreate submit `"$outDir`""
} finally {
  if ($tmp) { Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue }
}
