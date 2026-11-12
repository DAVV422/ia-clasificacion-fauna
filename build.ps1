<#
.SYNOPSIS
  Arma los instaladores de WWM Clasificador (HU-DS-12): estándar, completo o los dos, desde UN build --onedir.

.EXAMPLE
  .\build.ps1 -Variant both -Version 0.11.0
  .\build.ps1 -Variant standard -Version 0.11.0 -SkipBuild     # reusa dist\WWMClasificador
  .\build.ps1 -Variant both -Version 0.11.0 -Clean             # borra build\ al terminar (~1,3 GB)
  .\build.ps1 -Variant full -Version 0.11.0 -ModelsFrom C:\modelos   # modelos de otra carpeta (CI)

.NOTES
  1. La versión tiene que coincidir con desktop.__version__.
  2. Regenera installer\requirements.iss desde desktop\requirements_check.REQUIREMENTS (HU-DS-20).
  3. PyInstaller una sola vez + --selftest del ejecutable.
  4. Variante completa: dist\models con los modelos (hardlinks desde los de esta PC) y models.ok (HU-DS-13).
  5. ISCC por variante → dist\installers\, y SHA256SUMS.txt con los tamaños.
#>
param(
    [ValidateSet("standard", "full", "both")] [string] $Variant = "both",
    [Parameter(Mandatory = $true)] [string] $Version,
    [switch] $SkipBuild,
    [switch] $Clean,
    # Carpeta con md_v5a.0.0.pt y speciesnet-4.0.2a\ (p. ej. bajados del release models-v1, en el CI). Sin ella,
    # la variante completa usa los de esta PC (models\ del repo y la caché de Kaggle).
    [string] $ModelsFrom = ""
)
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }

function Step($text) { Write-Host "`n== $text" -ForegroundColor Cyan }
function Check-Exit($what) { if ($LASTEXITCODE -ne 0) { throw "$what falló (código $LASTEXITCODE)" } }

Step "Versión"
$appVersion = (& $Py -c "import desktop; print(desktop.__version__)").Trim()
Check-Exit "leer la versión"
if ($appVersion -ne $Version) { throw "La versión pedida ($Version) no coincide con desktop.__version__ ($appVersion)" }

Step "Requisitos del instalador"
& $Py tools\gen_requirements_iss.py
Check-Exit "gen_requirements_iss"

$Iscc = @(
    $env:ISCC,
    (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"),
    (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
    (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe")
) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if (-not $Iscc) { throw "No se encontró ISCC.exe (Inno Setup 6). Instálalo: winget install JRSoftware.InnoSetup" }

if (-not $SkipBuild) {
    Step "PyInstaller (--onedir)"
    & $Py -m PyInstaller desktop\desktop.spec --noconfirm
    Check-Exit "PyInstaller"
}
$Exe = Join-Path $Root "dist\WWMClasificador\WWMClasificador.exe"
if (-not (Test-Path $Exe)) { throw "No está $Exe (corre sin -SkipBuild)" }

Step "Selftest del ejecutable"
$selftest = & $Exe --selftest | Out-String  # el pipe hace que PowerShell espere a la app de ventana
Write-Host $selftest
if ($LASTEXITCODE -ne 0 -or $selftest -match "FALTA") { throw "El selftest del ejecutable falló" }

$variants = if ($Variant -eq "both") { @("standard", "full") } else { @($Variant) }
if ($variants -contains "full") {
    Step "Modelos para la variante completa (dist\models)"
    $models = Join-Path $Root "dist\models"
    if (Test-Path $models) { Remove-Item $models -Recurse -Force }
    if ($ModelsFrom) {
        & $Py tools\prepare_models.py --out $models --md (Join-Path $ModelsFrom "md_v5a.0.0.pt") --speciesnet (Join-Path $ModelsFrom "speciesnet-4.0.2a")
    } else {
        & $Py tools\prepare_models.py --out $models
    }
    Check-Exit "prepare_models"
    & $Py -c "from pathlib import Path; from desktop import model_store as ms; m = ms.load_manifest(); r = Path(r'$models'); assert ms.needs_install(r, m) == [], 'los modelos no coinciden con el manifiesto'; ms.write_models_ok(r, m); print('models.ok escrito')"
    Check-Exit "models.ok"
}

$outDir = Join-Path $Root "dist\installers"
New-Item -ItemType Directory -Force $outDir | Out-Null
foreach ($v in $variants) {
    Step "Instalador: $v"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & $Iscc /Q "/DVariant=$v" "/DAppVersion=$Version" installer\wwm.iss
    Check-Exit "ISCC ($v)"
    Write-Host ("  listo en {0:n1} min" -f $sw.Elapsed.TotalMinutes)
}

Step "SHA256SUMS.txt"
$lines = Get-ChildItem $outDir -File | Where-Object { $_.Name -like "WWM-Clasificador-*$Version*" } | ForEach-Object {
    $hash = (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower()
    "{0}  {1}  ({2:n0} MB)" -f $hash, $_.Name, ($_.Length / 1MB)
}
$lines | Set-Content -Encoding utf8 (Join-Path $outDir "SHA256SUMS.txt")
$lines | ForEach-Object { Write-Host "  $_" }

if ($Clean -and (Test-Path (Join-Path $Root "build"))) {
    Step "Limpieza de build\"
    Remove-Item (Join-Path $Root "build") -Recurse -Force
}
Write-Host "`nInstaladores en $outDir" -ForegroundColor Green
