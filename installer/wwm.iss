; WWM Clasificador de cámaras trampa — instalador (HU-DS-12, E-01, E-13, E-D2, E-D3).
;
; Un solo script, dos variantes del MISMO build (dist\WWMClasificador):
;   ISCC /DVariant=standard /DAppVersion=0.11.0 installer\wwm.iss   → WWM-Clasificador-Setup-0.11.0.exe
;   ISCC /DVariant=full     /DAppVersion=0.11.0 installer\wwm.iss   → WWM-Clasificador-Completo-0.11.0.exe
; La completa lleva además models\ con models.ok (HU-DS-13): nunca descarga nada.
; Normalmente lo llama build.ps1, que antes regenera requirements.iss.

#ifndef Variant
  #define Variant "standard"
#endif
#ifndef AppVersion
  #error Falta /DAppVersion=x.y.z (usa build.ps1)
#endif
#if Variant == "full"
  #define OutputName "WWM-Clasificador-Completo-" + AppVersion
  #define VariantLabel "completa (sin internet)"
#else
  #define OutputName "WWM-Clasificador-Setup-" + AppVersion
  #define VariantLabel "estándar (descarga los modelos la primera vez)"
#endif
#define AppName "WWM Clasificador"
#define AppExe "WWMClasificador.exe"
#define Root ".."

#include "requirements.iss"

[Setup]
; AppId fijo: una versión nueva reemplaza a la anterior y conserva %LOCALAPPDATA%\WWM (E-01).
AppId={{6C1E2B57-4F0A-4E0B-9B7E-2D5A9E3C1F40}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=CETERIS404 para WWF Bolivia
AppPublisherURL=https://github.com/DAVV422/ia-clasificacion-fauna
AppSupportURL=https://github.com/DAVV422/ia-clasificacion-fauna
VersionInfoVersion={#AppVersion}
VersionInfoDescription={#AppName} — variante {#VariantLabel}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; Por equipo (Program Files) con administrador, o "solo para mí" sin administrador: muchas laptops de campo
; no tienen permisos de administrador.
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog commandline
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir={#Root}\dist\installers
OutputBaseFilename={#OutputName}
SetupIconFile={#Root}\public\logo.ico
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName} {#AppVersion}
LicenseFile={#Root}\THIRD_PARTY_NOTICES.txt
Compression=lzma2/normal
SolidCompression=no
LZMAUseSeparateProcess=yes
#if Variant == "standard"
; Los modelos (~505 MB) + su caché, que se descargan en el primer inicio, en %LOCALAPPDATA%.
ExtraDiskSpaceRequired={#WwmModelsDiskMB * 1024 * 1024}
#endif
WizardStyle=modern
CloseApplications=yes

[Languages]
Name: "es"; MessagesFile: "compiler:Languages\Spanish.isl"

[Tasks]
Name: "desktopicon"; Description: "Crear un acceso directo en el escritorio"; GroupDescription: "Accesos directos:"; Flags: unchecked

[Files]
Source: "{#Root}\dist\WWMClasificador\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#Root}\LICENSE"; DestDir: "{app}"; DestName: "LICENSE.txt"; Flags: ignoreversion
Source: "{#Root}\THIRD_PARTY_NOTICES.txt"; DestDir: "{app}"; Flags: ignoreversion
#if Variant == "full"
Source: "{#Root}\dist\models\*"; DestDir: "{app}\models"; Flags: ignoreversion recursesubdirs createallsubdirs
#endif

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Abrir {#AppName} ahora"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Solo la carpeta de la aplicación. NUNCA las carpetas de destino de la investigadora (E-13).
Type: filesandordirs; Name: "{app}\_internal"
Type: dirifempty; Name: "{app}"

[Code]
function InitializeSetup: Boolean;
begin
  Result := WwmCheckRequirements;
end;

{ E-13: al desinstalar, preguntar por los modelos descargados y los registros (No por defecto). }
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
begin
  if CurUninstallStep <> usPostUninstall then
    exit;
  DataDir := ExpandConstant('{localappdata}\WWM');
  if UninstallSilent or not DirExists(DataDir) then
    exit;
  if MsgBox('¿Borrar también los modelos descargados, la configuración y los registros?' + #13#10 +
      '(' + DataDir + ')' + #13#10#13#10 +
      'Tus carpetas con videos y resultados no se tocan.', mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
    DelTree(DataDir, True, True, True);
end;
