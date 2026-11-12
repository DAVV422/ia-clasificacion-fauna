; GENERADO por tools/gen_requirements_iss.py desde desktop/requirements_check.REQUIREMENTS. No editar a mano.
#define WwmRamMinMB 8192
#define WwmRamMinGB 8
#define WwmModelsDiskMB 2048

[Code]
type
  TWwmMemoryStatusEx = record
    dwLength: Cardinal;
    dwMemoryLoad: Cardinal;
    ullTotalPhys: Int64;
    ullAvailPhys: Int64;
    ullTotalPageFile: Int64;
    ullAvailPageFile: Int64;
    ullTotalVirtual: Int64;
    ullAvailVirtual: Int64;
    ullAvailExtendedVirtual: Int64;
  end;

function WwmGlobalMemoryStatusEx(var lpBuffer: TWwmMemoryStatusEx): Boolean;
  external 'GlobalMemoryStatusEx@kernel32.dll stdcall';

function WwmTotalRamMB: Int64;
var
  Status: TWwmMemoryStatusEx;
begin
  Status.dwLength := SizeOf(Status);
  if WwmGlobalMemoryStatusEx(Status) then
    Result := Status.ullTotalPhys div (1024 * 1024)
  else
    Result := -1;
end;

{ E-01 paso 2: Windows de 64 bits obligatorio; poca RAM -> aviso con "Instalar de todos modos" (No por defecto). }
function WwmCheckRequirements: Boolean;
var
  RamMB: Int64;
begin
  Result := True;
  if not IsWin64 then
  begin
    MsgBox('Esta aplicación requiere Windows de 64 bits.', mbCriticalError, MB_OK);
    Result := False;
    exit;
  end;
  RamMB := WwmTotalRamMB;
  { Windows informa algo menos que lo instalado: se tolera medio GB. }
  if (RamMB >= 0) and (RamMB < {#WwmRamMinMB} - 512) then
    Result := MsgBox('Tu equipo no cumple los requisitos mínimos: tiene menos de {#WwmRamMinGB} GB de RAM.' + #13#10 +
      'La aplicación puede no funcionar o quedarse sin memoria.' + #13#10#13#10 +
      '¿Instalar de todos modos?', mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES;
end;
