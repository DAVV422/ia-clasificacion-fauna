"""Genera ``installer/requirements.iss`` desde ``desktop.requirements_check.REQUIREMENTS`` (HU-DS-20 UT-3).

Una sola tabla de requisitos: la app y el instalador usan los mismos números. ``build.ps1`` (HU-DS-12) lo corre
antes de ``ISCC``, y ``installer/wwm.iss`` lo incluye con ``#include "requirements.iss"``.

El instalador verifica (E-01): Windows de 64 bits (si no, no deja instalar) y la RAM (aviso con "Instalar de
todos modos", con No por defecto). El espacio en disco lo verifica Inno Setup con ``ExtraDiskSpaceRequired``.
AVX2 **no** se verifica aquí: lo hace la app al abrir.

    python tools/gen_requirements_iss.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from desktop.requirements_check import REQUIREMENTS  # noqa: E402

OUT = ROOT / "installer" / "requirements.iss"

TEMPLATE = r"""; GENERADO por tools/gen_requirements_iss.py desde desktop/requirements_check.REQUIREMENTS. No editar a mano.
#define WwmRamMinMB @RAM_MIN_MB@
#define WwmRamMinGB @RAM_MIN_GB@
#define WwmModelsDiskMB @DISK_MODELS_MB@

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
"""


def render() -> str:
    values = {
        "@RAM_MIN_MB@": REQUIREMENTS["ram_min_gb"] * 1024,
        "@RAM_MIN_GB@": REQUIREMENTS["ram_min_gb"],
        "@DISK_MODELS_MB@": REQUIREMENTS["disk_models_gb"] * 1024,
    }
    text = TEMPLATE
    for token, value in values.items():  # sin str.format: el Pascal usa llaves en los comentarios
        text = text.replace(token, str(value))
    return text


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(), encoding="utf-8-sig")  # Inno Setup lee UTF-8 con BOM (las tildes)
    print(f"escrito {OUT}")


if __name__ == "__main__":
    main()
