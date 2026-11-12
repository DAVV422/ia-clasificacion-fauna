"""Requisitos del equipo (HU-DS-20, E-01, E-02, E-12, F4-D7).

``check(probe())`` devuelve un ``Report``: una fila por verificación, con su nivel (``ok`` verde, ``warn``
amarillo, ``fail`` rojo) y ``blocked`` si la app no puede seguir. La pantalla BLOQUEADA, el aviso y
"Información del sistema" solo lo dibujan. Sin ``tkinter``.

| Verificación | Mínimo                 | Recomendado | Bajo el mínimo                          |
|--------------|------------------------|-------------|-----------------------------------------|
| Windows      | 10 u 11, de 64 bits    | —           | BLOQUEADA                               |
| CPU          | AVX2                   | 4 núcleos   | sin AVX2 → BLOQUEADA                    |
| RAM total    | 8 GB                   | 16 GB       | aviso                                   |
| Disco libre  | 2 GB (si faltan los modelos) | —     | aviso                                   |

**AVX2, por dos vías:** ``IsProcessorFeaturePresent(40)`` de Windows y la detección de NumPy
(``__cpu_features__``). En Windows 10 anteriores a 21H2 la API no conoce el código 40 y dice "no": por eso
basta con que **una** diga que sí. Si las dos fallan, es "desconocido" y no bloquea.

Para QA (``WWM_FAKE_REQUIREMENTS=no_avx2,ram4,...``) solo se lee desde el código o con ``--qa``, nunca en el
ejecutable de la investigadora.
"""

import contextlib
import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from desktop import __version__

# F4-D7. Una sola tabla: la usa la app y, generada por tools/gen_requirements_iss.py, el instalador.
REQUIREMENTS = {
    "windows_min_build": 10240,  # Windows 10
    "ram_min_gb": 8,
    "ram_recommended_gb": 16,
    "cores_recommended": 4,
    "disk_models_gb": 2,
}
# Provisional hasta IA-17 (fase 6): esta laptop, estacion_1 = 36 min / 27 videos ≈ 80 s por video.
SECONDS_PER_15S_VIDEO = 80
PF_AVX2_INSTRUCTIONS_AVAILABLE = 40
WIN11_FIRST_BUILD = 22000


@dataclass
class SystemInfo:
    windows: str
    windows_build: int
    is_64bit: bool
    cpu: str
    cores: int
    avx2: Optional[bool]  # None = no se pudo saber
    ram_gb: float
    disk_free_gb: Optional[float]
    versions: Dict[str, str] = field(default_factory=dict)


@dataclass
class Check:
    name: str
    label: str
    value: str
    level: str  # ok | warn | fail
    message: str = ""


@dataclass
class Report:
    checks: List[Check]
    info: SystemInfo

    @property
    def blocked(self) -> bool:
        return any(c.level == "fail" for c in self.checks)

    @property
    def warnings(self) -> List[Check]:
        return [c for c in self.checks if c.level == "warn"]

    @property
    def reason(self) -> str:
        """El motivo para REQUIREMENTS_NOT_MET ("Tu equipo no es compatible con el modelo de IA: {motivo}")."""
        return "; ".join(c.message for c in self.checks if c.level == "fail")

    def to_dict(self) -> dict:
        return {"blocked": self.blocked, "checks": [asdict(c) for c in self.checks], "info": asdict(self.info)}


# ---------------------------------------------------------------------------
# Leer el equipo
# ---------------------------------------------------------------------------
def avx2_from_windows() -> Optional[bool]:
    try:
        import ctypes

        return bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE))  # type: ignore[attr-defined]
    except Exception:
        return None


def avx2_from_numpy() -> Optional[bool]:
    try:
        from numpy.core._multiarray_umath import __cpu_features__  # type: ignore[import-not-found]

        return bool(__cpu_features__.get("AVX2"))
    except Exception:
        return None


def combine_avx2(sources: List[Callable[[], Optional[bool]]]) -> Optional[bool]:
    """``True`` si alguna fuente dice que sí; ``False`` si todas las que respondieron dicen que no; si no, ``None``."""
    answers = [src() for src in sources]
    if any(a is True for a in answers):
        return True
    if any(a is False for a in answers):
        return False
    return None


def _cpu_name() -> str:
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except Exception:
        return platform.processor() or "desconocida"


def _windows() -> tuple:
    version = platform.version()  # p. ej. 10.0.26200
    try:
        build = int(version.split(".")[2])
    except (IndexError, ValueError):
        build = 0
    if platform.system() != "Windows":
        return f"{platform.system()} {platform.release()}", build
    name = "Windows 11" if build >= WIN11_FIRST_BUILD else f"Windows {platform.release()}"
    return f"{name} (compilación {build})", build


def _versions() -> Dict[str, str]:
    from importlib import metadata

    out = {"app": __version__}
    for pkg in ("torch", "speciesnet", "opencv-python-headless", "numpy"):
        with contextlib.suppress(metadata.PackageNotFoundError):
            out[pkg] = metadata.version(pkg)
    return out


def probe(disk_path: Optional[Path] = None) -> SystemInfo:
    import psutil

    from desktop import paths

    windows, build = _windows()
    disk_path = disk_path or paths.data_dir()
    try:
        disk_free: Optional[float] = shutil.disk_usage(str(_existing_parent(disk_path))).free / 2**30
    except OSError:
        disk_free = None
    info = SystemInfo(
        windows=windows,
        windows_build=build,
        is_64bit=sys.maxsize > 2**32 and platform.machine().endswith("64"),
        cpu=_cpu_name(),
        cores=os.cpu_count() or 1,
        avx2=combine_avx2([avx2_from_windows, avx2_from_numpy]),
        ram_gb=psutil.virtual_memory().total / 2**30,
        disk_free_gb=disk_free,
        versions=_versions(),
    )
    return _apply_fakes(info)


def _existing_parent(path: Path) -> Path:
    path = Path(path)
    while not path.exists() and path.parent != path:
        path = path.parent
    return path


def _apply_fakes(info: SystemInfo) -> SystemInfo:
    """QA-14 simulado: ``WWM_FAKE_REQUIREMENTS`` solo fuera del ejecutable o con ``--qa`` (riesgo 2)."""
    raw = os.getenv("WWM_FAKE_REQUIREMENTS", "")
    if not raw or (getattr(sys, "frozen", False) and "--qa" not in sys.argv):
        return info
    for fake in (x.strip() for x in raw.split(",")):
        if fake == "no_avx2":
            info.avx2 = False
        elif fake == "avx2_unknown":
            info.avx2 = None
        elif fake == "32bit":
            info.is_64bit = False
        elif fake.startswith("ram"):
            info.ram_gb = float(fake[3:])
        elif fake.startswith("disk"):
            info.disk_free_gb = float(fake[4:])
        elif fake.startswith("cores"):
            info.cores = int(fake[5:])
    return info


# ---------------------------------------------------------------------------
# Evaluar
# ---------------------------------------------------------------------------
def check(info: SystemInfo, needs_model_download: bool = False) -> Report:
    r = REQUIREMENTS
    checks: List[Check] = []

    if not info.is_64bit:
        checks.append(Check("windows", "Windows", info.windows, "fail", "esta aplicación requiere Windows de 64 bits"))
    elif info.windows_build and info.windows_build < r["windows_min_build"]:
        checks.append(Check("windows", "Windows", info.windows, "fail", "requiere Windows 10 u 11"))
    else:
        checks.append(Check("windows", "Windows", f"{info.windows}, 64 bits", "ok"))

    cpu_value = f"{info.cpu} · {info.cores} núcleos"
    if info.avx2 is False:
        checks.append(
            Check("avx2", "Procesador (AVX2)", cpu_value + " · AVX2 ✗", "fail", "falta AVX2 en el procesador")
        )
    elif info.avx2 is None:
        checks.append(
            Check(
                "avx2",
                "Procesador (AVX2)",
                cpu_value + " · AVX2 ?",
                "warn",
                "No se pudo comprobar si el procesador tiene AVX2. Si la app se cierra al cargar el modelo, "
                "el equipo no es compatible.",
            )
        )
    elif info.cores < r["cores_recommended"]:
        checks.append(
            Check(
                "avx2",
                "Procesador",
                cpu_value + " · AVX2 ✓",
                "warn",
                f"Con menos de {r['cores_recommended']} núcleos el análisis será más lento.",
            )
        )
    else:
        checks.append(Check("avx2", "Procesador", cpu_value + " · AVX2 ✓", "ok"))

    ram_value = f"{info.ram_gb:.0f} GB"  # sin decimales: no hace falta la coma
    if info.ram_gb < r["ram_min_gb"] - 0.5:  # Windows informa algo menos que lo instalado
        checks.append(
            Check(
                "ram",
                "Memoria (RAM)",
                ram_value,
                "warn",
                f"Con menos de {r['ram_min_gb']} GB de RAM puede quedarse sin memoria. Cierra otras aplicaciones.",
            )
        )
    elif info.ram_gb < r["ram_recommended_gb"] - 0.5:
        checks.append(
            Check(
                "ram",
                "Memoria (RAM)",
                ram_value,
                "warn",
                f"Cumple el mínimo ({r['ram_min_gb']} GB); se recomiendan {r['ram_recommended_gb']} GB.",
            )
        )
    else:
        checks.append(Check("ram", "Memoria (RAM)", ram_value, "ok"))

    if info.disk_free_gb is None:
        checks.append(Check("disk", "Disco libre", "desconocido", "ok"))
    elif needs_model_download and info.disk_free_gb < r["disk_models_gb"]:
        checks.append(
            Check(
                "disk",
                "Disco libre",
                _gb(info.disk_free_gb),
                "warn",
                f"Necesitas al menos {r['disk_models_gb']} GB libres para los modelos.",
            )
        )
    else:
        checks.append(Check("disk", "Disco libre", _gb(info.disk_free_gb), "ok"))

    return Report(checks, info)


def _gb(value: float) -> str:
    return f"{value:.1f} GB".replace(".", ",")


def warning_text(report: Report) -> str:
    """El aviso no bloqueante (CA-3), con el tiempo estimado (provisional hasta IA-17)."""
    lines = [c.message for c in report.warnings]
    lines.append(f"Un video de 15 s tarda aproximadamente {SECONDS_PER_15S_VIDEO} s en este tipo de equipo (estimado).")
    return "Tu equipo puede usar la aplicación, con estas observaciones:\n\n• " + "\n• ".join(lines)


def system_info_text(report: Report) -> str:
    """Para "Copiar" en Información del sistema y para el diagnóstico (HU-DS-19)."""
    marks = {"ok": "✓", "warn": "!", "fail": "✗"}
    rows = [f"{marks[c.level]} {c.label}: {c.value}" for c in report.checks]
    rows += [f"  {k}: {v}" for k, v in report.info.versions.items()]
    return "\n".join(rows)
