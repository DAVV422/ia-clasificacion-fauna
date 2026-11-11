"""Punto de entrada de la app de escritorio (HU-DS-00).

Orden estricto (las librerías pesadas leen sus variables de entorno al importarse):
    1. freeze_support (PyInstaller)          4. logging JSON
    2. stdout/stderr en modo ventana         5. argumentos (--selftest, --version)
    3. rutas de datos + variables de entorno 6. recién entonces wwm_pipeline y la ventana

Uso:
    python -m desktop.main              abre la app
    python -m desktop.main --version    imprime la versión
    python -m desktop.main --selftest   verifica rutas, logs e imports (sin cargar los modelos); código 0 si todo bien
"""

import atexit
import multiprocessing
import os
import sys


class _NullStream:
    """En el ejecutable sin consola, stdout/stderr son None y algunas librerías fallan al escribir."""

    def write(self, data):
        return len(data) if data else 0

    def flush(self):
        pass

    def isatty(self):
        return False

    def fileno(self):
        return -1


def _fix_streams() -> None:
    if sys.stdout is None:
        sys.stdout = _NullStream()  # type: ignore[assignment]
    if sys.stderr is None:
        sys.stderr = _NullStream()  # type: ignore[assignment]


def _close_splash() -> None:
    """El splash de PyInstaller solo existe en el ejecutable; la ventana lo cierra, el selftest también."""
    try:
        import pyi_splash  # type: ignore[import-not-found]

        if pyi_splash.is_alive():
            pyi_splash.close()
    except Exception:
        pass


def selftest() -> int:
    """Comprueba que el entorno (o el ejecutable) tenga todo lo necesario, sin cargar los modelos."""
    import importlib

    from desktop import __version__, paths
    from desktop.logging_setup import log_event

    report = {"version": __version__, "data_dir": str(paths.data_dir())}
    failures = []
    for module in (
        "torch",
        "torchvision",
        "cv2",
        "PIL",
        "numpy",
        "speciesnet.classifier",
        "wwm_pipeline.pipeline",
        "desktop.job.folder_job",
        "tkinter",
    ):
        try:
            mod = importlib.import_module(module)
            report[module] = getattr(mod, "__version__", "ok")
        except Exception as exc:  # pragma: no cover - depende del entorno
            failures.append(f"{module}: {exc}")
    try:
        import torch

        report["torch_hub_dir"] = torch.hub.get_dir()
    except Exception:  # pragma: no cover
        pass
    log_event("desktop.selftest.completed", "selftest", failures=failures, report=report)
    for key, value in report.items():
        print(f"{key}: {value}")
    for failure in failures:
        print(f"FALTA {failure}")
    return 1 if failures else 0


def run(argv=None) -> int:
    multiprocessing.freeze_support()
    _fix_streams()
    args = list(sys.argv[1:] if argv is None else argv)

    from desktop import __version__

    if "--version" in args:
        print(__version__)
        return 0

    from desktop import paths

    paths.ensure_dirs()
    paths.configure_env()
    # HU-DS-04: si alguna librería escribe con rutas relativas, que caiga en AppData y no junto al .exe.
    os.chdir(paths.data_dir())
    paths.clear_tmp()  # restos de una sesión que terminó mal
    atexit.register(paths.clear_tmp)

    from desktop import logging_setup

    logging_setup.configure()
    logging_setup.log_event("desktop.started", f"WWM desktop {__version__}", argv=args)

    if "--selftest" in args:
        _close_splash()
        return selftest()

    # Evita que un subproceso que heredó el entorno vuelva a abrir la ventana (comportamiento del prototipo).
    if os.environ.get("WWM_GUI_LOCKED") == "1":
        return 0
    os.environ["WWM_GUI_LOCKED"] = "1"

    from desktop.ui.legacy_window import WildlifeDesktopApp

    try:
        app = WildlifeDesktopApp()
        app.mainloop()
    finally:
        paths.clear_tmp()
    return 0


if __name__ == "__main__":
    sys.exit(run())
