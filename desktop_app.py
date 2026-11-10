"""Alias del punto de entrada (para Iniciar_App.bat y el build viejo). Usar ``python -m desktop.main``."""

import sys

from desktop.main import run

if __name__ == "__main__":
    sys.exit(run())
