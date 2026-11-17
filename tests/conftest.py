"""Configuración común de pytest.

Tcl/Tk aborta el proceso (``Windows fatal exception 0x80000003``, ``Tcl_AsyncDelete``) si un objeto de una ventana
Tk se libera desde un hilo que no es el principal. Los tests de la ventana crean y destruyen ``Tk()``, y los de la
API dejan hilos vivos (``TestClient``, el cargador de modelos): si el *garbage collector* corre en uno de esos hilos
y encuentra un resto de Tk, se cae toda la suite.

Por eso, después de cada test que importó ``tkinter``, se fuerza la recolección **en el hilo principal**.
"""

import gc
import sys

import pytest


@pytest.fixture(autouse=True)
def _collect_tk_leftovers_in_main_thread():
    yield
    if "tkinter" in sys.modules:
        gc.collect()
