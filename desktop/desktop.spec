# -*- mode: python ; coding: utf-8 -*-
# HU-DS-00 · build reproducible de la app de escritorio (onedir).
#   pyinstaller desktop/desktop.spec --noconfirm
# Resultado: dist/WWMClasificador/WWMClasificador.exe  (probar con --selftest)
# Los modelos NO se empaquetan aquí (variante estándar: se descargan en el primer uso a %LOCALAPPDATA%\WWM).
import os
from PyInstaller.utils.hooks import collect_all

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
# pkg_resources (que usan yolov5/ultralytics) importa jaraco -> backports.tarfile, que setuptools trae
# "vendorizado" en setuptools/_vendor; sin esto el ejecutable falla al arrancar (ModuleNotFoundError: backports).
import setuptools

SETUPTOOLS_VENDOR = os.path.join(os.path.dirname(setuptools.__file__), "_vendor")

datas = [
    (os.path.join(ROOT, "public", "logo.ico"), "public"),
    (os.path.join(ROOT, "public", "logo.jpeg"), "public"),
    # HU-IA-14: nombres comunes en español (wwm_pipeline/taxonomy.py los busca junto al módulo)
    (os.path.join(ROOT, "wwm_pipeline", "data", "common_names_es.csv"), os.path.join("wwm_pipeline", "data")),
    # HU-DS-11: el código de YOLOv5 que carga MegaDetector sin torch.hub remoto (wwm_pipeline.models busca
    # <_MEIPASS>/third_party/yolov5_hub). Va como fuente: torch.hub.load(..., source="local") lo importa de ahí.
    (os.path.join(ROOT, "third_party", "yolov5_hub"), os.path.join("third_party", "yolov5_hub")),
    # HU-DS-13: qué modelos descargar del release models-v1 (URL, tamaño y sha256); model_store lo busca ahí.
    (os.path.join(ROOT, "desktop", "models", "model_manifest.json"), os.path.join("desktop", "models")),
]
binaries = []
hiddenimports = ["desktop.ui.legacy_window", "backports", "backports.tarfile"]
for pkg in ("cloudpathlib", "speciesnet", "ultralytics", "onnx2torch"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# pandas 3 trae sus DLL (MSVC runtime) en site-packages/pandas.libs (delvewheel) y el hook no las copia:
# sin ellas speciesnet falla con "DLL load failed while importing aggregations".
import glob
import pandas

_pandas_libs = os.path.join(os.path.dirname(os.path.dirname(pandas.__file__)), "pandas.libs")
binaries += [(dll, "pandas.libs") for dll in glob.glob(os.path.join(_pandas_libs, "*.dll"))]

a = Analysis(
    [os.path.join(ROOT, "desktop_app.py")],
    pathex=[ROOT, SETUPTOOLS_VENDOR],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["pytest", "matplotlib.tests"],  # speciesnet importa IPython; fastapi/uvicorn los usa app.detector indirectamente
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
splash = Splash(
    os.path.join(ROOT, "public", "splash.png"),
    binaries=a.binaries,
    datas=a.datas,
    text_pos=None,
    minify_script=True,
    always_on_top=True,
)
exe = EXE(
    pyz,
    a.scripts,
    splash,
    [],
    exclude_binaries=True,
    name="WWMClasificador",
    debug=False,
    strip=False,
    upx=False,
    console=False,
    icon=[os.path.join(ROOT, "public", "logo.ico")],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    splash.binaries,
    strip=False,
    upx=False,
    name="WWMClasificador",
)
