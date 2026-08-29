# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_all


project_dir = Path(SPECPATH)
venv_dir = project_dir / ".venv"

datas = [(str(project_dir / "assets"), "assets")]
binaries = [(str(venv_dir / "Lib" / "gmsh-4.15.dll"), ".")]
hiddenimports = []

# PyVista loads VTK modules dynamically, so collect its package metadata and
# dynamic submodules explicitly in addition to PyInstaller's standard hooks.
pyvista_datas, pyvista_binaries, pyvista_hiddenimports = collect_all("pyvista")
datas += pyvista_datas
binaries += pyvista_binaries
hiddenimports += pyvista_hiddenimports

a = Analysis(
    [str(project_dir / "ui_test.py")],
    pathex=[str(project_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="GridResistanceAnalyzer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="GridResistanceAnalyzer",
)
