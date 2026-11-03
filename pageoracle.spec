# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

datas = []
datas += collect_data_files("chromadb")
datas += collect_data_files("certifi")
datas += copy_metadata("tqdm")
datas += copy_metadata("regex")
datas += copy_metadata("requests")
datas += copy_metadata("torch")
datas += copy_metadata("sentence_transformers")
datas += copy_metadata("huggingface_hub")
datas += copy_metadata("transformers")

hiddenimports = []
hiddenimports += collect_submodules("chromadb")
hiddenimports += collect_submodules("langchain")
hiddenimports += collect_submodules("langchain_community")
hiddenimports += collect_submodules("langchain_chroma")
hiddenimports += collect_submodules("langchain_huggingface")
hiddenimports += collect_submodules("langchain_core")
hiddenimports += collect_submodules("langchain_deepseek")
hiddenimports += collect_submodules("langchain_gigachat")
hiddenimports += collect_submodules("langchain_openrouter")
hiddenimports += collect_submodules("langchain_text_splitters")
hiddenimports += collect_submodules("langgraph")
hiddenimports += collect_submodules("sentence_transformers")
hiddenimports += [
    "chardet",
    "pydantic",
    "yandexcloud",
    "rank_bm25",
]

excludes = [
    "flask",
    "jinja2",
    "werkzeug",
    "pytest",
    "unittest",
    "IPython",
    "notebook",
    "matplotlib",
]

a = Analysis(
    ["tkinter_gui/gui.py"],
    pathex=["tkinter_gui"],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PageOracle",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
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
    upx=False,
    upx_exclude=[],
    name="PageOracle",
)
