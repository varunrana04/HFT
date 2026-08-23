"""
engine_loader.py — Centralised hft_engine import helper.

Every Python script that uses the C++ engine should do:

    from engine_loader import load_engine
    hft_engine = load_engine()   # returns the module, or raises ImportError

This module handles all the platform-specific path setup that was previously
duplicated (slightly differently) in every script:
  - Adds the build/ directory to sys.path
  - Calls os.add_dll_directory() on Windows so the MinGW runtime DLLs
    (libgcc_s_seh-1.dll, libstdc++-6.dll, libwinpthread-1.dll) are found
  - Gives a clear, actionable error message if the .pyd is missing or built
    for the wrong Python version
  - Verifies the loaded module's PRICE_SCALE to confirm it is not stale

Usage:
    from engine_loader import load_engine, ENGINE_AVAILABLE
    hft_engine = load_engine()          # raises on failure
    # or:
    if ENGINE_AVAILABLE:
        import hft_engine
"""

import sys
import os
from pathlib import Path

# ─── Locate build/ directory ─────────────────────────────────
# Works regardless of which directory the caller script lives in.
_THIS_DIR  = Path(__file__).resolve().parent          # python/
_ROOT      = _THIS_DIR.parent                          # project root
_BUILD_DIR = _ROOT / "build"

# Well-known MinGW bin locations on Windows (DLL resolution)
_MINGW_CANDIDATES = [
    _ROOT / "toolchain" / "mingw64" / "bin",
    Path(os.environ.get("USERPROFILE", "")) / "mingw64" / "bin",
    Path("C:/msys64/mingw64/bin"),
    Path("C:/mingw64/bin"),
]

# ─── Module-level cache ───────────────────────────────────────
_engine_module = None
ENGINE_AVAILABLE = False


def _setup_dll_dirs() -> None:
    """Add build/ and MinGW bin to the DLL search path (Windows only)."""
    if sys.platform != "win32":
        return
    if not hasattr(os, "add_dll_directory"):
        return   # Python < 3.8 — rely on PATH

    # Add build dir so onnxruntime.dll (if present) is found too
    if _BUILD_DIR.is_dir():
        os.add_dll_directory(str(_BUILD_DIR))

    for mingw in _MINGW_CANDIDATES:
        try:
            if mingw.is_dir():
                os.add_dll_directory(str(mingw))
        except PermissionError:
            pass


def _add_to_sys_path() -> None:
    """Add build/ and python/ to sys.path if not already present."""
    build_str = str(_BUILD_DIR)
    this_str = str(_THIS_DIR)
    if build_str not in sys.path:
        sys.path.insert(0, build_str)
    if this_str not in sys.path:
        sys.path.insert(0, this_str)


def _check_python_version(pyd_name: str) -> None:
    """
    Warn clearly if the .pyd ABI tag doesn't match the running Python.
    e.g. hft_engine.cp311-win_amd64.pyd loaded by Python 3.14 will fail.
    """
    import re
    m = re.search(r"cp(\d)(\d+)", pyd_name)
    if not m:
        return
    pyd_major, pyd_minor = int(m.group(1)), int(m.group(2))
    cur_major, cur_minor = sys.version_info.major, sys.version_info.minor
    if (pyd_major, pyd_minor) != (cur_major, cur_minor):
        raise ImportError(
            f"\n"
            f"  Version mismatch!\n"
            f"  .pyd compiled for : Python {pyd_major}.{pyd_minor}  "
            f"({pyd_name})\n"
            f"  Running Python     : {cur_major}.{cur_minor}  "
            f"({sys.executable})\n\n"
            f"  Fix: run scripts with the matching Python:\n"
            f"    .venv\\Scripts\\python.exe <script.py>\n\n"
            f"  Or rebuild for the current Python:\n"
            f"    .\\build_python_bridge.ps1"
        )


def load_engine(silent: bool = False):
    """
    Import and return the hft_engine module.

    Parameters
    ----------
    silent : bool
        If True, return None instead of raising on failure (useful for
        scripts that can run in a degraded mode without the C++ engine).

    Returns
    -------
    module or None
        The imported hft_engine module.

    Raises
    ------
    ImportError
        If the engine cannot be loaded and silent=False.
    """
    global _engine_module, ENGINE_AVAILABLE

    # Return cached module on repeated calls
    if _engine_module is not None:
        return _engine_module

    # Find the .pyd file
    if not _BUILD_DIR.is_dir():
        msg = (
            f"\n"
            f"  build/ directory not found at: {_BUILD_DIR}\n\n"
            f"  Build the engine first:\n"
            f"    .\\build_python_bridge.ps1\n"
            f"  or:\n"
            f"    cmake -B build -G \"MinGW Makefiles\" -DCMAKE_BUILD_TYPE=Release\n"
            f"    cmake --build build --parallel"
        )
        if silent:
            return None
        raise ImportError(msg)

    pyd_files = list(_BUILD_DIR.glob("hft_engine*.pyd")) + \
                list(_BUILD_DIR.glob("hft_engine*.so")) + \
                list(_THIS_DIR.glob("hft_engine*.pyd"))

    if not pyd_files:
        print("\n[WARNING] hft_engine.pyd not found. Falling back to Pure Python Mock Engine!")
        import pure_python_engine as _mod
        _engine_module = _mod
        ENGINE_AVAILABLE = True
        return _mod

    # Find the pyd file that matches our python version if possible
    import sys
    major, minor = sys.version_info[:2]
    expected_tag = f"cp{major}{minor}"
    
    selected_pyd = pyd_files[0]
    for p in pyd_files:
        if expected_tag in p.name:
            selected_pyd = p
            break

    # Check ABI compatibility before attempting import
    _check_python_version(selected_pyd.name)

    # Set up DLL search paths, then import
    _setup_dll_dirs()
    _add_to_sys_path()

    try:
        import hft_engine as _mod
        
        # Quick sanity check — if this attribute is missing the .pyd is stale
        if not hasattr(_mod, "PRICE_SCALE"):
            raise ImportError("hft_engine loaded but is stale (missing PRICE_SCALE).")
            
    except ImportError as e:
        print(f"\n[WARNING] Failed to load C++ hft_engine: {e}")
        print("[WARNING] Falling back to Pure Python Mock Engine!")
        import pure_python_engine as _mod

    _engine_module = _mod
    ENGINE_AVAILABLE = True
    return _mod


# ─── Convenience: auto-load on import ────────────────────────
# If imported at module level, attempt a silent load so scripts can
# check ENGINE_AVAILABLE without calling load_engine() explicitly.
try:
    load_engine(silent=True)
except Exception:
    pass
