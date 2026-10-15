"""Build the static site for Cloudflare Pages into web/dist/.

    python web/build.py            # -> web/dist (Pages output directory)

Copies web/static and packs the Python the browser needs (the shared pipeline and
analytics from src/ plus web/py/pa_web) into a content-hashed zip that the Web
Worker unpacks into Pyodide. Standard library only, so it runs in the Pages build
image without installing anything.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "web" / "static"
DIST = ROOT / "web" / "dist"

PYODIDE_VERSION = "0.29.3"
PYODIDE_PACKAGES = ["numpy", "pandas", "pydantic"]  # scipy.stats via pa_web.stats_lite

# Only what the demo pipeline imports: no Qt UI, reports, charts, or desktop data feeds.
BUNDLE = [
    "src/__init__.py",
    "src/pipeline.py",
    "src/config/__init__.py",
    "src/config/models.py",
    "src/analytics/*.py",
    "src/data/__init__.py",
    "src/data/fetcher.py",
    "src/data/risk_free.py",
    "src/data/transforms.py",
    "src/data/quality.py",
    "src/data/meta_cache.py",
    "web/py/pa_web/*.py",
]


def bundle_files() -> list[tuple[Path, str]]:
    out = []
    for pat in BUNDLE:
        for p in sorted(ROOT.glob(pat)):
            arc = p.relative_to(ROOT).as_posix()
            if arc.startswith("web/py/"):
                arc = arc[len("web/py/"):]
            out.append((p, arc))
    return out


def build_bundle() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path, arc in bundle_files():
            info = zipfile.ZipInfo(arc, date_time=(2020, 1, 1, 0, 0, 0))  # reproducible
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, path.read_bytes())
    return buf.getvalue()


def main(argv: list[str]) -> int:
    proxy = ""
    for a in argv:
        if a.startswith("--proxy="):
            proxy = a.split("=", 1)[1]
    if DIST.exists():
        shutil.rmtree(DIST)
    shutil.copytree(STATIC, DIST)
    data = build_bundle()
    digest = hashlib.sha256(data).hexdigest()[:12]
    name = f"pa_engine.{digest}.zip"
    (DIST / name).write_bytes(data)
    manifest = {"bundle": name, "pyodide": PYODIDE_VERSION, "packages": PYODIDE_PACKAGES,
                "proxy": proxy}
    (DIST / "engine-manifest.json").write_text(json.dumps(manifest, indent=1))
    # Cache-bust the worker scripts by the bundle hash too.
    boot = DIST / "boot.js"
    boot.write_text(boot.read_text(encoding="utf-8").replace("__MANIFEST__", json.dumps(manifest)),
                    encoding="utf-8")
    for fn in ("index.html",):
        p = DIST / fn
        p.write_text(p.read_text(encoding="utf-8").replace("__BUILD__", digest), encoding="utf-8")
    print(f"web/dist built: {name} ({len(data) / 1024:.0f} KB), pyodide {PYODIDE_VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
