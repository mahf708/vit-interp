"""Execute the notebooks end to end (a test), saving executed copies elsewhere.

    PLAYGROUND_FAST=1 PLAYGROUND_CKPT=... .venv/bin/python notebooks/_run.py [names...]

The notebooks in this folder stay output-free; executed copies go to $PLAYGROUND_OUT
(default /lcrc/globalscratch/$USER/vit-interp/notebooks-executed).
"""

import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).parent
OUT = Path(os.environ.get("PLAYGROUND_OUT",
                          f"/lcrc/globalscratch/{os.environ['USER']}/vit-interp/notebooks-executed"))
OUT.mkdir(parents=True, exist_ok=True)
names = sys.argv[1:] or sorted(p.name for p in HERE.glob("[0-9]*.ipynb"))
failed = []
for name in names:
    nb = nbformat.read(HERE / name, as_version=4)
    t = time.time()
    try:
        NotebookClient(nb, timeout=3600, kernel_name="vit-interp",
                       resources={"metadata": {"path": str(HERE)}}).execute()
        status = "ok"
    except Exception as exc:  # report and keep going
        status = f"FAILED: {type(exc).__name__}: {str(exc)[-1500:]}"
        failed.append(name)
    nbformat.write(nb, OUT / name)
    print(f"{name}: {status} ({time.time() - t:.0f} s)", flush=True)
sys.exit(1 if failed else 0)
