import os
import sys
import runpy

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

runpy.run_module("webpilot.mcp_server", run_name="__main__")