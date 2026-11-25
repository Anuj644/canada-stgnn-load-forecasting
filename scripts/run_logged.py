"""Run a Python CLI with line-flushed logs, avoiding shell redirection buffering."""
from pathlib import Path
import runpy
import sys
import traceback

if len(sys.argv) < 3:
    raise SystemExit("Usage: run_logged.py LOG_FILE SCRIPT [SCRIPT_ARGUMENTS...]")
log_path, script = Path(sys.argv[1]), Path(sys.argv[2])
arguments = sys.argv[3:]
log_path.parent.mkdir(parents=True, exist_ok=True)
original_stdout, original_stderr = sys.stdout, sys.stderr
try:
    with log_path.open("x", encoding="utf-8", buffering=1) as log:
        sys.stdout = sys.stderr = log
        sys.argv = [str(script), *arguments]
        try:
            runpy.run_path(str(script), run_name="__main__")
        except BaseException:
            traceback.print_exc(file=log)
            raise
finally:
    sys.stdout, sys.stderr = original_stdout, original_stderr
