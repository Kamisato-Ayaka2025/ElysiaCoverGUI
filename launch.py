"""Start the GUI using the configured SVC environment."""
import subprocess
import sys
from runtime_config import APP, path

if __name__ == '__main__':
    python = path('svc_python')
    if not python.is_file():
        sys.exit('Configure svc_python in config.local.json first. See README.md.')
    subprocess.run([str(python), str(APP/'app.py')], cwd=str(APP), check=True)
