"""Resolve portable paths; machine-specific configuration stays untracked."""
import json
from pathlib import Path
APP = Path(__file__).resolve().parent
CONFIG = json.loads((APP/'config.example.json').read_text(encoding='utf-8'))
if (APP/'config.local.json').exists():
    CONFIG.update(json.loads((APP/'config.local.json').read_text(encoding='utf-8-sig')))
def path(key):
    value = Path(CONFIG[key]).expanduser()
    return (value if value.is_absolute() else APP/value).resolve()
