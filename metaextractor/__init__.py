from pathlib import Path

from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parents[1] / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

from .config import setup_logging  # noqa: E402

setup_logging()

# noqa: E402 — dotenv and logging must load before other modules read env vars
from .main import app as app  # noqa: E402
from .main import cli_app as cli_app  # noqa: E402

__version__ = "0.1.0"
