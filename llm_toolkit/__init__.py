# Init: load .env so environment variables from the repository are available
from pathlib import Path
from dotenv import load_dotenv

# Load .env from repository root if present, otherwise use default search
env_path = Path(__file__).resolve().parents[1] / ".env"
if env_path.exists():
	load_dotenv(env_path)
else:
	load_dotenv()

from .core import ask

__all__ = ["ask"]
