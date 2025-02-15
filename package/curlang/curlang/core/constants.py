import re
import sys

from datetime import datetime, timedelta, timezone
from importlib.metadata import version

from pathlib import Path

HOME_DIR = Path.home() / ".curlang"
HOME_DIR.mkdir(exist_ok=True)
PACKAGE_DIR = Path(sys.modules["curlang"].__file__).parent
CONFIG_FILE_PATH = HOME_DIR / ".curlang_config.toml"

GIT_CACHE_FILE = HOME_DIR / ".curlang_git.cache"
IMPORT_CACHE_FILE = PACKAGE_DIR / ".curlang_import_cache"

BASE_URL = "https://raw.githubusercontent.com/romlin/curlang/main/warehouse"
GITHUB_REPO_URL = "https://api.github.com/repos/romlin/curlang"
TEMPLATE_REPO_URL = "https://api.github.com/repos/romlin/template"

CONNECTIONS_FILE = "connections.json"
HOOKS_FILE = "hooks.json"

COOLDOWN_PERIOD = timedelta(minutes=1)
GIT_CACHE_EXPIRY = timedelta(hours=1)
SERVER_START_TIME = None

ANSI_ESCAPE = re.compile(rb'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

CSRF_EXEMPT_PATHS = ["/", "/csrf-token", "/favicon.ico", "/static"]

MAX_ATTEMPTS = 1000000
VALIDATION_ATTEMPTS = 0

VERSION = version("curlang")
