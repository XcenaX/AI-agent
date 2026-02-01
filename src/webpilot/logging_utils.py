import logging
import os
from rich.logging import RichHandler
from rich.traceback import install as rich_install

def setup_logging() -> None:
    rich_install(show_locals=False)
    level = os.getenv("WEBPILOT_LOG_LEVEL", "INFO").upper()

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=False, markup=True, show_path=False)],
    )

    for name in ["httpcore", "hpack", "h2", "urllib3"]:
        logging.getLogger(name).setLevel(logging.ERROR)

    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("playwright").setLevel(os.getenv("PLAYWRIGHT_LOG_LEVEL", "WARNING"))