import os
from dataclasses import dataclass
from typing import Optional, List

from playwright.async_api import async_playwright, Page, BrowserContext


@dataclass
class BrowserConfig:
    user_data_dir: str = "./profile"
    headless: bool = False
    slow_mo_ms: int = 50
    viewport_w: int = 1280
    viewport_h: int = 800


class BrowserController:
    def __init__(self, cfg: BrowserConfig):
        self.cfg = cfg
        self._pw = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

    async def start(self) -> None:
        self._pw = await async_playwright().start()

        # для persistent лучше абсолютный путь на Windows (меньше сюрпризов)
        user_data_dir = os.path.abspath(self.cfg.user_data_dir)

        self.context = await self._pw.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=self.cfg.headless,
            slow_mo=self.cfg.slow_mo_ms,
            viewport={"width": self.cfg.viewport_w, "height": self.cfg.viewport_h},
            channel="chrome",
        )

        pages: List[Page] = self.context.pages
        self.page = pages[0] if pages else await self.context.new_page()
        await self.page.goto("about:blank")

    async def stop(self) -> None:
        if self.context:
            await self.context.close()
        if self._pw:
            await self._pw.stop()

    async def ensure_active_page(self) -> None:
        # Если после клика открылся новый таб — берём последний
        if not self.context:
            return
        pages = self.context.pages
        if pages:
            self.page = pages[-1]