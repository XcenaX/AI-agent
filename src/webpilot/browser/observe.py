from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from playwright.async_api import Page, Error as PlaywrightError
from playwright.async_api import Page


@dataclass
class ObservedElement:
    eid: str
    role: str
    name: str
    disabled: bool
    bbox: Optional[List[float]]
    score: float = 0.0
    href: str = ""


class PageObserver:
    def __init__(self):
        # We intentionally avoid caching ElementHandle for every element each step.
        # Agent tools use locators by [data-webpilot-eid], which is faster and stable enough.
        pass

    async def observe(self, page: Page, max_elements: int = 120, max_text_chars: int = 1500) -> Dict[str, Any]:
        url = page.url
        title = await page.title()

        # 1) Короткий видимый текст без HTML
        visible_text = await page.evaluate(
          """() => {
            try {
              const parts = [];

              const isElVisible = (el) => {
                if (!el || !(el instanceof Element)) return false;
                const style = getComputedStyle(el);
                if (!style) return false;
                if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return false;
                return true;
              };

              const pushText = (el) => {
                if (!el || !(el instanceof Node)) return;
                const t = (el.innerText || el.textContent || "");
                if (t && t.trim()) parts.push(t);
              };

              // 0) СНАЧАЛА модалки/диалоги (важно, потому что дальше текст режется до preview)
              const dialogs = Array.from(document.querySelectorAll(
                'dialog[open], [role="dialog"], [role="alertdialog"], [aria-modal="true"]'
              )).filter(isElVisible);

              for (const d of dialogs) pushText(d);

              // 1) потом header и main
              const header =
                document.querySelector("header") ||
                document.querySelector('[role="banner"]');

              const main =
                document.querySelector("main") ||
                document.querySelector("#root") ||
                document.querySelector("#app");

              pushText(header);
              pushText(main);

              // fallback
              if (!parts.length) {
                pushText(document.body || document.documentElement);
              }

              return parts.join("\\n");
            } catch (e) {
              return (document.body && (document.body.innerText || document.body.textContent)) || "";
            }
          }"""
        )



        # 2) Список интерактивных элементов (умный сбор + ранжирование + data-webpilot-eid)
        raw = await page.evaluate(
            """({ maxElements }) => {
              // очистим старые id
              try {
                document.querySelectorAll('[data-webpilot-eid]').forEach(el => el.removeAttribute('data-webpilot-eid'));
              } catch (e) {}

              const candidates = new Set();

              const addAll = (sel) => {
                try { document.querySelectorAll(sel).forEach(el => candidates.add(el)); } catch (e) {}
              };

              // базовые интерактивные
              addAll('a[href], button, input:not([type=hidden]), textarea, select, summary');
              // ARIA
              addAll('[role=button], [role=link], [role=menuitem], [role=tab], [role=option], [role=checkbox], [role=radiobutton], [role=switch], [role=combobox], [role=searchbox]');
              // фокусируемые/кликабельные
              addAll('[onclick]');
              addAll('[tabindex]:not([tabindex="-1"])');
              addAll('[contenteditable="true"]');

              const viewportH = window.innerHeight || 800;
              const viewportW = window.innerWidth || 1200;

              const getLabel = (el) => {
                const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim();

                const aria = norm(el.getAttribute('aria-label'));
                if (aria) return aria;

                const labelledBy = norm(el.getAttribute('aria-labelledby'));
                if (labelledBy) {
                  const parts = labelledBy.split(/\\s+/).map(id => {
                    const ref = document.getElementById(id);
                    return ref ? norm(ref.innerText || ref.textContent) : '';
                  }).filter(Boolean);
                  if (parts.length) return parts.join(' ').slice(0, 120);
                }

                const title = norm(el.getAttribute('title'));
                if (title) return title;

                const alt = norm(el.getAttribute('alt'));
                if (alt) return alt;

                const ph = norm(el.getAttribute('placeholder'));
                if (ph) return ph;

                // value для input
                if ('value' in el) {
                  const val = norm(String(el.value || ''));
                  if (val && val.length < 60) return val;
                }

                // связанный label для input
                const tag = (el.tagName || '').toLowerCase();
                if (tag === 'input' || tag === 'textarea' || tag === 'select') {
                  const id = el.getAttribute('id');
                  if (id) {
                    const lab = document.querySelector(`label[for="${CSS.escape(id)}"]`);
                    if (lab) {
                      const t = norm(lab.innerText || lab.textContent);
                      if (t) return t.slice(0, 120);
                    }
                  }
                  const wrapLab = el.closest('label');
                  if (wrapLab) {
                    const t = norm(wrapLab.innerText || wrapLab.textContent);
                    if (t) return t.slice(0, 120);
                  }
                }

                const txt = norm(el.innerText || el.textContent);
                if (txt) return txt.slice(0, 120);

                // последний шанс: name атрибут (часто у форм)
                const nm = norm(el.getAttribute('name'));
                if (nm) return nm.slice(0, 120);

                return '';
              };

              const isVisible = (el, rect) => {
                if (!rect || rect.width < 6 || rect.height < 6) return false;
                const style = getComputedStyle(el);
                if (!style) return false;
                if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return false;
                if (style.pointerEvents === 'none') return false;
                // не отбрасываем полностью вне экрана, но ограничим "слишком далеко"
                if (rect.bottom < -200 || rect.top > viewportH * 3) return false;
                if (rect.right < -200 || rect.left > viewportW * 3) return false;
                return true;
              };

              const scoreEl = (el, rect, label) => {
                let s = 0;
                const tag = (el.tagName || '').toLowerCase();
                const role = (el.getAttribute('role') || '').toLowerCase();

                // base importance
                if (tag === 'button') s += 6;
                if (tag === 'a') s += 5;
                if (tag === 'input' || tag === 'textarea' || tag === 'select') s += 5;
                if (role === 'button' || role === 'link') s += 4;
                if (el.hasAttribute('onclick')) s += 2;

                if (label) s += 4;
                else s -= 4;

                // prefer main content area
                try {
                  if (el.closest('main')) s += 2;
                } catch (e) {}

                // HUGE bonus for modal/dialog content (confirm buttons etc)
                try {
                  if (el.closest('dialog[open],[role="dialog"],[role="alertdialog"],[aria-modal="true"]')) s += 12;
                } catch (e) {}

                // de-prioritize footers/banners/cookies (often noisy)
                try {
                  if (el.closest('footer') || el.closest('[id*="footer"],[class*="footer"]')) s -= 6;
                  if (el.closest('[id*="cookie"],[class*="cookie"],[id*="banner"],[class*="banner"]')) s -= 4;
                } catch (e) {}

                // marketing / ads keywords
                try {
                  const ll = (label || '').toLowerCase();
                  if (/(реклама|promo|промо|интенсив|курс|обучен|вебинар|webinar|advert|ads?)/i.test(ll)) s -= 6;
                } catch (e) {}

                // prioritize elements in viewport
                const inView = rect.top >= 0 && rect.bottom <= viewportH;
                if (inView) s += 4;
                else if (rect.top >= -150 && rect.top <= viewportH * 1.5) s += 2;

                // distance penalty
                s -= Math.min(Math.abs(rect.top), 2000) / 800;

                // external links penalty (+ tracking params)
                try {
                  const hrefAttr = (el.getAttribute && el.getAttribute('href')) ? el.getAttribute('href') : '';
                  if (hrefAttr) {
                    const u = new URL(hrefAttr, location.href);
                    if (u.host && u.host !== location.host) s -= 8;
                    if (u.search && /(utm_|gclid|yclid|fbclid)/i.test(u.search)) s -= 2;
                  }
                } catch (e) {}

                // disabled/aria-disabled
                const dis = !!(el.disabled) || (el.getAttribute('aria-disabled') === 'true');
                if (dis) s -= 10;

                return s;
              };

              const out = [];
              for (const el of candidates) {
                try {
                  const rect = el.getBoundingClientRect();
                  if (!isVisible(el, rect)) continue;

                  const label = getLabel(el);

                  // Если совсем пустой label — берём только если это “явная” кнопка/линк в зоне видимости
                  const tag = (el.tagName || '').toLowerCase();
                  const role = (el.getAttribute('role') || '').toLowerCase();
                  const isExplicit = (tag === 'button' || tag === 'a' || role === 'button' || role === 'link');

                  if (!label && !isExplicit) continue;

                  const disabled = !!(el.disabled) || (el.getAttribute('aria-disabled') === 'true');
                  const s = scoreEl(el, rect, label);

                  out.push({
                    el,
                    score: s,
                    role: role || tag,
                    name: (label || `${tag}`).slice(0, 120),
                    disabled: !!disabled,
                    href: (() => { try { return (el.getAttribute && el.getAttribute('href')) ? el.getAttribute('href') : ''; } catch(e){ return ''; } })(),
                    bbox: [rect.x, rect.y, rect.width, rect.height],
                  });
                } catch (e) {}
              }

              // Сортировка: сначала по score, потом сверху-вниз
              out.sort((a, b) => (b.score - a.score) || (a.bbox[1] - b.bbox[1]) || (a.bbox[0] - b.bbox[0]));

              const result = [];
              const seen = new Set();

              for (let i = 0; i < out.length && result.length < maxElements; i++) {
                const it = out[i];
                const key = `${it.role}|${it.name}|${Math.round(it.bbox[0])}|${Math.round(it.bbox[1])}`;
                if (seen.has(key)) continue;
                seen.add(key);

                const eid = `E${result.length + 1}`;
                try { it.el.setAttribute('data-webpilot-eid', eid); } catch (e) {}

                result.push({
                  eid,
                  role: it.role,
                  name: it.name,
                  disabled: it.disabled,
                  bbox: it.bbox,
                  score: it.score,
                  href: it.href || '',
                });
              }

              return { elements: result };
            }""",
            {"maxElements": max_elements},
        )

        elements: List[ObservedElement] = []

        # Важно: не делаем page.query_selector() для каждого элемента (это дорого на больших DOM).
        for e in (raw.get("elements") or []):
            try:
                elements.append(ObservedElement(
                    eid=e.get("eid", ""),
                    role=e.get("role", ""),
                    name=e.get("name", ""),
                    disabled=bool(e.get("disabled", False)),
                    bbox=e.get("bbox"),
                    score=float(e.get("score", 0.0) or 0.0),
                    href=e.get("href", "") or "",
                ))
            except Exception:
                continue

        return {
            "url": url,
            "title": title,
            "visible_text": visible_text,
            "elements": [e.__dict__ for e in elements],
        }
