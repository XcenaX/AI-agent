import asyncio
import json
import os
import re
from typing import Any, Dict, List
import uuid

from rich.console import Console
from rich.panel import Panel

from .browser.controller import BrowserController, BrowserConfig
from .browser.observe import PageObserver
from .memory import CompactMemory
from .safety import is_sensitive
from .llm.openai_adapter import OpenAIAdapter
from .llm.anthropic_adapter import AnthropicAdapter
from .llm.comet_messages_adapter import CometMessagesAdapter
import logging
import time
from urllib.parse import urlparse
import ast


log = logging.getLogger("webpilot.agent")

# ---------------------------
# Prompt size / latency knobs (env)
# ---------------------------
OBS_MAX_ELEMS = int(os.getenv('OBS_MAX_ELEMS', '25'))  # elements shown to LLM
OBS_ELEM_NAME_CHARS = int(os.getenv('OBS_ELEM_NAME_CHARS', '60'))
OBS_INCLUDE_TEXT = os.getenv('OBS_INCLUDE_TEXT', '1') == '1'
OBS_TEXT_CHARS = int(os.getenv('OBS_TEXT_CHARS', '450'))
OBS_INCLUDE_HREF = os.getenv('OBS_INCLUDE_HREF', '0') == '1'

# how much the browser observer collects (affects CPU + tokens)
OBS_MAX_ELEMS_OBSERVER = int(os.getenv('OBS_MAX_ELEMS_OBSERVER', '60'))
OBS_MAX_TEXT_CHARS_OBSERVER = int(os.getenv('OBS_MAX_TEXT_CHARS_OBSERVER', '6000'))  # 0 = skip text extraction

# OpenAI: keep only last N steps in the message history to avoid latency from huge prompts
OPENAI_MAX_HISTORY_STEPS = int(os.getenv('OPENAI_MAX_HISTORY_STEPS', os.getenv('MAX_HISTORY', '12')))

# Facts + summary injection
FACTS_IN_PROMPT = os.getenv('FACTS_IN_PROMPT', '1') == '1'
FACTS_MAX_ITEMS = int(os.getenv('FACTS_MAX_ITEMS', '10'))
FACT_MAX_CHARS = int(os.getenv('FACT_MAX_CHARS', '90'))
STATE_SUMMARY_MAX_CHARS = int(os.getenv('STATE_SUMMARY_MAX_CHARS', '180'))
MEMORY_PROMPT_CHARS = int(os.getenv('MEMORY_PROMPT_CHARS', '900'))
DONE_MAX_ITEMS = int(os.getenv('DONE_MAX_ITEMS', '20'))

# Behavior toggles (env)
ASK_USER_MODE = os.getenv('ASK_USER_MODE', 'auto').strip().lower()  # interactive|auto|halt
ROLE_GUARD = os.getenv('ROLE_GUARD', '1') != '0'
ROLE_KEYWORDS = os.getenv('ROLE_KEYWORDS', '').strip()  # comma-separated extra role keywords
AUTO_TASK_TRACK = os.getenv('AUTO_TASK_TRACK', '1') == '1'  # auto-detect milestones from page text
FACT_STICKY_KEYS = os.getenv('FACT_STICKY_KEYS', 'role,resume_summary,resume_skills,resume_experience,resume_education,target_role,target_stack')
_STICKY_KEYS_SET = {k.strip() for k in FACT_STICKY_KEYS.split(',') if k.strip()}

# Auto extractor (extra LLM call) — off by default for speed
AUTO_EXTRACT = os.getenv('AUTO_EXTRACT', '1') == '1'
AUTO_EXTRACT_EVERY = int(os.getenv('AUTO_EXTRACT_EVERY', '3'))

# ---------------------------
# LLM Trace (JSONL) — for debugging prompt/obs truncation
# ---------------------------
LLM_TRACE = os.getenv("LLM_TRACE", "1") == "1"
LLM_TRACE_PATH = os.getenv("LLM_TRACE_PATH", "logs/llm_trace_{run_id}.jsonl")
LLM_TRACE_FULL = os.getenv("LLM_TRACE_FULL", "0") == "1"  # 0=only first+last messages
LLM_TRACE_MAX_MSG_CHARS = int(os.getenv("LLM_TRACE_MAX_MSG_CHARS", "12000"))
LLM_TRACE_MAX_MESSAGES = int(os.getenv("LLM_TRACE_MAX_MESSAGES", "8"))
LLM_TRACE_ELEMS_SAMPLE = int(os.getenv("LLM_TRACE_ELEMS_SAMPLE", "35"))
LLM_TRACE_TEXT_PREVIEW = int(os.getenv("LLM_TRACE_TEXT_PREVIEW", "500"))
LLM_TRACE_REDACT_PII = os.getenv("LLM_TRACE_REDACT_PII", "0") == "1"


_TOOLNAME_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


def _clip(s: str, n: int) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else (s[:n] + "…")

def _is_typeable_role(role: str) -> bool:
    r = (role or "").lower()
    return any(x in r for x in ("input", "textarea", "searchbox", "combobox")) or r == "textbox"


def _smart_text_clip(url: str, title: str, text: str, max_chars: int) -> str:
    """Return a compact text snippet that is more informative than a plain head() cut."""
    if not text or max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text

    low = text.lower()
    url_l = (url or "").lower()
    title_l = (title or "").lower()

    if "/resume/" in url_l or "резюме" in title_l:
        needles = ["ключевые навыки", "навыки", "стек", "опыт работы", "опыт", "проекты", "обо мне", "образование"]
    elif "/vacancy/" in url_l or "вакансия" in title_l:
        needles = ["требования", "обязанности", "условия", "ключевые навыки", "стек", "технолог", "python", "backend", "api", "fastapi", "django"]
    else:
        needles = ["ваканс", "отклик", "подобрал", "подобрали", "search", "login", "войти"]

    hit = -1
    for n in needles:
        i = low.find(n)
        if i != -1:
            hit = i
            break

    head_len = max(120, max_chars // 2)
    head = text[:head_len]
    if hit == -1 or hit < head_len:
        return text[:max_chars]

    win_len = max_chars - len(head) - 10
    start = max(0, hit - 120)
    window = text[start:start + win_len]
    return head + "\n...\n" + window


def _looks_like_apply_element_name(name: str) -> bool:
    n = (name or "").lower()
    return any(x in n for x in ("отклик", "откликнуться", "apply", "submit", "подать"))


def _find_elem_by_eid(obs: dict, eid: str) -> dict:
    for e in (obs or {}).get("elements") or []:
        if e.get("eid") == eid:
            return e
    return {}


def _derive_role_keywords(facts: dict) -> List[str]:
    """Try to build a small set of keywords that represent the target role/stack."""
    kws = set()

    def add_words(s: str):
        if not s:
            return
        s = s.lower()
        for w in re.findall(r"[a-z0-9\+\#\.-]{2,}", s):
            if len(w) <= 24:
                kws.add(w)
        for w in re.findall(r"[а-яё]{3,}", s):
            if w in ("работа", "опыт", "лет", "год", "компания", "проект", "удаленно", "удалённо", "офис"):
                continue
            if len(w) <= 24:
                kws.add(w)

    role = ""
    if isinstance(facts, dict):
        if "role" in facts and isinstance(facts["role"], dict):
            role = str(facts["role"].get("value") or "")
        if not role and "resume_summary" in facts and isinstance(facts["resume_summary"], dict):
            role = str(facts["resume_summary"].get("value") or "")

    add_words(role)

    role_low = role.lower()
    if any(x in role_low for x in ("backend", "бэкенд", "back-end", "сервер")):
        kws.update({"backend", "бэкенд", "api", "python", "fastapi", "django", "developer", "разработчик"})
    if "python" in role_low:
        kws.update({"python", "fastapi", "django"})
    if "fullstack" in role_low or "full-stack" in role_low:
        kws.update({"fullstack", "react", "frontend", "backend"})
    if "frontend" in role_low or "react" in role_low:
        kws.update({"frontend", "react", "javascript", "typescript"})

    if ROLE_KEYWORDS:
        for p in ROLE_KEYWORDS.split(","):
            p = p.strip().lower()
            if p:
                kws.add(p)

    prio = [w for w in ("backend","python","fastapi","django","api","developer","разработчик","бэкенд","react") if w in kws]
    rest = [w for w in kws if w not in prio]
    rest = sorted(rest)[:8]
    return (prio + rest)[:12]


def _vacancy_title_matches(title: str, role_keywords: List[str]) -> bool:
    tl = (title or "").lower().replace("вакансия", " ")
    if not role_keywords:
        return True
    hits = sum(1 for kw in role_keywords if kw and kw in tl)
    if hits >= 1:
        return True
    if any(x in tl for x in ("разработчик", "developer", "engineer")) and any(x in role_keywords for x in ("backend","python","developer","разработчик","бэкенд")):
        return True
    return False


def _auto_answer_for_question(question: str) -> str | None:
    q = (question or "").lower()
    if any(x in q for x in ("captcha", "капча", "смс", "код", "2fa", "однораз", "пароль", "логин", "подтверд")):
        return None
    if any(x in q for x in ("сопровод", "cover letter", "мотивац", "письмо")):
        return "Сформируй сопроводительное письмо автоматически на основе моего резюме и текста вакансии; если имя не указано — пиши без имени."
    return ""

def _ensure_dir_for(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _jsonl_append(path: str, obj: dict) -> None:
    try:
        _ensure_dir_for(path)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        log.exception("[trace] failed to write jsonl to %s: %s", path, e)


def _norm_messages_for_log(messages):
    """Return JSON-serializable, size-capped messages."""
    def norm_block(block):
        if not isinstance(block, dict):
            return _clip(block, 2000)
        b = dict(block)
        if b.get("type") == "text":
            b["text"] = _clip(b.get("text", ""), LLM_TRACE_MAX_MSG_CHARS)
        if b.get("type") == "tool_result":
            # tool_result content can be huge
            if "content" in b:
                b["content"] = _clip(b.get("content", ""), 3000)
        return b

    out = []
    for m in (messages or []):
        if not isinstance(m, dict):
            out.append(_clip(m, 2000))
            continue
        mm = dict(m)
        c = mm.get("content")
        if isinstance(c, str):
            mm["content"] = _clip(c, LLM_TRACE_MAX_MSG_CHARS)
        elif isinstance(c, list):
            mm["content"] = [norm_block(x) for x in c][:20]
        else:
            mm["content"] = _clip(c, 2000)
        out.append(mm)

    if LLM_TRACE_FULL:
        return out[:LLM_TRACE_MAX_MESSAGES]

    # default: first + last few (to keep file small but useful)
    if len(out) <= LLM_TRACE_MAX_MESSAGES:
        return out
    keep_tail = max(1, LLM_TRACE_MAX_MESSAGES - 1)
    return out[:1] + out[-keep_tail:]


def _normalize_tool_call(name: str, input_args: dict | None):
    raw = (name or "").strip()

    # 1) если прилетело "browser_scroll({'dy': 600})</arg_value>"
    #    вырезаем теги
    raw = re.sub(r"<[^>]+>", "", raw).strip()

    # 2) вытащим базовое имя
    m = _TOOLNAME_RE.match(raw)
    base = m.group(0) if m else raw

    args = dict(input_args or {})

    # 3) если args пустые, а внутри есть "(...)" — попробуем распарсить
    if not args and "(" in raw and ")" in raw:
        inside = raw.split("(", 1)[1].rsplit(")", 1)[0].strip()
        if inside:
            try:
                parsed = ast.literal_eval(inside)  # понимает {'dy': 600}
                if isinstance(parsed, dict):
                    args = parsed
            except Exception:
                pass

    return base, args


def _compact_args(args: dict) -> str:
    if not isinstance(args, dict):
        return str(args)
    a = dict(args)
    if "text" in a and isinstance(a["text"], str):
        a["text"] = (a["text"][:40] + "…") if len(a["text"]) > 40 else a["text"]
    return str(a)


def _compact_obs(obs: dict) -> dict:
    # Keep observation tiny to reduce tokens/latency.
    text = (obs.get('visible_text') or '')
    if not OBS_INCLUDE_TEXT:
        text = ''
    else:
        text = _smart_text_clip(obs.get('url') or '', obs.get('title') or '', text, OBS_TEXT_CHARS)

    elems = obs.get('elements') or []

    def role_is_input(role: str) -> bool:
        r = (role or '').lower()
        return any(x in r for x in ['input', 'textarea', 'select', 'searchbox', 'combobox'])

    def in_view(e) -> bool:
        b = e.get('bbox') or []
        if len(b) != 4:
            return False
        _, y, _, h = b
        return -150 <= y <= 1000 + h

    def is_clickable(role: str) -> bool:
        r = (role or '').lower()
        return any(x in r for x in ['button', 'link', 'menuitem', 'tab', 'option', 'checkbox', 'radio', 'switch'])

    def score(e) -> float:
        try:
            return float(e.get('score', 0.0) or 0.0)
        except Exception:
            return 0.0

    def looks_like_noise(e) -> bool:
        name = (e.get('name') or '').lower()
        href = (e.get('href') or '').lower()
        if any(w in name for w in ['реклама', 'promo', 'промо', 'интенсив', 'курс', 'обучен', 'вебинар', 'webinar', 'advert', 'ads']):
            return True
        if any(x in href for x in ['utm_', 'gclid', 'yclid', 'fbclid']):
            return True
        return False

    inputs = [e for e in elems if role_is_input(e.get('role', ''))]
    # Always try to keep navigation-critical buttons/links in view for the LLM
    important_kw = ['подобрал', 'подобрали', 'подходящ', 'ваканс', 'отклик', 'откликн', 'apply', 'submit', 'search', 'поиск']
    important = [e for e in elems if any(k in ((e.get('name') or '').lower()) for k in important_kw)]
    visible = [e for e in elems if in_view(e)]
    clickables = [e for e in elems if is_clickable(e.get('role', '')) and not looks_like_noise(e)]
    clickables = sorted(clickables, key=lambda e: score(e), reverse=True)[:OBS_MAX_ELEMS]

    seen = set()
    merged = []
    for group in (inputs, important, visible, clickables):
        for e in group:
            eid = e.get('eid')
            if not eid or eid in seen:
                continue
            seen.add(eid)
            merged.append(e)
            if len(merged) >= OBS_MAX_ELEMS:
                break
        if len(merged) >= OBS_MAX_ELEMS:
            break

    def slim_elem(e: dict) -> dict:
        name = (e.get("name") or "")[:OBS_ELEM_NAME_CHARS]
        out = {
            "eid": e.get("eid"),
            "role": e.get("role"),
            "name": name,
            "disabled": bool(e.get("disabled", False)),
        }

        # помогает не кликать “по названию” в не-то
        href = e.get("href") or ""
        if href:
            out["href"] = href[:160]

        # дешёвая подсказка “где элемент” (ускоряет выбор)
        b = e.get("bbox") or []
        if len(b) == 4:
            out["y"] = round(float(b[1]), 1)

        return out


    return {
        'url': obs.get('url'),
        'title': (obs.get('title') or '')[:120],
        'visible_text': text,
        'elements': [slim_elem(e) for e in merged[:OBS_MAX_ELEMS]],
        'elements_count': len(elems),
    }

def obs_to_prompt_min(o: dict) -> str:
    lines = [
        f"URL: {o.get('url','')}",
        f"TITLE: {o.get('title','')}",
        f"ELEMENTS_SHOWN: {len(o.get('elements') or [])} / TOTAL_COLLECTED: {o.get('elements_count', 0)}",
        "ELEMENTS (eid|role|name|href):"
    ]

    # показываем столько, сколько реально отправляем в LLM
    for e in (o.get("elements") or [])[:OBS_MAX_ELEMS]:
        name = (e.get("name") or "").replace("\n"," ")
        href = (e.get("href") or "")
        if href:
            href = href[:120]
        lines.append(f"- {e.get('eid')}|{e.get('role')}|{name}|{href}")

    # вот это было ключевым: текст страницы
    if OBS_INCLUDE_TEXT and (o.get("visible_text") or "").strip():
        lines.append("TEXT (visible_text preview):")
        lines.append(o["visible_text"])

    # если мы заранее распарсили число — тоже покажем
    if o.get("found_count_hint") is not None:
        lines.append(f"FOUND_COUNT_HINT: {o['found_count_hint']}")

    return "\n".join(lines)



def _trim_openai_items(items: List[Dict[str, Any]], keep_steps: int = OPENAI_MAX_HISTORY_STEPS) -> List[Dict[str, Any]]:
    """Keep the initial Task message + last N user-step blocks (and all tool results between them).

    This avoids sending an ever-growing prompt to the LLM, which is the #1 cause of slow inference.
    """
    if not items or keep_steps <= 0:
        return items

    # user messages are natural step boundaries (we add exactly one per step)
    user_idxs = [i for i, it in enumerate(items) if isinstance(it, dict) and it.get("role") == "user"]
    if len(user_idxs) <= 1 + keep_steps:
        return items

    cutoff = user_idxs[-keep_steps]  # start of the oldest step we keep
    if cutoff <= 0:
        return items

    return [items[0]] + items[cutoff:]


console = Console()

def _root_domain(url: str) -> str:
    """Best-effort 'registrable' domain. Not perfect for all TLDs, but good enough for guardrails."""
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return ""
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return ""
    parts = host.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host


async def ainput(prompt: str = "") -> str:
    return await asyncio.to_thread(input, prompt)


class WebPilotAgent:
    def __init__(self, provider: str, model: str, user_data_dir: str, max_steps: int = 80):
        self.max_steps = max_steps
        self.browser = BrowserController(BrowserConfig(user_data_dir=user_data_dir))
        self.observer = PageObserver()
        self.memory = CompactMemory()
        self.provider = (provider or "openai").strip().lower()
        self.facts = {} # Контекст
        self.done = []
        self.state_summary = ""
        self.last_action = ""
        # Domain guard: prevent accidental navigation to ads/external sites
        self.domain_guard_enabled = os.getenv("DOMAIN_GUARD", "1") != "0"
        self.allowed_root_domains = set()
        env_domains = os.getenv("ALLOWED_DOMAINS", "")
        for d in [x.strip().lower() for x in env_domains.split(",") if x.strip()]:
            # allow both full host and root domain; we normalize to root domain
            if "://" in d:
                self.allowed_root_domains.add(_root_domain(d))
            else:
                # if user gives "hh.ru", keep as-is
                self.allowed_root_domains.add(d.lstrip("."))
        self.allow_pii_memory = os.getenv("ALLOW_PII_MEMORY", "0") == "1"

        self.run_id = os.getenv("RUN_ID") or uuid.uuid4().hex[:10]
        self.llm_trace_path = (
            LLM_TRACE_PATH.format(run_id=self.run_id)
            if "{run_id}" in LLM_TRACE_PATH else LLM_TRACE_PATH
        )

        # DEBUG: confirm trace is enabled + where it writes
        if LLM_TRACE:
            log.info("[trace] LLM_TRACE=1, path=%s", self.llm_trace_path)
            _jsonl_append(self.llm_trace_path, {"kind": "startup", "run_id": self.run_id, "ts": time.time()})
        else:
            log.info("[trace] LLM_TRACE is OFF (set LLM_TRACE=1 to enable)")


        if self.provider == "cometapi":
            api_key = os.getenv("COMETAPI_KEY")
            base_url = os.getenv("COMETAPI_BASE_URL", "https://api.cometapi.com/v1")
            self.llm = CometMessagesAdapter(model=model, api_key=api_key, base_url=base_url)
        elif self.provider == "zai":
            api_key = os.getenv("ZAI_API_KEY")
            base_url = os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/anthropic/v1")
            model_name = (model or "").strip() or os.getenv("ZAI_MODEL", "glm-4.6")
            self.llm = CometMessagesAdapter(model=model_name, api_key=api_key, base_url=base_url)
        elif self.provider == "openai":
            self.llm = OpenAIAdapter(model=model)
        elif self.provider == "anthropic":
            self.llm = AnthropicAdapter(model=model)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")      

    def _trace_llm(self, *, step: int, kind: str, messages, obs: dict, obs_small: dict,
                   latency_s: float, resp: Any, tool_calls, text: str):
        if not LLM_TRACE:
            return

        # tool_calls may be objects (openai) or dicts (anthropic-like)
        calls_out = []
        for c in (tool_calls or []):
            try:
                if isinstance(c, dict):
                    calls_out.append({"name": c.get("name"), "input": c.get("input")})
                else:
                    calls_out.append({
                        "name": getattr(c, "name", None),
                        "arguments": _clip(getattr(c, "arguments", None), 2000),
                    })
            except Exception:
                calls_out.append(_clip(str(c), 500))

        # usage if exists
        usage = None
        try:
            if isinstance(resp, dict):
                usage = resp.get("usage")
            else:
                usage = getattr(resp, "usage", None)
        except Exception:
            usage = None

        evt = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int((time.time()%1)*1000):03d}Z",
            "run_id": self.run_id,
            "step": step,
            "kind": kind,
            "provider": self.provider,
            "model": getattr(self.llm, "model", None) or os.getenv("MODEL") or os.getenv("OPENAI_MODEL") or "",
            "page": {
                "url": (obs_small or {}).get("url") or (obs or {}).get("url"),
                "title": (obs_small or {}).get("title") or (obs or {}).get("title"),
            },
            "obs_stats": {
                "observer_max_text_chars": OBS_MAX_TEXT_CHARS_OBSERVER,
                "compact_text_chars": OBS_TEXT_CHARS,
                "visible_text_collected_chars": len((obs or {}).get("visible_text") or ""),
                "visible_text_sent_chars": len((obs_small or {}).get("visible_text") or ""),
                "elements_collected": len((obs or {}).get("elements") or []),
                "elements_sent": len((obs_small or {}).get("elements") or []),
                "obs_max_elems_observer": OBS_MAX_ELEMS_OBSERVER,
                "obs_max_elems_sent": OBS_MAX_ELEMS,
            },
            # mini-sample to debug "не видит куда кликать"
            "obs_samples": {
                "visible_text_preview": _clip((obs or {}).get("visible_text") or "", LLM_TRACE_TEXT_PREVIEW),
                "elements_preview": [
                    {
                        "eid": e.get("eid"),
                        "role": e.get("role"),
                        "name": _clip(e.get("name") or "", 140),
                        "href": _clip(e.get("href") or "", 200),
                        "bbox": e.get("bbox"),
                        "score": e.get("score"),
                    }
                    for e in ((obs or {}).get("elements") or [])[:LLM_TRACE_ELEMS_SAMPLE]
                ],
            },
            "messages": _norm_messages_for_log(messages),
            "latency_s": round(float(latency_s or 0.0), 3),
            "response": {
                "text_preview": _clip(text or "", 1200),
                "tool_calls": calls_out[:30],
                "usage": usage,
            },
        }

        _jsonl_append(self.llm_trace_path, evt)    

    def _get_elem_from_last_obs(self, eid: str):
        elems = (getattr(self, "_last_obs", {}) or {}).get("elements", [])
        return next((e for e in elems if e.get("eid") == eid), None)

    def _find_elements(self, query: str, limit: int = 8):
        q = (query or "").lower().strip()
        elems = (getattr(self, "_last_obs", {}) or {}).get("elements", [])
        scored = []

        q_words = [w for w in q.split() if w]

        for e in elems:
            name = (e.get("name") or "").lower()
            role = (e.get("role") or "").lower()
            if not name:
                continue

            s = 0
            if q and q in name:
                s += 5
            for w in q_words:
                if w in name:
                    s += 1
            if "button" in role:
                s += 0.5
            if "link" in role:
                s += 0.2

            scored.append((s, e))

        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for score, e in scored:
            if score <= 0:
                break
            out.append({"eid": e.get("eid"), "name": e.get("name"), "role": e.get("role")})
            if len(out) >= limit:
                break
        return out

    def _auto_save_fact(self, key: str, value: str, why: str = "auto-extracted", source_url: str = "") -> bool:
        key = (key or "").strip()[:60]
        value = (value or "").strip()[:800]
        if not key or not value:
            return False

        now = time.time()

        if not hasattr(self, "facts") or self.facts is None:
            self.facts = {}

        prev = self.facts.get(key, {}).get("value")
        if prev == value:
            # освежим ts, чтобы “живые” факты оставались наверху
            self.facts[key]["ts"] = now
            if source_url:
                self.facts[key]["source_url"] = source_url[:300]
            return False

        sticky = (key in _STICKY_KEYS_SET) or key.startswith("resume_") or key.startswith("target_") or key == "role"
        self.facts[key] = {
            "value": value,
            "why": (why or "")[:200],
            "ts": now,
            "source_url": (source_url or "")[:300],
            "sticky": bool(sticky),
        }
        return True

    def _infer_goal_targets(self, goal: str) -> Dict[str, int]:
        """Infer simple numeric targets from the task (e.g., 'find 3 vacancies', 'make 3 applies')."""
        if hasattr(self, "_goal_targets_cache") and isinstance(getattr(self, "_goal_targets_cache"), dict):
            return self._goal_targets_cache

        goal_low = (goal or "").lower()

        num_words = {
            "один": 1, "одна": 1,
            "два": 2, "две": 2,
            "три": 3,
            "четыре": 4,
            "пять": 5,
            "шесть": 6,
            "семь": 7,
            "восемь": 8,
            "девять": 9,
            "десять": 10,
        }

        def find_num_for(kind: str) -> int | None:
            if kind == "vacancies":
                pats = [
                    r"(\d+)\s*(?:ваканс|вакансии|vacanc)",
                    r"(один|одна|два|две|три|четыре|пять|шесть|семь|восемь|девять|десять)\s*(?:ваканс|вакансии|vacanc)",
                ]
            else:
                pats = [
                    r"(\d+)\s*(?:отклик|отклика|откликов|apply|application)",
                    r"(один|одна|два|две|три|четыре|пять|шесть|семь|восемь|девять|десять)\s*(?:отклик|отклика|откликов|apply|application)",
                ]
            for pat in pats:
                m = re.search(pat, goal_low)
                if not m:
                    continue
                token = m.group(1)
                if token.isdigit():
                    return int(token)
                return num_words.get(token)
            return None

        vacancies = find_num_for("vacancies")
        applies = find_num_for("applies")

        # fallback: first explicit digit in the task (common pattern: "найди 3 ... сделай 3 ...")
        if vacancies is None or applies is None:
            m_any = re.search(r"\b(\d+)\b", goal_low)
            if m_any:
                n = int(m_any.group(1))
                vacancies = vacancies or n
                applies = applies or n

        self._goal_targets_cache = {"vacancies": vacancies or 0, "applies": applies or 0}
        return self._goal_targets_cache

    def _add_done_local(self, kind: str, target: str, note: str = "auto") -> None:
        kind = (kind or "").strip()[:40]
        target = (target or "").strip()[:300]
        note = (note or "").strip()[:300]
        if not kind or not target:
            return
        if not hasattr(self, "done") or self.done is None:
            self.done = []
        self.done.append({"kind": kind, "target": target, "note": note})
        uniq = {(d.get("kind"), d.get("target")): d for d in self.done if isinstance(d, dict)}
        self.done = list(uniq.values())

    def _auto_track_from_obs(self, goal: str, obs_full: Dict[str, Any]) -> None:
        """Heuristics so the agent doesn't 'forget' progress even if the LLM skips task_mark_done."""
        if not AUTO_TASK_TRACK:
            return
        try:
            url = str((obs_full or {}).get("url") or "")
            txt = str((obs_full or {}).get("visible_text") or "")
            tl = txt.lower()

            # track current vacancy
            if "/vacancy/" in url:
                self.current_vacancy_url = url
                self._add_done_local("seen_vacancy", url, note="auto-seen")

            # track resume page
            if ("/resume/" in url) or ("/applicant/resume" in url):
                self._add_done_local("seen_resume", url, note="auto-seen")

            # detect apply success
            success_phrases = [
                "отклик отправлен", "ваш отклик отправлен", "отклик направлен",
                "вы откликнулись", "отклик успешно", "отклик принят",
            ]
            if any(p in tl for p in success_phrases):
                target = getattr(self, "current_vacancy_url", "") or url
                if target:
                    self._add_done_local("applied", target, note="auto-detected")

            # detect "already applied" (also prevents repeats)
            already_phrases = ["вы уже откликались", "отклик уже отправлен", "вы откликнулись на эту вакансию"]
            if any(p in tl for p in already_phrases):
                target = getattr(self, "current_vacancy_url", "") or url
                if target:
                    self._add_done_local("applied", target, note="auto-already")

        except Exception:
            pass

    def _progress_block(self, goal: str) -> str:
        targets = self._infer_goal_targets(goal)
        done = getattr(self, "done", []) or []
        applied = [d for d in done if str(d.get("kind", "")).startswith("applied")]
        seen_vac = [d for d in done if d.get("kind") == "seen_vacancy"]
        seen_resume = [d for d in done if d.get("kind") == "seen_resume"]

        applies_target = int(targets.get("applies") or 0)
        vacancies_target = int(targets.get("vacancies") or 0)

        applied_urls = []
        for d in applied:
            t = d.get("target")
            if t and t not in applied_urls:
                applied_urls.append(t)

        lines = [
            f"- seen_resume: {len(seen_resume)}",
            f"- vacancies_seen: {len(seen_vac)}" + (f" / {vacancies_target}" if vacancies_target else ""),
            f"- applied: {len(applied_urls)}" + (f" / {applies_target}" if applies_target else ""),
        ]
        if applied_urls:
            lines.append("- applied_urls: " + "; ".join(applied_urls[:6]))
        return "\n".join(lines)

    async def _maybe_auto_extract(self, goal: str, obs_full: Dict[str, Any], obs_small: Dict[str, Any], step: int) -> None:
        """Occasionally extract reusable facts from the current page and persist them."""
        if not AUTO_EXTRACT:
            return

        try:
            sig = (obs_small.get("url"), obs_small.get("title"))
            last_sig = getattr(self, "_last_extract_sig", None)
            last_step = getattr(self, "_last_extract_step", 0)

            should = (sig != last_sig) or (step - last_step >= AUTO_EXTRACT_EVERY)
            if not should:
                return

            def _redact_pii(s: str) -> str:
                # very lightweight PII redaction (email/phone)
                s = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "<email>", s)
                s = re.sub(r"\+?\d[\d\s\-\(\)]{7,}\d", "<phone>", s)
                return s

            # Дадим extractor’у БОЛЬШЕ текста, чем main-модель видит в compact obs
            full_text = (obs_full or {}).get("visible_text") or ""

            def _pick_snippet(t: str, max_chars: int) -> str:
                t = t or ""
                if len(t) <= max_chars:
                    return t
                anchors = [
                    "ключевые навыки", "навыки", "опыт работы", "опыт", "проекты", "стек",
                    "skills", "experience", "projects", "tech stack",
                    "требования", "обязанности", "описание вакансии", "requirements", "responsibilities",
                ]
                tl = t.lower()
                pos = -1
                for a in anchors:
                    p = tl.find(a)
                    if p != -1:
                        pos = p
                        break
                if pos != -1:
                    start = max(pos - max_chars // 3, 0)
                    return t[start : start + max_chars]
                half = max_chars // 2
                return t[:half] + "\n...\n" + t[-half:]

            def _build_resume_summary(t: str) -> str:
                # lightweight HH resume parser (no PII)
                t = t or ""
                lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
                noise = {
                    "чаты", "резюме и профиль", "отклики", "сервисы", "помощь", "поиск",
                    "создать резюме", "мои резюме", "редактировать", "контакты",
                    "мобильный телефон", "электронная почта",
                }
                clean = [ln for ln in lines if ln.lower() not in noise]

                title = ""
                for ln in clean:
                    if 3 <= len(ln) <= 60 and re.search(r"[A-Za-zА-Яа-я]", ln) and not re.search(r"\d", ln):
                        title = ln
                        break

                salary = ""
                m_sal = re.search(r"(\d[\d\s]{3,})\s*₸", t)
                if m_sal:
                    salary = re.sub(r"\s+", "", m_sal.group(1)) + " ₸"

                employment = ""
                work_format = ""
                for ln in clean:
                    if ln.startswith("Тип занятости:"):
                        employment = ln.replace("Тип занятости:", "").strip()
                    elif ln.startswith("Формат работы:"):
                        work_format = ln.replace("Формат работы:", "").strip()

                parts = []
                if title:
                    parts.append(f"Цель/должность: {title}")
                if salary:
                    parts.append(f"Желаемая ЗП: {salary} на руки")
                if employment:
                    parts.append(f"Занятость: {employment}")
                if work_format:
                    parts.append(f"Формат: {work_format}")

                # try to include a short skills/experience hint if present
                for kw in ["Ключевые навыки", "Навыки", "Опыт работы", "Опыт", "Стек"]:
                    pos = t.lower().find(kw.lower())
                    if pos != -1:
                        hint = re.sub(r"\s+", " ", t[pos : pos + 900]).strip()
                        parts.append(hint[:220])
                        break

                return " | ".join(parts)[:750]

            obs_for_extract = dict(obs_small)
            max_chars = int(os.getenv("EXTRACT_OBS_TEXT_CHARS", "3600"))
            obs_for_extract["visible_text"] = _pick_snippet(full_text, max_chars=max_chars)
            obs_for_extract["elements"] = (obs_small.get("elements") or [])[:60]

            t_ex = time.time()
            result = await asyncio.to_thread(self.llm.call_subagent, "extractor", goal, obs_for_extract, self.memory.render())

            # normalize to dict
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except Exception:
                    result = {}

            facts = (result or {}).get("facts") or []
            saved = 0
            for f in facts:
                if not isinstance(f, dict):
                    continue

                k = str(f.get("key", "") or "").strip()
                v = _redact_pii(str(f.get("value", "") or "").strip())

                if self._auto_save_fact(k, v, why="auto-extracted", source_url=str(obs_small.get("url") or "")):
                    saved += 1
                    if saved >= FACTS_MAX_ITEMS:
                        break

            # Resume-specific fallback: save a compact summary so later steps can match vacancies.
            url_now = str(obs_small.get("url") or "")
            goal_low = (goal or "").lower()
            if ("/resume/" in url_now or "/applicant/resume" in url_now) and ("резюме" in goal_low or "resume" in goal_low):
                try:
                    if not getattr(self, "facts", None) or not self.facts.get("resume_summary", {}).get("value"):
                        summary = _redact_pii(_build_resume_summary(full_text))
                        if summary:
                            if self._auto_save_fact("resume_summary", summary, why="auto-resume-summary", source_url=url_now):
                                saved += 1
                except Exception:
                    pass

            missing = (result or {}).get("missing_info")
            if isinstance(missing, str) and missing.strip():
                self.memory.add(f"Extractor missing_info: {missing.strip()[:200]}")

            self._last_extract_sig = sig
            self._last_extract_step = step

            if saved:
                self.memory.add(f"Auto-saved {saved} facts from page.")

        except Exception as e:
            self.memory.add(f"Auto-extract error: {str(e)[:200]}")


    async def _post_nav_wait(self) -> None:
        page = self.browser.page
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=15000)
        except Exception:
            pass
        try:
            await page.wait_for_load_state("networkidle", timeout=5000)
        except Exception:
            pass
        await page.wait_for_timeout(600)

    async def run(self, goal: str) -> None:
        await self.browser.start()

        # Seed domain guard allowlist from any explicit URLs in the Task
        try:
            for u in re.findall(r"https?://\S+", goal or ""):
                self.allowed_root_domains.add(_root_domain(u))
        except Exception:
            pass

        # Also allow plain domains mentioned in Task (e.g., hh.ru)
        try:
            for d in re.findall(r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b", goal or ""):
                self.allowed_root_domains.add(_root_domain("https://" + d))
        except Exception:
            pass

        if self.provider in ("anthropic", "cometapi", "zai"):
            await self._run_anthropic(goal)
            return
        
        try:
            input_items: List[Dict[str, Any]] = [
                {"role": "user", "content": f"Task: {goal}\nStart now."}
            ]

            for step in range(1, self.max_steps + 1):
                await self.browser.ensure_active_page()
                obs = await self.observer.observe(self.browser.page, max_elements=OBS_MAX_ELEMS_OBSERVER, max_text_chars=OBS_MAX_TEXT_CHARS_OBSERVER)
                self._last_obs = obs
                obs_small = _compact_obs(obs)
                self._auto_track_from_obs(goal, obs)


                # lightweight state summary (kept short, survives history truncation)
                self.state_summary = (f"On page: {(obs.get('title') or '—')[:80]} | {obs.get('url') or ''} | last_action: {self.last_action or '—'}")[:STATE_SUMMARY_MAX_CHARS]

                # Auto-extract reusable facts occasionally
                await self._maybe_auto_extract(goal, obs, obs_small, step)
                log.info("[bold cyan]STEP %d[/] url=%s title=%s elems=%d", step, obs.get("url"), (obs.get("title") or "")[:80], len(obs.get("elements", [])))

                prompt = obs_to_prompt_min(obs_small)

                # --- anti stuck ---
                if not hasattr(self, "_same_page"):
                    self._same_page = 0
                    self._last_sig = None


                sig = (obs.get("url"), obs.get("title"))
                if sig == self._last_sig:
                    self._same_page += 1
                else:
                    self._same_page = 0
                    self._last_sig = sig


                # если 3 шага подряд на той же странице получим совет навигатора и подкинем в контекст
                if self._same_page >= 3:
                    t_nav = time.time()
                    nav = await asyncio.to_thread(
                        self.llm.call_subagent, "navigator", goal, obs_small, self.memory.render()
                    )

                    self.memory.add(f"Navigator advice: {nav}")
                    input_items.append({
                        "role": "user",
                        "content": f"Navigator advice (JSON): {json.dumps(nav, ensure_ascii=False)}"
                    })
                    self._same_page = 0

                    self._trace_llm(
                        step=step,
                        kind="navigator",
                        messages=[{"role": "user", "content": f"Task: {goal}\nObs: {json.dumps(obs_small, ensure_ascii=False)}"}],
                        obs=obs,
                        obs_small=obs_small,
                        latency_s=(time.time() - t_nav),
                        resp={"content": [{"type": "text", "text": json.dumps(nav, ensure_ascii=False)}]},
                        tool_calls=[],
                        text=json.dumps(nav, ensure_ascii=False),
                    )


                mem = (self.memory.render() or "")
                if MEMORY_PROMPT_CHARS and len(mem) > MEMORY_PROMPT_CHARS:
                    mem = mem[-MEMORY_PROMPT_CHARS:]

                # Prepare compact context blocks
                facts_text = ""
                if FACTS_IN_PROMPT and getattr(self, 'facts', None):
                    items_all = [(k, v) for k, v in (self.facts or {}).items() if isinstance(v, dict)]
                    sticky_items = [(k, v) for k, v in items_all if v.get("sticky")]
                    sticky_items.sort(key=lambda kv: float(kv[1].get("ts", 0) or 0), reverse=True)
                    other_items = [(k, v) for k, v in items_all if not v.get("sticky")]
                    other_items.sort(key=lambda kv: float(kv[1].get("ts", 0) or 0), reverse=True)
                    items = (sticky_items + other_items)[:FACTS_MAX_ITEMS]
                    facts_text = "\\n".join([f"- {k}: {(v.get('value','') or '')[:FACT_MAX_CHARS]}" for k, v in items])

                done_text = ""
                if getattr(self, 'done', None):
                    done_text = "\n".join([f"- {d.get('kind')}: {str(d.get('target'))[:120]}" for d in self.done][-DONE_MAX_ITEMS:])
                input_items.append({
                    'role': 'user',
                    'content': (
                        # Avoid repeating the full Task every step (token saver).
                        ((f'Task (reminder): {goal}\n\n') if (step == 1 or step % 5 == 0) else '')
                        + f'State:\n{self.state_summary or "(none)"}\n\n'
                        + f'Progress:\n{self._progress_block(goal)}\n\n'
                        + f'Completed:\n{done_text or "(none)"}\n\n'
                        + f'Facts:\n{facts_text or "(none)"}\n\n'
                        + f'Memory (recent):\n{mem or "(none)"}\n\n'
                        + f'Observation:\n{prompt}\n\n'
                        + 'Protocol:\n- Use tools only. If done, call finish(success, summary).\n'
                    )
                })

                console.print(Panel(f"[bold]STEP {step}[/bold]\n{obs.get('title','')}\n{obs.get('url','')}", expand=False))

                resp = self.llm.call_main(input_items)

                # Save assistant text (if any) into compact memory so it doesn't "vanish"
                try:
                    txt_raw = (self.llm.output_text(resp) or "").strip()
                except Exception:
                    txt_raw = ""
                if txt_raw:                    
                    self.memory.add(f"Assistant: {txt_raw[:220]}")
                input_items += getattr(resp, "output", [])

                tool_calls = self.llm.get_tool_calls(resp)

                # Если модель решила закончить текстом (без finish) считаем это ошибкой протокола и продолжаем
                if not tool_calls:
                    txt = self.llm.output_text(resp)
                    console.print(Panel(f"[yellow]Model text (no tool call):[/yellow]\n{txt[:800]}", expand=False))
                    self.memory.add(f"Step {step}: model produced text without tool call. Asked it to use finish/tool.")
                    input_items.append({"role": "user", "content": "Use tools only. If done, call finish(success, summary)."})
                    input_items = _trim_openai_items(input_items)
                    continue

                for call in tool_calls:
                    name = call.name
                    args = json.loads(call.arguments or "{}")

                    # Unified action dict for safety check
                    action = {"tool": name, **args}

                    # Guardrail: avoid applying to clearly unrelated vacancies (common "LLM drift" failure).
                    if ROLE_GUARD and name in ("browser_click", "browser_click_force", "browser_click_bbox") and args.get("eid"):
                        elem = _find_elem_by_eid(obs, args.get("eid"))
                        if elem and _looks_like_apply_element_name(elem.get("name") or ""):
                            url_now = str(obs.get("url") or "")
                            title_now = str(obs.get("title") or "")
                            if "/vacancy/" in url_now:
                                role_kws = _derive_role_keywords(getattr(self, "facts", {}) or {})
                                if role_kws and not _vacancy_title_matches(title_now, role_kws):
                                    note = f"Skipped apply: vacancy title '{title_now}' doesn't match target keywords {role_kws}"
                                    self.memory.add(note)
                                    console.print(Panel(f"[yellow]SKIP APPLY[/yellow]\n{note}", expand=False))
                                    input_items.append({
                                        "type": "function_call_output",
                                        "call_id": call.call_id,
                                        "output": json.dumps({"ok": False, "skipped": True, "reason": "vacancy_role_mismatch", "keywords": role_kws}, ensure_ascii=False)
                                    })
                                    continue

                    if name == "finish":
                        console.print(Panel(f"[green]FINISH[/green]\nsuccess={args.get('success')}\n{args.get('summary')}", expand=False))
                        return

                    if name == "ask_user":
                        q = args.get("question", "")
                        # Modes: interactive (ask), auto (best-effort answer), halt (stop execution)
                        if ASK_USER_MODE == "halt":
                            console.print(Panel(f"[cyan]USER INPUT REQUIRED[/cyan]\n{q}\n\n(Stopped. Set ASK_USER_MODE=interactive to answer in CLI.)", expand=False))
                            return
                        if ASK_USER_MODE == "auto":
                            auto = _auto_answer_for_question(q)
                            if auto is None:
                                console.print(Panel(f"[cyan]USER INPUT REQUIRED[/cyan]\n{q}\n\n(Needs human input. Stopped.)", expand=False))
                                return
                            user_answer = auto
                            console.print(Panel(f"[cyan]AUTO-ANSWER[/cyan]\n{q}\n\n→ {user_answer or '(empty)'}", expand=False))
                        else:
                            console.print(Panel(f"[cyan]USER INPUT REQUIRED[/cyan]\n{q}", expand=False))
                            user_answer = (await ainput("> ")).strip()

                        self.memory.add(f"User answered: {user_answer}")
                        input_items.append({"role": "user", "content": f"User answer: {user_answer}"})
                        # tool output back
                        input_items.append({
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": json.dumps({"ok": True}, ensure_ascii=False)
                        })
                        continue

                    if is_sensitive(action, obs):
                        console.print(Panel(f"[red]SENSITIVE ACTION[/red]\n{action}\nConfirm? (y/n)", expand=False))
                        if (await ainput("> ")).strip().lower() != "y":
                            self.memory.add(f"Denied sensitive action: {action}")
                            input_items.append({
                                "type": "function_call_output",
                                "call_id": call.call_id,
                                "output": json.dumps({"denied": True, "reason": "User denied"}, ensure_ascii=False)
                            })
                            continue

                    try:
                        log.info("[green]TOOL[/] %s args=%s", name, args)
                        out = await self._execute_tool(name, args)
                        log.info("[green]TOOL[/] %s ok", name)
                        self.last_action = f"{name}({_compact_args(args)})"

                        self.memory.add(f"Step {step}: {name}({_compact_args(args)}) => {str(out)[:180]}")
                        input_items.append({
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": json.dumps(out, ensure_ascii=False)
                        })
                    except Exception as e:
                        err = {"error": str(e), "tool": name, "args": args}
                        self.memory.add(f"Tool error: {str(err)[:220]}")
                        log.warning("[red]TOOL ERROR[/] %s: %s", name, e)
                        self.last_action = f"{name}(error:{str(e)[:80]})"
                        input_items.append({
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": json.dumps(err, ensure_ascii=False)
                        })

                # Trim OpenAI history to keep prompts small (speed)
                input_items = _trim_openai_items(input_items)

        finally:
            await self.browser.stop()

    async def _execute_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        page = self.browser.page

        async def _page_sig() -> tuple[str, str, str]:
            try:
                url = page.url
            except Exception:
                url = ""
            try:
                title = await page.title()
            except Exception:
                title = ""
            try:
                txt = await page.evaluate("() => document.body ? document.body.innerText : ''")
                txt = re.sub(r"\\s+", " ", str(txt))[:2000]
            except Exception:
                txt = ""
            return (url, title, txt)

        def _alternatives_for_eid(eid: str) -> List[str]:
            obs = getattr(self, "_last_obs", None) or {}
            elems = obs.get("elements", []) or []
            el = next((e for e in elems if e.get("eid") == eid), None)
            el_name = (el or {}).get("name")
            if not el_name:
                return []
            alts = [
                e.get("eid")
                for e in elems
                if e.get("eid") != eid
                and e.get("name") == el_name
                and e.get("role") in ("button", "a")
            ]
            return alts[:6]


        if name == "browser_click":
            eid = args["eid"]
            alternatives = _alternatives_for_eid(eid)

            locator = page.locator(f"[data-webpilot-eid='{eid}']").first

            try:
                if not await locator.is_visible():
                    return {"ok": False, "error": "element not visible", "eid": eid, "alternatives": alternatives}
                if not await locator.is_enabled():
                    return {"ok": False, "error": "element disabled", "eid": eid, "alternatives": alternatives}
            except Exception:
                # если проверки упали — продолжаем, пусть click попробует
                pass

            prev_sig = await _page_sig()
            try:
                await locator.click(timeout=15000)
            except Exception as ex:
                return {
                    "ok": False,
                    "error": f"click failed: {type(ex).__name__}: {ex}",
                    "eid": eid,
                    "alternatives": alternatives,
                }

            await self._post_nav_wait()
            cur_sig = await _page_sig()

            if cur_sig == prev_sig:
                return {
                    "ok": False,
                    "error": "click had no visible effect",
                    "eid": eid,
                    "alternatives": alternatives,
                    "url": page.url,
                }

            return {"ok": True, "url": page.url, "eid": eid, "alternatives": alternatives}


        if name == "browser_type":
            eid = args["eid"]
            text = args["text"]
            clear = bool(args.get("clear", True))

            el = self._get_elem_from_last_obs(eid)
            if el and not _is_typeable_role(el.get("role", "")):
                return {
                    "ok": False,
                    "error": f"eid {eid} is not typeable (role={el.get('role')}, name={el.get('name')})"
                }

            locator = page.locator(f"[data-webpilot-eid='{eid}']").first
            await locator.click(timeout=5000)
            if clear:
                try:
                    await locator.fill("")
                except Exception:
                    try:
                        await page.keyboard.press("Control+A")
                        await page.keyboard.press("Backspace")
                    except Exception:
                        pass
            await page.keyboard.type(text, delay=15)
            return {"ok": True}


        if name == "browser_scroll":
            dy = int(args["dy"])
            await page.mouse.wheel(0, dy)
            return {"ok": True}

        if name == "browser_wait":
            ms = int(args["ms"])
            await asyncio.sleep(ms / 1000)
            return {"ok": True}
        
        if name == "browser_goto":
            url = (args.get("url") or "").strip()
            if not url:
                return {"ok": False, "error": "url is required for browser_goto"}

            # Treat explicit goto as intentional; allow the domain.
            try:
                rd = _root_domain(url)
                if rd:
                    self.allowed_root_domains.add(rd)
            except Exception:
                pass

            await self.browser.page.goto(url, wait_until="domcontentloaded")
            await self._post_nav_wait()
            return {"ok": True, "url": self.browser.page.url}

        if name == "browser_back":
            try:
                await page.go_back()
            except Exception:
                # some pages don't have history
                return {"ok": False, "error": "cannot go back"}
            await self._post_nav_wait()
            return {"ok": True, "url": page.url}
        
        if name == "browser_click_force":
            eid = args["eid"]
            alternatives = _alternatives_for_eid(eid)

            locator = page.locator(f"[data-webpilot-eid='{eid}']").first

            prev_sig = await _page_sig()
            try:
                await locator.click(timeout=15_000, force=True)
            except Exception as ex:
                return {
                    "ok": False,
                    "error": f"force click failed: {type(ex).__name__}: {ex}",
                    "eid": eid,
                    "alternatives": alternatives,
                }

            await self._post_nav_wait()
            cur_sig = await _page_sig()

            if cur_sig == prev_sig:
                return {
                    "ok": False,
                    "error": "force click had no visible effect",
                    "eid": eid,
                    "alternatives": alternatives,
                    "url": page.url,
                }

            return {"ok": True, "url": page.url, "eid": eid, "alternatives": alternatives}


        if name == "browser_click_bbox":
            eid = args.get("eid")
            if not eid:
                return {"ok": False, "error": "eid is required"}

            prev_url = page.url

            obs = await self.observer.observe(self.browser.page, max_elements=OBS_MAX_ELEMS_OBSERVER, max_text_chars=OBS_MAX_TEXT_CHARS_OBSERVER)
            self._last_obs = obs
            el = next((e for e in obs.get("elements", []) if e.get("eid") == eid), None)
            if not el:
                return {"ok": False, "error": f"element {eid} not found"}

            b = el.get("bbox") or [0, 0, 0, 0]
            if isinstance(b, dict):
                x = int(b.get("x", 0) + b.get("w", 0) / 2)
                y = int(b.get("y", 0) + b.get("h", 0) / 2)
            elif isinstance(b, (list, tuple)) and len(b) == 4:
                x = int(b[0] + b[2] / 2)
                y = int(b[1] + b[3] / 2)
            else:
                return {"ok": False, "error": "bad bbox format"}

            await self.browser.page.mouse.click(x, y)
            await self._post_nav_wait()

            cur_url = page.url

            return {"ok": True, "eid": eid, "x": x, "y": y, "url": cur_url}

        if name == "browser_scroll_to":
            where = args["where"]
            if where == "top":
                await page.evaluate("() => window.scrollTo(0, 0)")
            else:
                await page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
            return {"ok": True}

        if name == "browser_find":
            query = args["query"]
            limit = int(args.get("limit", 8))
            return {"ok": True, "matches": self._find_elements(query, limit=limit)}

        if name == "memory_save":
            key = str(args.get("key", "")).strip()[:60]
            value = str(args.get("value", "")).strip()[:800]
            why = str(args.get("why", "")).strip()[:200]

            if not key or not value:
                return {"ok": False, "error": "key/value required"}

            # Save via _auto_save_fact so the entry has ts/sticky and stays visible in "Long-term facts"
            self._auto_save_fact(key, value, why=why or "memory_save", source_url=str(page.url if page else ""))
            return {"ok": True}


        if name == "task_mark_done":
            if not hasattr(self, "done"):
                self.done = []
            kind = str(args.get("kind", "")).strip()[:40]
            target = str(args.get("target", "")).strip()[:300]
            note = str(args.get("note", "")).strip()[:300]
            self.done.append({"kind": kind, "target": target, "note": note})
            # дедуп по (kind,target)
            uniq = {(d["kind"], d["target"]): d for d in self.done}
            self.done = list(uniq.values())
            return {"ok": True, "count": len(self.done)}

        if name == "delegate":
            agent = args["agent"]
            task = args["task"]
            result = self.llm.call_subagent(agent, task, await self.observer.observe(page, max_elements=OBS_MAX_ELEMS_OBSERVER, max_text_chars=OBS_MAX_TEXT_CHARS_OBSERVER), self.memory.render())
            return {"ok": True, "agent": agent, "result": result}

        raise ValueError(f"Unknown tool: {name}")    

    async def _run_anthropic(self, goal: str) -> None:        
        def _normalize_tool_call(call: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
            # 1) args могут лежать в input / args / arguments
            args_raw = call.get("input", None)
            if args_raw is None:
                args_raw = call.get("args", None)
            if args_raw is None:
                args_raw = call.get("arguments", None)
            if args_raw is None:
                args_raw = {}

            # args иногда прилетают строкой
            if isinstance(args_raw, str):
                try:
                    args_raw = json.loads(args_raw)
                except Exception:
                    args_raw = {}

            if not isinstance(args_raw, dict):
                args_raw = {}

            # 2) name иногда прилетает как: "browser_click({'eid': 'E2'})</arg_value>"
            name_raw = str(call.get("name", "") or "")
            cleaned = re.sub(r"<[^>]+>", "", name_raw).strip()

            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(?:\((.*)\))?\s*$", cleaned)
            if not m:
                return cleaned, args_raw

            name = m.group(1)
            inside = (m.group(2) or "").strip()

            # 3) если input пустой — пытаемся распарсить args из "(...)" как dict
            if (not args_raw) and inside:
                try:
                    parsed = ast.literal_eval(inside)  # понимает {'eid': 'E2'}
                    if isinstance(parsed, dict):
                        args_raw = parsed
                except Exception:
                    try:
                        parsed = json.loads(inside)
                        if isinstance(parsed, dict):
                            args_raw = parsed
                    except Exception:
                        pass

            return name, args_raw



        messages = []

        MAX_HISTORY = int(os.getenv("MAX_HISTORY", "20"))

        for step in range(1, self.max_steps + 1):
            await self.browser.ensure_active_page()

            obs = await self.observer.observe(self.browser.page, max_elements=OBS_MAX_ELEMS_OBSERVER, max_text_chars=OBS_MAX_TEXT_CHARS_OBSERVER)
            self._last_obs = obs
            obs_small = _compact_obs(obs)
            self._auto_track_from_obs(goal, obs)

            # --- anti-stuck (IMPORTANT for zai/cometapi/anthropic) ---
            if not hasattr(self, "_same_page"):
                self._same_page = 0
                self._last_sig = None

            sig = (
                obs_small.get("url"),
                obs_small.get("title"),
                (obs_small.get("visible_text") or "")[:120],  # помогает отличать модалки на той же странице
            )

            if sig == self._last_sig:
                self._same_page += 1
            else:
                self._same_page = 0
                self._last_sig = sig

            nav_hint = None
            if self._same_page >= 3:
                t_nav = time.time()
                nav_hint = await asyncio.to_thread(
                    self.llm.call_subagent, "navigator", goal, obs_small, self.memory.render()
                )
                self.memory.add(f"Navigator advice: {nav_hint}")
                self._same_page = 0

                self._trace_llm(
                    step=step,
                    kind="navigator",
                    messages=[{
                        "role": "user",
                        "content": [{"type": "text", "text": f"Task: {goal}\nObs: {json.dumps(obs_small, ensure_ascii=False)}"}],
                    }],
                    obs=obs,
                    obs_small=obs_small,
                    latency_s=(time.time() - t_nav),
                    resp={"content": [{"type": "text", "text": json.dumps(nav_hint, ensure_ascii=False)}]},
                    tool_calls=[],
                    text=json.dumps(nav_hint, ensure_ascii=False),
                )


            # keep a small state summary for continuity
            self.state_summary = f"On page: {(obs_small.get('title') or '—')[:80]} | {obs_small.get('url') or ''} | last_action: {self.last_action or '—'}"

            # Auto-extract reusable facts occasionally
            await self._maybe_auto_extract(goal, obs, obs_small, step)

            log.info(
                "[cyan]STEP %d[/] %s | %s",
                step,
                (obs_small.get("title") or "—")[:60],
                (obs_small.get("url") or "")[:80],
            )

            mem = (self.memory.render() or "")
            if MEMORY_PROMPT_CHARS and len(mem) > MEMORY_PROMPT_CHARS:
                mem = mem[-MEMORY_PROMPT_CHARS:]

            prompt = obs_to_prompt_min(obs_small)

            facts_text = ""
            if getattr(self, "facts", None):
                max_items = int(os.getenv("FACTS_MAX_ITEMS", "4"))
                max_chars = int(os.getenv("FACT_MAX_CHARS", "120"))

                items_all = [(k, v) for k, v in (self.facts or {}).items() if isinstance(v, dict)]
                sticky_items = []
                for k, v in items_all:
                    if v.get("sticky") or (k in _STICKY_KEYS_SET) or k.startswith("resume_") or k.startswith("target_") or k == "role":
                        sticky_items.append((k, v))
                sticky_items.sort(key=lambda kv: float(kv[1].get("ts", 0) or kv[1].get("updated_at", 0) or 0), reverse=True)
                sticky_keys = {k for k, _ in sticky_items}

                other_items = [(k, v) for k, v in items_all if k not in sticky_keys]
                other_items.sort(key=lambda kv: float(kv[1].get("ts", 0) or kv[1].get("updated_at", 0) or 0), reverse=True)

                items = (sticky_items + other_items)[:max_items]

                def _clip(s: str) -> str:
                    s = str(s)
                    return s if len(s) <= max_chars else s[:max_chars] + "…"

                facts_text = "\n".join([f"- {k}: {_clip(v.get('value', ''))}" for k, v in items])

            done_text = ""
            if getattr(self, "done", None):
                done_text = "\n".join([f"- {d['kind']}: {d['target']}" for d in self.done][-DONE_MAX_ITEMS:])

            task_block = f"Task: {goal}\n\n" if (step == 1 or step % 5 == 0) else ""

            user_text = (
                task_block
                + f"State:\n{self.state_summary}\n\n"
                f"Progress:\n{self._progress_block(goal)}\n\n"
                f"Completed milestones:\n{done_text or '(none)'}\n\n"
                f"Long-term facts (saved by you):\n{facts_text or '(none)'}\n\n"
                f"Memory (recent):\n{mem or '(none)'}\n\n"
                f"Observation (compact):\n{prompt}"
                + (f"\n\nNavigator advice (JSON): {json.dumps(nav_hint, ensure_ascii=False)}" if nav_hint else "")
                
            )

            messages.append({"role": "user", "content": [{"type": "text", "text": user_text}]})

            # Обрезаем историю, чтобы не росла бесконечно
            if len(messages) > MAX_HISTORY:
                # оставим самый первый user (как контекст) + хвост
                messages = messages[:1] + messages[-(MAX_HISTORY - 1):]

            llm_timeout = int(os.getenv("LLM_TIMEOUT_S", "40"))
            t0 = time.time()

            try:
                req_messages = list(messages)
                resp = await asyncio.wait_for(
                    asyncio.to_thread(self.llm.call_main, messages),
                    timeout=llm_timeout,
                )
            except asyncio.TimeoutError:
                log.warning("[red]LLM timeout[/] waited %ss. Retrying…", llm_timeout)
                _jsonl_append(self.llm_trace_path, {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                    "run_id": self.run_id,
                    "step": step,
                    "kind": "llm_timeout",
                    "timeout_s": llm_timeout,
                    "page": {"url": obs.get("url"), "title": obs.get("title")},
                    "obs_stats": {
                        "visible_text_collected_chars": len((obs or {}).get("visible_text") or ""),
                        "elements_collected": len((obs or {}).get("elements") or []),
                        "visible_text_sent_chars": len((obs_small or {}).get("visible_text") or ""),
                        "elements_sent": len((obs_small or {}).get("elements") or []),
                    },
                    "last_user_chars": len(user_text or ""),
                })
                await asyncio.sleep(2)
                continue
            except Exception as e:
                log.warning("[red]LLM error[/] %s. Retrying…", e)
                await asyncio.sleep(2)
                continue
            finally:
                log.info("[yellow]LLM[/] done in %.1fs", time.time() - t0)

            # Короткий лог: сколько tools и кусок текста
            try:
                tool_calls = self.llm.get_tool_calls(resp)
            except Exception:
                tool_calls = []

            try:
                txt = (self.llm.output_text(resp) or "").strip().replace("\n", " ")
            except Exception:
                txt = ""
            if len(txt) > 160:
                txt = txt[:160] + "…"
            log.info("[yellow]LLM[/] tools=%d text=%s", len(tool_calls), txt if txt else "—")
            self._trace_llm(
                step=step,
                kind="main",
                messages=req_messages,
                obs=obs,
                obs_small=obs_small,
                latency_s=(time.time() - t0),
                resp=resp,
                tool_calls=tool_calls,
                text=self.llm.output_text(resp) if hasattr(self.llm, "output_text") else txt,
            )
            # сохраняем ответ ассистента в историю
            assistant_content = resp["content"] if isinstance(resp, dict) else resp.content
            messages.append({"role": "assistant", "content": assistant_content})

            # Persist assistant text into compact memory for continuity
            if txt:
                self.memory.add(f"Assistant: {txt[:220]}")

            if not tool_calls:
                # если нет tool calls — заставим модель использовать tools
                self.memory.add(f"Step {step}: model text without tool call: {txt[:400]}")
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": "Use tools only. If done, call finish(success, summary)."}],
                })
                continue

            tool_results_blocks = []

            for call in tool_calls:
                name, args = _normalize_tool_call(call)

                log.info("[green]TOOL[/] %s %s", name, _compact_args(args))

                # finish
                if name == "finish":
                    summary = args.get("summary", "")
                    success = bool(args.get("success", True))
                    print(f"\n[FINISH] success={success}\n{summary}\n")
                    return

                # ask_user
                if name == "ask_user":
                    q = args.get("question", "")
                    print(f"\n[USER INPUT REQUIRED]\n{q}\n")
                    user_answer = (await ainput("> ")).strip()
                    self.memory.add(f"User answered: {user_answer}")

                    tool_results_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": call["id"],
                        "content": json.dumps({"answer": user_answer}, ensure_ascii=False),
                    })
                    continue

                # Security layer
                action = {"tool": name, **args}
                if is_sensitive(action, obs_small):
                    print(f"\n[SENSITIVE ACTION]\n{action}\nConfirm? (y/n)")
                    if (await ainput("> ")).strip().lower() != "y":
                        self.memory.add(f"Denied sensitive action: {action}")
                        tool_results_blocks.append({
                            "type": "tool_result",
                            "tool_use_id": call["id"],
                            "content": json.dumps({"denied": True, "reason": "User denied"}, ensure_ascii=False),
                            "is_error": False,
                        })
                        continue

                # delegate -> sub-agent call
                if name == "delegate":
                    agent = args.get("agent")
                    task = args.get("task", "")
                    # В subagent тоже отправляем compact obs
                    result = self.llm.call_subagent(agent, task, obs_small, self.memory.render())
                    self.memory.add(f"delegate({agent}): {result}")
                    tool_results_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": call["id"],
                        "content": json.dumps({"ok": True, "result": result}, ensure_ascii=False),
                    })
                    continue

                # обычные browser tools
                try:
                    out = await self._execute_tool(name, args)
                    self.memory.add(f"Step {step}: {name}({_compact_args(args)}) => {str(out)[:180]}")
                    tool_results_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": call["id"],
                        "content": json.dumps(out, ensure_ascii=False),
                    })
                except Exception as e:
                    err = {"error": str(e), "tool": name, "args": args}
                    self.memory.add(f"Tool error: {str(err)[:220]}")
                    tool_results_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": call["id"],
                        "content": json.dumps(err, ensure_ascii=False),
                        "is_error": True,
                    })

            # отправляем результаты tool’ов обратно модели
            messages.append({"role": "user", "content": tool_results_blocks})

            # снова обрезаем историю после tool_results (важно!)
            if len(messages) > MAX_HISTORY:
                messages = messages[:1] + messages[-(MAX_HISTORY - 1):]

        print("\n[STOP] Max steps reached.\n")
