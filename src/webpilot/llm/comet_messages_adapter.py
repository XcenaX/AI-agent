import json
import os
import re
from typing import Any, Dict, List, Optional

import httpx

import time
import random

from .tools import TOOLS

SYSTEM_MAIN = """
You are an autonomous web agent operating a real browser via tools.

Output MUST be tool calls only (no narration). If the task is complete, call finish(success, summary).

PRIMARY GOAL
- Complete the user's task exactly as stated in Task. Do not assume extra goals.
- Base decisions on (1) Task, (2) current page content, (3) Completed milestones + Long-term facts.

CHECKLIST & PROGRESS (MANDATORY)
- Create an internal checklist of milestones derived from Task (typically 3–10 items).
- Track progress ONLY via Completed milestones and Long-term facts shown to you.
- After finishing ANY milestone or meaningful sub-step, you MUST call task_mark_done(kind, target, note) immediately.
  - Use stable kinds: visited, opened, read, extracted, shortlisted, submitted, sent, purchased, downloaded, uploaded, applied, filled_form, etc.
  - Use a stable target (URL or unique identifier).
- Before repeating a major action, ALWAYS check Completed milestones to avoid duplicates.

MEMORY-FIRST (MANDATORY)
- Treat Long-term facts as your durable working memory across steps.
- When you learn stable info that will be reused later, you MUST call memory_save(key, value, why) immediately.
  Examples of stable info:
  - user profile summary, preferences, constraints, acceptance criteria
  - chosen filters/search queries, selected options, final decisions
  - concise summaries of important documents/pages you must rely on later (e.g., a profile/resume/spec)
  - items already processed and why (e.g., shortlisted items, rejected reasons)
  - state needed for continuity (e.g., logged_in=yes, verification_pending, current step)
- On pages that contain key task-specific information, aim to save 1–3 concise facts/summaries.
- If you have taken several steps without saving anything and the task is multi-step, reassess and save the most important stable info you have learned so far.
- Keep values compact (<= 400 chars). Summarize; never store full page text.
- Use clear keys with namespaces when helpful: profile.*, constraints.*, filters.*, doc.*, item.*, state.*, progress.*.
- Never store secrets (passwords, API keys, 2FA codes) or unnecessary PII.

ITEM SELECTION & DEDUP
- When you select an item to act on later (e.g., a listing/document/message/order), record it:
  task_mark_done("candidate", "<item_url_or_id>", "why it fits")
  Optionally store a compact snippet in memory_save("item.<id>.summary", "...", "to act later").
- After completing an irreversible action that the Task authorizes (submit/send/purchase/apply), ALWAYS record it:
  task_mark_done("submitted|sent|purchased|applied", "<item_url_or_id>", "done")

EFFICIENCY
- Prefer 1–3 high-signal actions per step.
- Prefer browser_find + targeted clicks over random scrolling/clicking.
- Avoid re-opening pages you already processed unless necessary; if you must, note why.

SAFETY & USER INPUT
- Ask the user ONLY for: login/2FA/captcha, missing critical info required to proceed, or confirmation before an irreversible action IF the Task does not clearly authorize it.
- If the Task explicitly requests an irreversible action, proceed without asking again.

NAVIGATION
- Stay on sites relevant to Task. Do not navigate to unrelated external sites unless explicitly required by Task.
- If you end up on an unrelated page, go back immediately.

ROBUSTNESS / RECOVERY
- If a click/type fails or the page doesn't change: observe again, wait briefly, scroll a bit, retry with browser_click_force or browser_click_bbox.
- Use browser_find to re-locate elements after navigation/updates.
- Do not give up due to “dynamic JS”; use waits, retries, find, force/bbox click.

DATA HYGIENE
- Never request or store passwords. Never store full page text.
"""

SYSTEM_NAVIGATOR = """
You are a specialist sub-agent: NAVIGATOR.
Given Task + Observation + (Completed milestones + Long-term facts), propose the next 3–6 concrete actions the main agent can try.

Rules:
- No tool calls executed here; you only propose them.
- Output strict JSON only. No extra text.
- Prefer browser_find first when the correct eid is uncertain.
- If proposing browser_click/browser_type, you MUST use only eids present in Observation.
- Avoid repeating actions that are already in Completed milestones unless you explain why it's necessary.
- Do NOT propose navigating to unrelated external sites.

JSON schema:
{
  "hypothesis": "short",
  "next_actions": [
    {"tool": "browser_find|browser_click|browser_click_force|browser_click_bbox|browser_type|browser_scroll|browser_scroll_to|browser_wait|browser_goto|browser_back",
     "args": {...},
     "why": "short"}
  ]
}
"""

SYSTEM_EXTRACTOR = """
You are a specialist sub-agent: EXTRACTOR.
Given Task + Observation text, extract only the most relevant facts.

Rules:
- No tool calls.
- Output strict JSON only.
- Prefer stable, reusable facts (requirements, salary, location, skills) over long descriptions.
- Keep each value short (<= 200 chars). No long quotes.
- Do NOT include passwords or secrets. Avoid PII unless explicitly needed by Task.

JSON schema:
{
  "facts": [{"key":"...", "value":"..."}, ...],
  "missing_info": "what is needed from user (if anything), short"
}
"""

def _to_anthropic_tools() -> List[Dict[str, Any]]:
    # OUR: {"type":"function","name":..,"description":..,"parameters":{...}}
    # ANTHROPIC: {"name":..,"description":..,"input_schema":{...}}
    out = []
    for t in TOOLS:
        out.append({
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
        })
    return out


class CometMessagesAdapter:
    """
    CometAPI Anthropic Messages endpoint:
    POST https://api.cometapi.com/v1/messages
    (Claude only)
    """
    def __init__(self, model: str, api_key: str, base_url: str = "https://api.cometapi.com/v1"):
        if not api_key:
            raise ValueError("API key is missing")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.tools = _to_anthropic_tools()
        timeout = httpx.Timeout(connect=8.0, read=120.0, write=30.0, pool=30.0)
        limits = httpx.Limits(max_connections=10, max_keepalive_connections=5, keepalive_expiry=30.0)
        self._client = httpx.Client(timeout=timeout, limits=limits, http2=True)

    def _post_json(self, url: str, payload: dict, headers: dict) -> dict:
        last_exc = None
        for attempt in range(4):
            try:
                r = self._client.post(url, headers=headers, json=payload)
                if r.status_code >= 400:
                    raise RuntimeError(f"LLM HTTP {r.status_code}: {r.text}")
                return r.json()
            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError, httpx.TransportError) as e:
                last_exc = e
                # backoff: 1s,2s,4s,8s + небольшой джиттер
                time.sleep(min(2 ** attempt, 8) + random.random() * 0.2)
        raise RuntimeError(f"LLM request timed out after retries: {last_exc}")

    def _messages_url(self) -> str:
        base = self.base_url.rstrip("/")
        # если уже указали .../messages — не добавляем второй раз
        if base.endswith("/messages"):
            return base
        return f"{base}/messages"

    def _headers(self) -> Dict[str, str]:
        """Build headers for Anthropic-compatible endpoints (CometAPI / ZAI)."""
        h: Dict[str, str] = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            # Many Anthropic-compatible gateways require this version header.
            "anthropic-version": os.getenv("ANTHROPIC_VERSION", "2023-06-01"),
        }
        # Some gateways (e.g., CometAPI) accept/require x-api-key; keep it on by default.
        if os.getenv("SEND_X_API_KEY", "1") != "0":
            h["X-Api-Key"] = self.api_key
            h["x-api-key"] = self.api_key
        return h

    def call_main(self, messages: List[Dict[str, Any]]) -> Any:
        url = self._messages_url()
        payload = {
            "model": self.model,
            "max_tokens": int(os.getenv("MAIN_MAX_TOKENS", "512")),
            "system": SYSTEM_MAIN,
            "tools": self.tools,
            "messages": messages,
        }
        headers=self._headers()
        rj = self._post_json(
            url,
            payload,
            headers=headers
        )
        return rj

    def get_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        calls = []
        for block in (response.get("content") or []):
            if block.get("type") == "tool_use":
                calls.append({"id": block.get("id"), "name": block.get("name"), "input": block.get("input") or {}})
        return calls

    def output_text(self, response: Any) -> str:
        parts = []
        for block in (response.get("content") or []):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts).strip()

    def call_subagent(self, agent: str, task: str, observation: Dict[str, Any], memory: str) -> Dict[str, Any]:
        system = SYSTEM_NAVIGATOR if agent == "navigator" else SYSTEM_EXTRACTOR if agent == "extractor" else None
        if system is None:
            raise ValueError(f"Unknown sub-agent: {agent}")

        max_obs_chars = int(os.getenv("SUBAGENT_OBS_TEXT_CHARS", "4200"))

        slim_obs = {
            "url": observation.get("url"),
            "title": observation.get("title"),
            "visible_text": (observation.get("visible_text") or "")[:max_obs_chars],
            "elements": (observation.get("elements") or [])[:60],
        }

        url = self._messages_url()
        payload = {
            "model": self.model,
            "max_tokens": int(os.getenv("SUBAGENT_MAX_TOKENS", "700")),
            "system": system,
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": json.dumps({
                    "task": task,
                    "memory": memory,
                    "observation": slim_obs
                }, ensure_ascii=False)}]
            }]
        }

        data = self._post_json(url, payload, headers=self._headers())
        txt = (self.output_text(data) or "").strip()

        def _loads_best_effort(s: str) -> Optional[dict]:
            s = (s or "").strip()
            if not s:
                return None
            # 1) direct JSON
            try:
                return json.loads(s)
            except Exception:
                pass

            # 2) JSON inside fenced block
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.S | re.I)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    pass

            # 3) first {...} span
            i = s.find("{")
            j = s.rfind("}")
            if i != -1 and j != -1 and j > i:
                chunk = s[i : j + 1]
                try:
                    return json.loads(chunk)
                except Exception:
                    pass
            return None

        parsed = _loads_best_effort(txt)
        if parsed is None:
            return {"raw": txt}
        return parsed

