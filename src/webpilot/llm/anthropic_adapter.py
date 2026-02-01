import json
from typing import Any, Dict, List

from anthropic import Anthropic
from .tools import TOOLS

SYSTEM_MAIN = """
You are an autonomous web agent operating a real browser via tools.

Output MUST be tool calls only (no narration). If the task is complete, call finish(success, summary).

PRIMARY GOAL
- Complete the user's task exactly as stated in Task. Do not assume extra goals.
- Derive search keywords/filters ONLY from Task and on-page context. Do NOT hardcode topic keywords.

EFFICIENCY
- Prefer 1–3 high-signal actions per step.
- Prefer browser_find + targeted clicks over random scrolling/clicking.
- Avoid re-opening pages you already processed unless necessary.

SAFETY & USER INPUT
- Ask the user ONLY for: login/2FA/captcha, missing critical info, or approval right before irreversible actions.
- Irreversible actions include: submit/apply/send/purchase/delete/confirm. If user already explicitly authorized in Task, still ask_user right before the final irreversible click if outcome cannot be undone.

NAVIGATION
- Stay on the target site relevant to Task. Do not navigate to unrelated external sites (reviews, ads, third-party pages) unless explicitly required by Task.
- If you end up on an unrelated page, go back immediately.

STATE & MEMORY
- Use memory_save(key,value,why) for stable facts needed later (keep value compact, no long text dumps).
  Examples: extracted_resume_summary, user_preferences, login_state, chosen_resume_name.
- Use task_mark_done(kind,target,note) after completing a milestone to prevent repeats.
  Examples: applied:<vacancy_url>, extracted_resume:<resume_url>, saved_fact:<key>.
- Before doing something that seems redundant, check Completed milestones / Long-term facts and avoid repeating work.

ROBUSTNESS / RECOVERY
- If a click/type fails or page doesn't change: observe again, wait briefly, scroll a bit, press Escape (if available via your tools), retry with browser_click_force or browser_click_bbox.
- Do not claim you cannot act because the site is “dynamic JS”. Use waits, retries, find, force/bbox click.

DATA HYGIENE
- Never request or store passwords. Never store full page text. Keep PII out of memory unless strictly needed to fill a form.
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
    out = []
    for t in TOOLS:
        # our format: {"type":"function","name":..,"description":..,"parameters":{...}}
        out.append({
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
        })
    return out


class AnthropicAdapter:
    def __init__(self, model: str):
        # Anthropic SDK читает ANTHROPIC_API_KEY из env автоматически
        self.client = Anthropic()
        self.model = model
        self.tools = _to_anthropic_tools()

    def call_main(self, messages: List[Dict[str, Any]]) -> Any:
        return self.client.messages.create(
            model=self.model,
            max_tokens=800,
            system=SYSTEM_MAIN,
            tools=self.tools,
            messages=messages,
        )

    def get_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        calls = []
        for block in response.content:
            if block.type == "tool_use":
                calls.append({"id": block.id, "name": block.name, "input": block.input})
        return calls

    def output_text(self, response: Any) -> str:
        parts = []
        for block in response.content:
            if block.type == "text":
                parts.append(block.text)
        return "\n".join(parts).strip()

    def call_subagent(self, agent: str, task: str, observation: Dict[str, Any], memory: str) -> Dict[str, Any]:
        system = SYSTEM_NAVIGATOR if agent == "navigator" else SYSTEM_EXTRACTOR if agent == "extractor" else None
        if system is None:
            raise ValueError(f"Unknown sub-agent: {agent}")

        payload = {
            "task": task,
            "memory": memory,
            "observation": {
                "url": observation.get("url"),
                "title": observation.get("title"),
                "visible_text": (observation.get("visible_text") or "")[:1200],
                "elements": (observation.get("elements") or [])[:60],
            }
        }

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=600,
            system=system,
            messages=[{
                "role": "user",
                "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
            }]
        )
        txt = ""
        for b in resp.content:
            if b.type == "text":
                txt += b.text

        txt = txt.strip()
        try:
            return json.loads(txt)
        except Exception:
            return {"raw": txt}