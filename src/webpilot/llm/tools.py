# Единый набор tools (провайдер-агностично по смыслу)
TOOLS = [
    {
        "type": "function",
        "name": "browser_click",
        "description": "Click an element on the current page by its eid from the observation.",
        "parameters": {
            "type": "object",
            "properties": {"eid": {"type": "string"}},
            "required": ["eid"],
        },
    },
    {
        "type": "function",
        "name": "browser_type",
        "description": "Type text into an element (input/textarea) by eid from the observation.",
        "parameters": {
            "type": "object",
            "properties": {
                "eid": {"type": "string"},
                "text": {"type": "string"},
                "clear": {"type": "boolean"},
            },
            "required": ["eid", "text"],
        },
    },
    {
        "type": "function",
        "name": "browser_scroll",
        "description": "Scroll the page vertically by dy pixels (positive = down, negative = up).",
        "parameters": {
            "type": "object",
            "properties": {"dy": {"type": "integer"}},
            "required": ["dy"],
        },
    },
    {
        "type": "function",
        "name": "browser_wait",
        "description": "Wait for ms milliseconds (useful for dynamic pages).",
        "parameters": {
            "type": "object",
            "properties": {"ms": {"type": "integer"}},
            "required": ["ms"],
        },
    },
    {
        "type": "function",
        "name": "ask_user",
        "description": "Ask the user for extra information (login, captcha, missing data).",
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    },
    {
        "type": "function",
        "name": "delegate",
        "description": "Call a specialized sub-agent (navigator/extractor) for a short result.",
        "parameters": {
            "type": "object",
            "properties": {
                "agent": {"type": "string", "enum": ["navigator", "extractor"]},
                "task": {"type": "string"},
            },
            "required": ["agent", "task"],
        },
    },
    {
        "type": "function",
        "name": "finish",
        "description": "Finish the task with success status and a short summary.",
        "parameters": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "summary": {"type": "string"},
            },
            "required": ["success", "summary"],
        },
    },
    {
        "type": "function",
        "name": "browser_goto",
        "description": "Navigate to a URL.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"]
        }
    },
    {
        "type": "function",
        "name": "browser_back",
        "description": "Go back in browser history.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "type": "function",
        "name": "browser_click_force",
        "description": "Click element by eid using force=True (useful when overlays intercept clicks).",
        "parameters": {
            "type": "object",
            "properties": {"eid": {"type": "string"}},
            "required": ["eid"]
        }
    },
    {
        "type": "function",
        "name": "browser_click_bbox",
        "description": "Click at the center of the element bounding box by eid (fallback when normal clicks fail).",
        "parameters": {
            "type": "object",
            "properties": {"eid": {"type": "string"}},
            "required": ["eid"]
        }
    },
    {
        "type": "function",
        "name": "browser_scroll_to",
        "description": "Scroll to top or bottom of the page.",
        "parameters": {
            "type": "object",
            "properties": {"where": {"type": "string", "enum": ["top", "bottom"]}},
            "required": ["where"]
        }
    },
    {
        "type": "function",
        "name": "browser_find",
        "description": "Find best-matching elements from the last snapshot by text query. Returns list of candidates with eid/name/role.",
        "parameters": {
            "type": "object",
            "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 8}
            },
            "required": ["query"]
        }
    },
    {
        "type": "function",
        "name": "memory_save",
        "description": "Save an important fact or constraint for later. Use for stable info needed across many steps. Keep it short.",
        "parameters": {
            "type": "object",
            "properties": {
            "key": {"type": "string", "description": "Short key, e.g. 'user_profile', 'requirements', 'login_state'"},
            "value": {"type": "string", "description": "Compact value, <= 400 chars"},
            "why": {"type": "string", "description": "Why this matters (<= 120 chars)"}
            },
            "required": ["key", "value"]
        }
    },
    {
        "type": "function",
        "name": "task_mark_done",
        "description": "Record a completed milestone to avoid repeating work.",
        "parameters": {
            "type": "object",
            "properties": {
            "kind": {"type": "string", "description": "e.g. applied, purchased, emailed, downloaded, filled_form"},
            "target": {"type": "string", "description": "URL or short identifier"},
            "note": {"type": "string", "description": "Short note <= 200 chars"}
            },
            "required": ["kind", "target"]
        }
    }
]