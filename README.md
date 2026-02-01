# AI-agent

Simple browser AI agent (MVP) based on a ZAI model

> **What it is:** a minimal “web pilot” that opens a **visible** Playwright Chromium browser, can keep **persistent sessions** (so you can log in once manually), and runs an **autonomous tool-calling loop** with **security confirmation** for sensitive actions
---

## Features

- **Visible Playwright browser (headed)**   
- **Persistent sessions** via `USER_DATA_DIR` (log in manually, the agent continues)   
- **Autonomous tool-calling loop**   
- **Security confirmation** for sensitive actions   

---

## Repository layout

- `src/webpilot/` — core agent implementation   
- `mcp_entry.py` — entrypoint for MCP-style integration/usage (if applicable)   
- `run.ps1` — Windows runner script   
- `requirements.txt` — Python dependencies   

---

## Quickstart (Windows PowerShell)

1) Create venv + install deps + install Playwright Chromium:   
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m playwright install chromium
```

2) Set API keys (look at .env.example):   

3) Run:   
```powershell
.\run.ps1
```

## Usage

The agent runs a loop where it:
1) reads the current page/state
2) decides which tool/action to execute (e.g., click, type, navigate)
3) optionally asks for confirmation for sensitive steps
4) repeats until the task is finished. 

**Example tasks**
- “Find the official documentation for X and summarize key steps.”
- “Log into a website (I will do the login), then navigate to billing page and download an invoice.”
- “Search for a product and compare top 3 options.”

---

## Safety notes

Some actions may require explicit confirmation especially anything that looks like it can:
- submit forms
- purchase/pay
- change account settings
- delete data   

---

## Disclaimer

This is an MVP browser automation agent. Use carefully on real accounts and production systems, and keep secrets out of logs. 
