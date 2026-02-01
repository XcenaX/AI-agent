## WebPilot Agent (MVP)

- Visible Playwright browser (headed)
- Persistent sessions via USER_DATA_DIR (login manually, agent continues)
- Autonomous tool-calling loop
- Security confirmation for sensitive actions

### Run (Windows PowerShell)
1) Install deps + browsers:
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   python -m playwright install chromium

2) Set API key (OpenAI):
   setx OPENAI_API_KEY "..."

3) Run:
   ./run.ps1