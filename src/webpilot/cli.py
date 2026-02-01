import os
import asyncio
from dotenv import load_dotenv

from .agent import WebPilotAgent
from .logging_utils import setup_logging

def main():
    load_dotenv()
    setup_logging()

    goal = input("Введите задачу агенту: ").strip()

    provider = os.getenv("PROVIDER", "openai").strip().lower()
    user_data_dir = os.getenv("USER_DATA_DIR", "./profile")
    max_steps = int(os.getenv("MAX_STEPS", "80"))

    if provider == "cometapi":
        model = os.getenv("COMETAPI_MODEL", "claude-sonnet-4-5-20250929")
    elif provider == "zai":
        model = os.getenv("ZAI_MODEL", "glm-4.6")
    elif provider == "anthropic":
        model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
    else:
        model = os.getenv("OPENAI_MODEL", "gpt-5")

    agent = WebPilotAgent(
        provider=provider,
        model=model,
        user_data_dir=user_data_dir,
        max_steps=max_steps,
    )

    asyncio.run(agent.run(goal))


if __name__ == "__main__":
    main()