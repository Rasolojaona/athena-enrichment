"""
Web Searcher Agent - Searches for company info from their website.

This agent specializes in extracting public information from company websites
to complement LinkedIn data.
"""

import os
from google.adk.agents import LlmAgent
from .tools import get_web_scraper_tool, get_person_web_search_tool
from ...tools.dev_callbacks import get_dev_callbacks
import logging

logger = logging.getLogger(__name__)

# Load the instruction prompt from file
prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "web_searcher_prompt.txt")

with open(prompt_path, "r", encoding="utf-8") as f:
    instruction = f.read()

def make_web_searcher_agent() -> LlmAgent:
    _dev = get_dev_callbacks()
    return LlmAgent(
        name="web_searcher",
        model="gemini-2.5-flash",
        instruction=instruction,
        tools=[get_web_scraper_tool(), get_person_web_search_tool()],
        description="Expert agent for extracting company website data and individual web presence",
        before_tool_callback=_dev.before_tool if _dev else None,
        after_tool_callback=_dev.after_tool  if _dev else None,
    )

# Instance singleton (pour root_agent webhook)
web_searcher_agent = make_web_searcher_agent()

logger.info("✅ Web Searcher Agent initialized")
