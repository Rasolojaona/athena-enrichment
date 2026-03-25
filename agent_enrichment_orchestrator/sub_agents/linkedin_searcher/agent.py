"""
LinkedIn Searcher Agent - Specialized agent for LinkedIn profile enrichment.

This agent is an expert at finding and enriching LinkedIn profiles for event visitors.
It uses the search_linkedin_profile tool to find professional information.
"""

import os
from google.adk.agents import LlmAgent
from .tools import get_linkedin_search_tool
from ...tools.dev_callbacks import get_dev_callbacks
import logging

logger = logging.getLogger(__name__)

# Load the instruction prompt from file
prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "linkedin_prompt.txt")

with open(prompt_path, "r", encoding="utf-8") as f:
    instruction = f.read()

def make_linkedin_agent() -> LlmAgent:
    _dev = get_dev_callbacks()
    return LlmAgent(
        name="linkedin_searcher",
        model="gemini-2.5-flash",
        instruction=instruction,
        tools=[get_linkedin_search_tool()],
        description="Expert agent for finding and enriching LinkedIn profiles",
        before_tool_callback=_dev.before_tool if _dev else None,
        after_tool_callback=_dev.after_tool  if _dev else None,
    )

# Instance singleton (pour root_agent webhook)
linkedin_agent = make_linkedin_agent()

logger.info("✅ LinkedIn Searcher Agent initialized")