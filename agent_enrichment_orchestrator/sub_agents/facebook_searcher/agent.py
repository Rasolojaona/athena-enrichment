"""
Facebook Searcher Agent - Searches for Facebook profiles.

Agent spécialisé dans la recherche de profils Facebook publics
utilisant Apify.
"""

import os
from google.adk.agents import LlmAgent
from .tools import get_facebook_search_tool
from ...tools.dev_callbacks import get_dev_callbacks
import logging

logger = logging.getLogger(__name__)

# Load the instruction prompt from file
prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "facebook_prompt.txt")

with open(prompt_path, "r", encoding="utf-8") as f:
    instruction = f.read()

def make_facebook_agent() -> LlmAgent:
    _dev = get_dev_callbacks()
    return LlmAgent(
        name="facebook_searcher",
        model="gemini-2.5-flash",
        instruction=instruction,
        tools=[get_facebook_search_tool()],
        description="Expert agent for finding Facebook profiles",
        before_tool_callback=_dev.before_tool if _dev else None,
        after_tool_callback=_dev.after_tool  if _dev else None,
    )

# Instance singleton (pour root_agent webhook)
facebook_agent = make_facebook_agent()

logger.info("✅ Facebook Searcher Agent initialized")