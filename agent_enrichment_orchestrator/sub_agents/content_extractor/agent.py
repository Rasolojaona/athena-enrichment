"""
Content Extractor Agent — Jina.ai + Gemini web extraction.

Runs in parallel with linkedin_searcher and facebook_searcher (Phase 2).
Reads discovered_urls from session.state, fetches pages via Jina.ai,
extracts professional data with Gemini, writes web_extracted_data.
"""

import os
from google.adk.agents import LlmAgent
from .tools import get_content_extract_tool
from ...tools.dev_callbacks import get_dev_callbacks
import logging

logger = logging.getLogger(__name__)

prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "content_extractor_prompt.txt")

with open(prompt_path, "r", encoding="utf-8") as f:
    instruction = f.read()


def make_content_extractor_agent() -> LlmAgent:
    _dev = get_dev_callbacks()
    return LlmAgent(
        name="content_extractor",
        model="gemini-2.5-flash",
        instruction=instruction,
        tools=[get_content_extract_tool()],
        description="Fetches URLs discovered by web_searcher via Jina.ai and extracts professional data with Gemini",
        before_tool_callback=_dev.before_tool if _dev else None,
        after_tool_callback=_dev.after_tool  if _dev else None,
    )


# Instance singleton (pour root_agent)
content_extractor_agent = make_content_extractor_agent()

logger.info("✅ Content Extractor Agent initialized")
