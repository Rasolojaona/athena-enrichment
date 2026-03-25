"""
Shared Pydantic schemas for inter-agent session.state data.

Each schema validates the data produced by an agent (via output_key) before
the next agent consumes it. This catches malformed data early with clear errors.

Usage in tools:
    raw = tool_context.state.get("web_profile_data")
    web = parse_state("web_profile_data", raw, WebProfileData)
"""

import json
import logging
from typing import Any, Dict, List, Optional, Type, TypeVar
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)


def parse_state(key: str, raw: Any, model: Type[M]) -> Optional[M]:
    """
    Parse and validate a session.state value against a Pydantic model.

    - Accepts str (JSON) or dict.
    - Returns None (not raises) on missing or invalid data, with a clear log.
    """
    if raw is None:
        return None
    try:
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not isinstance(raw, dict):
            logger.warning(f"state['{key}']: expected dict, got {type(raw).__name__}")
            return None
        return model.model_validate(raw)
    except Exception as e:
        logger.warning(f"state['{key}']: schema validation failed — {e}")
        return None


# ── Web Searcher output ──────────────────────────────────────────────────────

class WebMention(BaseModel):
    source: str = ""
    title: str = ""
    snippet: str = ""
    url: str = ""


class WebScrapeData(BaseModel):
    """Company website scrape data (nested inside WebProfileData.data for companies)."""
    title: str = ""
    description: str = ""
    about: List[str] = []
    mission_vision: List[str] = []
    values: List[str] = []
    services: List[str] = []
    products: List[str] = []
    team: List[str] = []
    testimonials: List[str] = []
    news_blog: List[str] = []
    job_openings: List[str] = []
    geographic_presence: List[str] = []
    partnerships: List[str] = []
    contact_info: Dict[str, str] = {}
    social_links: Dict[str, str] = {}
    technology: List[str] = []

    class Config:
        extra = "allow"


class WebProfileData(BaseModel):
    """Output of web_searcher agent → session.state['web_profile_data']"""
    status: str = "not_found"
    # Individual fields
    full_name: str = ""
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    personal_site: Optional[str] = None
    inferred_company: Optional[str] = None
    mentions: List[WebMention] = []
    # Company fields
    website_url: Optional[str] = None
    company_name: Optional[str] = None
    data: Optional[WebScrapeData] = None
    data_quality_score: float = 0.0
    provider: Optional[str] = None

    class Config:
        extra = "allow"


# ── LinkedIn Searcher output ─────────────────────────────────────────────────

class LinkedInExperience(BaseModel):
    title: Optional[str] = None
    company: Optional[str] = None
    starts_at: Optional[Any] = None
    ends_at: Optional[Any] = None
    description: Optional[str] = None

    class Config:
        extra = "allow"


class LinkedInEducation(BaseModel):
    school: Optional[str] = None
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    starts_at: Optional[Any] = None
    ends_at: Optional[Any] = None

    class Config:
        extra = "allow"


class LinkedInProfileData(BaseModel):
    """Output of linkedin_searcher agent → session.state['linkedin_profile_data']"""
    status: Optional[str] = None          # "not_found" | None (found)
    type: Optional[str] = None            # "individual_profile" | "company_page"
    profile_url: Optional[str] = None
    headline: Optional[str] = None
    current_position: Optional[str] = None
    current_company: Optional[str] = None
    location: Optional[str] = None
    industry: Optional[str] = None
    connections_count: Optional[int] = None
    experience: List[LinkedInExperience] = []
    education: List[LinkedInEducation] = []
    skills: List[str] = []
    summary: Optional[str] = None
    data_quality_score: float = 0.0
    provider: Optional[str] = None

    class Config:
        extra = "allow"


# ── Facebook Searcher output ─────────────────────────────────────────────────

class FacebookProfile(BaseModel):
    profile_url: Optional[str] = None
    display_name: Optional[str] = None
    snippet: Optional[str] = None
    location: Optional[str] = None
    work: Optional[str] = None
    education: Optional[str] = None

    class Config:
        extra = "allow"


class FacebookProfileData(BaseModel):
    """Output of facebook_searcher agent → session.state['facebook_profile_data']"""
    status: Optional[str] = None
    full_name: str = ""
    facebook_profile: Optional[FacebookProfile] = None
    public_mentions: List[Dict[str, Any]] = []
    data_quality_score: float = 0.0
    provider: Optional[str] = None

    class Config:
        extra = "allow"
