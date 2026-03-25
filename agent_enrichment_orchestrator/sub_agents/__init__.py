"""
Sub-Agents Package.

Contient tous les agents spécialisés utilisés par l'orchestrateur :
- web_searcher      : Recherche présence web (GitHub, LinkedIn URL hint, site perso)
                      → session.state["web_profile_data"] + session.state["discovered_urls"]
- linkedin_searcher : Recherche profils LinkedIn
                      → session.state["linkedin_profile_data"]
- facebook_searcher : Recherche profils Facebook
                      → session.state["facebook_profile_data"]
- content_extractor : Extraction Jina.ai + Gemini depuis les URLs découvertes
                      → session.state["web_extracted_data"]
"""

from .linkedin_searcher.agent import linkedin_agent, make_linkedin_agent
from .facebook_searcher.agent import facebook_agent, make_facebook_agent
from .web_searcher.agent import web_searcher_agent, make_web_searcher_agent
from .content_extractor.agent import content_extractor_agent, make_content_extractor_agent

__all__ = [
    "linkedin_agent",
    "facebook_agent",
    "web_searcher_agent",
    "content_extractor_agent",
    "make_linkedin_agent",
    "make_facebook_agent",
    "make_web_searcher_agent",
    "make_content_extractor_agent",
]