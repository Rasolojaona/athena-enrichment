"""
Root Agent - Enrichment Pipeline (batch + Pub/Sub).

Workflow :
1. Parallel Phase 1 :
   - LinkedIn Searcher  → linkedin_profile_data  (harvestapi, direct par nom)
   - Facebook Searcher  → facebook_profile_data  (lexis-solutions Apify, direct par nom)
   - Web Searcher       → web_profile_data + discovered_urls (GitHub, site perso, mentions)

2. Sequential Phase 2 :
   - Content Extractor  → web_extracted_data (Jina.ai + Gemini sur les URLs découvertes)

L'agrégation et la sauvegarde Firestore sont faites par main.py en Python pur.
"""

from google.adk.agents import SequentialAgent, ParallelAgent
from .sub_agents import (
    make_linkedin_agent,
    make_facebook_agent,
    make_web_searcher_agent,
    make_content_extractor_agent,
)


def make_parallel_phase1(name: str = "parallel_phase1") -> ParallelAgent:
    """Factory — Phase 1 : LinkedIn + Facebook + web_searcher en parallèle."""
    return ParallelAgent(
        name=name,
        description=(
            "Phase 1: LinkedIn search (harvestapi by name), Facebook search (lexis-solutions Apify), "
            "and web search (Google) run fully in parallel — no dependency between them."
        ),
        sub_agents=[
            make_linkedin_agent(),      # → session.state.linkedin_profile_data
            make_facebook_agent(),      # → session.state.facebook_profile_data
            make_web_searcher_agent(),  # → session.state.web_profile_data + discovered_urls
        ],
    )


# Pipeline principal — utilisé par main.py (Pub/Sub + batch)
root_agent_batch = SequentialAgent(
    name="enrichment_pipeline_batch",
    description=(
        "Enrichment pipeline: LinkedIn + Facebook + web search in parallel (Phase 1), "
        "then content extraction via Jina.ai (Phase 2)."
    ),
    sub_agents=[
        make_parallel_phase1("parallel_phase1_batch"),  # Step 1 → linkedin + facebook + web
        make_content_extractor_agent(),                 # Step 2 → web_extracted_data
    ],
)

# Alias requis par google-adk (cherche `root_agent` dans le module)
root_agent = root_agent_batch
