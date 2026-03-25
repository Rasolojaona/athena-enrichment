"""
Content Extraction Tool — Jina.ai + Gemini

Reads discovered_urls from session.state (populated by web_searcher),
fetches content via Jina.ai, extracts professional data with Gemini,
and stores the result in session.state['web_extracted_data'].

Injection prioritaire : le snippet LinkedIn indexé par Google (web_profile_data)
est passé directement à Gemini sans Jina.ai (LinkedIn bloque tous les scrapers).
"""

import json
import logging
from google.adk.tools import FunctionTool, ToolContext
from ...web_searcher.tools.web_content_extractor import extract_profile_from_urls, _gemini_extract_sync
import asyncio

logger = logging.getLogger(__name__)


def get_content_extract_tool() -> FunctionTool:

    async def extract_web_content(tool_context: ToolContext = None) -> str:
        """
        Fetch and extract professional data from URLs discovered by web_searcher.

        Reads discovered_urls from session.state, fetches pages via Jina.ai,
        extracts structured professional data with Gemini, and stores the result
        in session.state['web_extracted_data'].

        Returns:
            Brief status string (data is in session.state['web_extracted_data'])
        """
        if tool_context is None:
            return "no_tool_context"

        visiteur_data = tool_context.state.get("visiteur_data") or {}
        full_name = visiteur_data.get("full_name") or ""
        if not full_name:
            web_data = tool_context.state.get("web_profile_data") or {}
            if isinstance(web_data, str):
                try:
                    web_data = json.loads(web_data)
                except Exception:
                    web_data = {}
            full_name = web_data.get("full_name") or ""

        if not full_name:
            logger.warning("Content extractor: no full_name in session state")
            tool_context.state["web_extracted_data"] = {}
            return "skipped: no full_name"

        # ── Snippet LinkedIn depuis web_profile_data ──────────────────────────
        # Google indexe les pages LinkedIn et retourne title + snippet + og:description.
        # On injecte ce texte directement à Gemini — pas besoin de Jina.ai pour LinkedIn.
        web_raw = tool_context.state.get("web_profile_data") or {}
        if isinstance(web_raw, str):
            try:
                web_raw = json.loads(web_raw)
            except Exception:
                web_raw = {}

        linkedin_context = ""
        li_title   = web_raw.get("linkedin_title", "") or ""
        li_snippet = web_raw.get("linkedin_snippet", "") or ""
        if li_title or li_snippet:
            linkedin_context = f"LinkedIn Profile (Google index):\nTitle: {li_title}\nDescription: {li_snippet}"
            logger.info(f"Content extractor: LinkedIn snippet available ({len(li_snippet)} chars)")

        # ── Contenu pré-fetché par s.jina.ai (web_searcher Phase 1) ─────────
        # Si web_searcher a déjà fetché le contenu via Jina Search, on l'utilise
        # directement — pas besoin de re-fetcher via r.jina.ai.
        prefetched = tool_context.state.get("jina_prefetched_content") or []
        extracted = {}

        if prefetched:
            logger.info(f"Content extractor: using {len(prefetched)} pre-fetched page(s) from Jina Search for {full_name}")
            combined = ""
            for item in prefetched[:4]:
                url     = item.get("url", "")
                content = item.get("content", "")
                if content and len(content.strip()) > 100:
                    combined += f"\n\n--- Source: {url} ---\n{content[:2000]}"
            if combined.strip():
                extracted = await asyncio.to_thread(_gemini_extract_sync, combined[:6000], full_name) or {}
                if extracted:
                    extracted["source_urls"] = [item["url"] for item in prefetched[:4]]

        # ── URLs r.jina.ai (fallback si pas de contenu pré-fetché) ───────────
        if not extracted:
            discovered = tool_context.state.get("discovered_urls") or {}
            urls_to_extract = []
            if discovered.get("github"):
                urls_to_extract.append(discovered["github"])
            if discovered.get("personal_site"):
                urls_to_extract.append(discovered["personal_site"])
            for m in discovered.get("mentions", []):
                if isinstance(m, dict) and m.get("url"):
                    u = m["url"]
                    if "linkedin.com" in u or "facebook.com" in u:
                        continue
                    urls_to_extract.append(u)

            if urls_to_extract:
                logger.info(f"Content extractor: fetching {len(urls_to_extract)} URL(s) via Jina.ai for {full_name}")
                extracted = await extract_profile_from_urls(full_name, urls_to_extract) or {}

        # ── Si Jina n'a rien trouvé mais on a le snippet LinkedIn ─────────────
        # On passe le snippet seul à Gemini pour extraction
        if linkedin_context and not extracted.get("current_position") and not extracted.get("skills"):
            logger.info(f"Content extractor: extracting from LinkedIn snippet for {full_name}")
            snippet_extracted = await asyncio.to_thread(_gemini_extract_sync, linkedin_context, full_name)
            if snippet_extracted:
                # Merge : Jina data (si existante) prime sur snippet
                for k, v in snippet_extracted.items():
                    if v and not extracted.get(k):
                        extracted[k] = v

        tool_context.state["web_extracted_data"] = extracted if extracted else {}

        if extracted:
            return (
                f"web_extracted_data stored: "
                f"position={extracted.get('current_position')}, "
                f"company={extracted.get('current_company')}, "
                f"skills={len(extracted.get('skills') or [])}, "
                f"experience={len(extracted.get('experience') or [])}"
            )
        return "web_extracted_data stored: no data extracted"

    return FunctionTool(func=extract_web_content)
