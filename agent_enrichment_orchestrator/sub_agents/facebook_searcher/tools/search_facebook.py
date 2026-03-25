"""
Facebook Search Tool - Searches for public Facebook profile info.

Provider priority:
1. Apify (APIFY_API_KEY)  — richer structured data via Facebook scraper actor
2. SerpAPI (SERPAPI_KEY)  — Google Search fallback for public profile snippets
3. No provider            — returns error JSON
"""

import asyncio
import httpx
import os
import logging
import json
import re
import unicodedata
from typing import Optional, Dict, Any, List
from datetime import datetime
from google.adk.tools import FunctionTool, ToolContext

from ....tools.serpapi_utils import serpapi_search, serpapi_search_with_variants, extract_public_mentions

logger = logging.getLogger(__name__)


def _normalize_name(s: str) -> set:
    """Lowercase, supprime les accents, retourne un set de tokens."""
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return set(s.split())


def _name_matches(expected: str, found: str) -> bool:
    """
    Vérifie que found est bien la même personne que expected.
    Règle : tous les tokens d'expected doivent être présents dans found.
    Ex: "Jean Martin" → {"jean", "martin"} ⊆ tokens de found.
    Retourne False si found est vide ou trop court.
    """
    if not expected or not found or len(found.strip()) < 3:
        return False
    exp = _normalize_name(expected)
    fnd = _normalize_name(found)
    if not exp:
        return False
    # Tous les tokens du nom attendu doivent être dans le nom trouvé
    return exp.issubset(fnd)


def get_facebook_search_tool() -> FunctionTool:
    """Create ADK function tool for Facebook / public web profile search."""

    async def search_facebook_profile(
        full_name: str,
        email: str = "",
        location: str = "",
        company: str = "",
        search_type: str = "individual",
        tool_context: ToolContext = None,
    ) -> str:
        """
        Search for a person's public Facebook profile and online presence.

        Uses SerpAPI (Google Search) to find public Facebook profiles and
        general web mentions. Result is written directly to
        session.state['facebook_profile_data'].

        Args:
            full_name: Full name of the person (required)
            email:     Email address for more accurate matching (optional)
            location:  City or country (optional)
            company:   Current company / employer (optional)

        Returns:
            Brief status string (data is in session.state['facebook_profile_data'])
        """
        logger.info(f"Searching Facebook/web presence for: {full_name}")

        apify_key   = os.getenv("APIFY_API_KEY")
        serpapi_key = os.getenv("SERPAPI_KEY")

        result_str = None

        # ── 1. Apify — désactivé par défaut (souscrire lexis-solutions sur Apify d'abord)
        # Pour activer : définir APIFY_ACTOR_ID=lexis-solutions~facebook-user-search-scraper
        actor_id = os.getenv("APIFY_ACTOR_ID", "")
        if apify_key and actor_id:
            result_str = await _search_via_apify(full_name, email, location, company, apify_key, search_type)

        # ── 2. SerpAPI — fallback ─────────────────────────────────────────────
        if not result_str and serpapi_key:
            result_str = await _search_via_serpapi(full_name, email, location, company, serpapi_key, search_type)

        # ── 3. No provider configured ─────────────────────────────────────────
        if not result_str:
            logger.warning("No Facebook/social search provider configured (APIFY_API_KEY or SERPAPI_KEY)")
            result_str = json.dumps({
                "status": "no_provider",
                "reason": "Set APIFY_API_KEY or SERPAPI_KEY to enable Facebook search",
                "data_quality_score": 0.0,
            })

        # Écriture directe dans session.state — bypasse le reformatage LLM
        if tool_context is not None:
            try:
                tool_context.state["facebook_profile_data"] = json.loads(result_str)
            except Exception:
                tool_context.state["facebook_profile_data"] = {"status": "error", "data_quality_score": 0.0}

        try:
            parsed = json.loads(result_str)
            found = bool((parsed.get("facebook_profile") or {}).get("profile_url"))
        except Exception:
            found = False
        return f"facebook_profile_data stored: {'found' if found else 'not_found'} for {full_name}"

    return FunctionTool(search_facebook_profile)


# ─────────────────────────────────────────────
# Apify search (primary)
# ─────────────────────────────────────────────

async def _search_via_apify(
    full_name: str,
    email: str,
    location: str,
    company: str,
    apify_key: str,
    search_type: str = "individual",
) -> Optional[str]:
    """
    Search Facebook via Apify actor (start run → poll → fetch dataset).

    Actor is configurable via APIFY_ACTOR_ID env var.
    Default: lexis-solutions~facebook-user-search-scraper

    Returns JSON string on success, None on failure (triggers SerpAPI fallback).
    """
    actor_id = os.getenv("APIFY_ACTOR_ID", "lexis-solutions~facebook-user-search-scraper")
    timeout  = int(os.getenv("TIMEOUT_SECONDS", "60"))
    poll_interval = 5  # seconds between status checks

    # lexis-solutions actor: search by name query directly on Facebook
    search_query = full_name
    if location:
        search_query += f" {location}"

    actor_input = {
        "queries": [search_query],
        "scrollsAmount": 2,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # ── Start run ──────────────────────────────────────────────────────
            run_resp = await client.post(
                f"https://api.apify.com/v2/acts/{actor_id}/runs",
                params={"token": apify_key},
                json=actor_input,
            )
            run_resp.raise_for_status()
            run_data   = run_resp.json()["data"]
            run_id     = run_data["id"]
            dataset_id = run_data["defaultDatasetId"]
            logger.info(f"Apify run started: {run_id} (actor: {actor_id})")

            # ── Poll for completion ────────────────────────────────────────────
            waited = 0
            status = "RUNNING"
            while waited < timeout and status not in ("SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"):
                await asyncio.sleep(poll_interval)
                waited += poll_interval
                status_resp = await client.get(
                    f"https://api.apify.com/v2/actor-runs/{run_id}",
                    params={"token": apify_key},
                )
                status = status_resp.json()["data"]["status"]
                logger.info(f"Apify run {run_id} status: {status} ({waited}s)")

            if status != "SUCCEEDED":
                logger.warning(f"Apify run ended with status '{status}' — falling back to SerpAPI")
                return None

            # ── Fetch dataset items ────────────────────────────────────────────
            items_resp = await client.get(
                f"https://api.apify.com/v2/datasets/{dataset_id}/items",
                params={"token": apify_key, "format": "json"},
            )
            items_resp.raise_for_status()
            items: List[Dict[str, Any]] = items_resp.json()

        if not items:
            logger.warning(f"Apify returned 0 results for {full_name}")
            return None

        fb_profile      = _parse_apify_profile(items, full_name)
        public_mentions = extract_public_mentions(items, full_name)
        quality         = _compute_quality(fb_profile, public_mentions)

        result = {
            "status":             "found" if fb_profile.get("profile_url") else "partial",
            "full_name":          full_name,
            "facebook_profile":   fb_profile,
            "public_mentions":    public_mentions,
            "data_quality_score": quality,
            "provider":           "apify",
            "searched_at":        datetime.utcnow().isoformat(),
        }
        logger.info(f"✅ Apify Facebook search done for {full_name} — quality: {quality:.2f}")
        return json.dumps(result)

    except Exception as e:
        logger.error(f"Apify Facebook search error for {full_name}: {e}")
        return None


def _parse_apify_profile(items: List[Dict[str, Any]], full_name: str) -> Dict[str, Any]:
    """Parse Apify actor output into a normalised facebook_profile dict.

    Handles lexis-solutions~facebook-user-search-scraper output format:
      { name, profileUrl, userId, userData: [{type, text, icon}] }
    """
    for item in items:
        profile_url = (
            item.get("profileUrl")
            or item.get("url")
            or item.get("facebookUrl")
            or item.get("link")
        )
        if not profile_url or "facebook.com" not in str(profile_url):
            continue

        name = (
            item.get("name")
            or item.get("title")
            or item.get("displayName")
            or ""
        )
        if not _name_matches(full_name, name):
            logger.debug(f"Apify Facebook: nom rejeté '{name}' (attendu: '{full_name}')")
            continue

        profile: Dict[str, Any] = {"profile_url": profile_url}

        if name:
            profile["display_name"] = name

        # ── lexis-solutions userData array ─────────────────────────────────────
        for entry in item.get("userData", []):
            entry_type = entry.get("type", "")
            text = entry.get("text", "")
            if not text:
                continue
            if entry_type == "work" and not profile.get("work"):
                profile["work"] = text
            elif entry_type == "education" and not profile.get("education"):
                profile["education"] = text
            elif entry_type == "location" and not profile.get("location"):
                profile["location"] = text

        # ── Fallback fields from other actor formats ───────────────────────────
        if not profile.get("location"):
            loc = item.get("location") or item.get("city") or item.get("hometown")
            if loc:
                profile["location"] = loc if isinstance(loc, str) else str(loc)

        about = item.get("about") or item.get("description") or item.get("snippet")
        if about:
            profile["snippet"] = str(about)[:500]

        return profile

    return {}


# ─────────────────────────────────────────────
# SerpAPI search (fallback)
# ─────────────────────────────────────────────

async def _search_via_serpapi(
    full_name: str,
    email: str,
    location: str,
    company: str,
    api_key: str,
    search_type: str = "individual",
) -> str:
    timeout = int(os.getenv("TIMEOUT_SECONDS", "30"))

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # ── Query 1: Facebook profile or company page ──────────────────────
            if search_type == "company":
                # Try name variants to handle typos / long names
                fb_results = await serpapi_search_with_variants(
                    client, api_key,
                    query_prefix="site:facebook.com",
                    name=full_name,
                    num=5,
                )
            else:
                # 1. Essai sans company — les profils Facebook ne contiennent pas le nom d'entreprise
                fb_query = f'site:facebook.com "{full_name}"'
                if location:
                    fb_query += f" {location}"
                fb_results = await serpapi_search(client, api_key, fb_query, num=5)
                # 2. Fallback avec company si aucun résultat
                if not fb_results and company:
                    fb_results = await serpapi_search(
                        client, api_key, f'site:facebook.com "{full_name}" {company}', num=5
                    )

            # ── Query 2: General public mentions ──────────────────────────────
            web_query = f'"{full_name}"'
            if company:
                web_query += f' "{company}"'
            if location:
                web_query += f" {location}"

            web_results = await serpapi_search(client, api_key, web_query, num=10)

        # ── Parse results ──────────────────────────────────────────────────────
        fb_profile      = _extract_facebook_profile(fb_results, full_name, search_type)
        public_mentions = extract_public_mentions(web_results, full_name)

        quality = _compute_quality(fb_profile, public_mentions)

        result = {
            "status":           "found" if fb_profile.get("profile_url") else "partial",
            "full_name":        full_name,
            "facebook_profile": fb_profile,
            "public_mentions":  public_mentions,
            "data_quality_score": quality,
            "provider":         "serpapi",
            "searched_at":      datetime.utcnow().isoformat(),
        }

        logger.info(f"✅ Facebook/web search done for {full_name} — quality: {quality:.2f}")
        return json.dumps(result)

    except Exception as e:
        logger.error(f"SerpAPI Facebook search error for {full_name}: {e}")
        return json.dumps({
            "status": "error",
            "reason": str(e),
            "data_quality_score": 0.0,
        })


# ─────────────────────────────────────────────
# Result parsers
# ─────────────────────────────────────────────

def _extract_facebook_profile(
    results: List[Dict], full_name: str, search_type: str = "individual"
) -> Dict[str, Any]:
    """Extract Facebook profile or company page info from search results."""
    profile: Dict[str, Any] = {
        "profile_url":  None,
        "display_name": None,
        "snippet":      None,
        "location":     None,
        "work":         None,
        "education":    None,
    }

    for r in results:
        link = r.get("link", "")
        if "facebook.com" not in link:
            continue

        # Match pattern depends on search_type
        if search_type == "company":
            url_match = re.search(r'facebook\.com/(?!share|sharer|groups|events|login|help)([a-zA-Z0-9.\-_]+)/?', link)
        else:
            # Profil uniquement — exclure photos, videos, posts, reels, watch, etc.
            url_match = re.search(
                r'facebook\.com/(?:people/[^/]+/?$|profile\.php\?id=\d+|[a-zA-Z0-9][a-zA-Z0-9.]+/?$)',
                link
            )
            if url_match and re.search(r'facebook\.com/[^/]+/(?:photos|videos|posts|reels|watch|stories)', link):
                url_match = None  # rejeter les URLs de contenu même si elles matchent

        if url_match:
            display_name = r.get("title", "").replace(" | Facebook", "").strip()
            # Vérification stricte : rejeter si le nom trouvé ne correspond pas
            if not _name_matches(full_name, display_name):
                logger.debug(f"SerpAPI Facebook: titre rejeté '{display_name}' (attendu: '{full_name}')")
                continue
            profile["profile_url"]  = link
            profile["display_name"] = display_name
            profile["snippet"]      = r.get("snippet", "")

            snippet = r.get("snippet", "")
            if search_type == "company":
                # Extract company-relevant info from snippet
                loc_m = re.search(r'(?:located in|based in|headquarters?)\s+([A-Z][^.·,]+)', snippet, re.IGNORECASE)
                if loc_m:
                    profile["location"] = loc_m.group(1).strip()
            else:
                # Extract person-relevant info from snippet
                loc_m = re.search(r'(?:lives? in|based in|from)\s+([A-Z][^.·,]+)', snippet, re.IGNORECASE)
                if loc_m:
                    profile["location"] = loc_m.group(1).strip()

                work_m = re.search(r'(?:works?\s+at|employed\s+at|at)\s+([^.·,\n]+)', snippet, re.IGNORECASE)
                if work_m:
                    profile["work"] = work_m.group(1).strip()

                edu_m = re.search(r'(?:studied\s+at|attended)\s+([^.·,\n]+)', snippet, re.IGNORECASE)
                if edu_m:
                    profile["education"] = edu_m.group(1).strip()

            break  # Take the first matching result

    return {k: v for k, v in profile.items() if v}


def _compute_quality(fb_profile: Dict, mentions: List) -> float:
    score = 0.0
    if fb_profile.get("profile_url"):   score += 0.40
    if fb_profile.get("display_name"):  score += 0.10
    if fb_profile.get("location"):      score += 0.10
    if fb_profile.get("work"):          score += 0.10
    if fb_profile.get("education"):     score += 0.05
    score += min(len(mentions) * 0.05, 0.25)
    return round(min(score, 1.0), 2)
