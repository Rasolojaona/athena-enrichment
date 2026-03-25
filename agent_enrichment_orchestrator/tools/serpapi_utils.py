"""
Search utilities — Serper.dev (primary) + SerpAPI (fallback).

Priority:
1. Serper.dev  (SERPER_API_KEY)  — 10x cheaper, ~300ms, POST /search
2. SerpAPI     (SERPAPI_KEY)     — fallback si Serper échoue ou clé absente

Tous les outils utilisent serpapi_search() — le routing est transparent.
"""

import os
import re
import logging
from typing import Any, Dict, List, Optional, Set

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_SKIP_DOMAINS: Set[str] = {
    "facebook.com", "instagram.com", "twitter.com", "x.com"
}


async def _serper_search(
    client: httpx.AsyncClient,
    api_key: str,
    query: str,
    num: int = 10,
) -> List[Dict[str, Any]]:
    """Recherche Google via Serper.dev (POST). Retourne [] en cas d'erreur."""
    try:
        resp = await client.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": num},
        )
        if resp.status_code == 200:
            # Serper retourne "organic" (pas "organic_results")
            return resp.json().get("organic", [])
        logger.warning(f"Serper HTTP {resp.status_code} for query: {query}")
    except Exception as e:
        logger.warning(f"Serper request failed for query '{query}': {e}")
    return []


async def _serpapi_search(
    client: httpx.AsyncClient,
    api_key: str,
    query: str,
    num: int = 10,
) -> List[Dict[str, Any]]:
    """Recherche Google via SerpAPI (GET). Retourne [] en cas d'erreur."""
    try:
        resp = await client.get(
            "https://serpapi.com/search",
            params={"api_key": api_key, "engine": "google", "q": query, "num": num},
        )
        if resp.status_code == 200:
            return resp.json().get("organic_results", [])
        logger.warning(f"SerpAPI HTTP {resp.status_code} for query: {query}")
    except Exception as e:
        logger.warning(f"SerpAPI request failed for query '{query}': {e}")
    return []


async def serpapi_search(
    client: httpx.AsyncClient,
    api_key: str,
    query: str,
    num: int = 10,
) -> List[Dict[str, Any]]:
    """
    Recherche Google unifiée : Serper en primaire, SerpAPI en fallback.

    Serper.dev ne supporte pas bien l'opérateur site: — il retourne 0 résultats
    même quand des pages existent. On retire donc site: pour Serper et on laisse
    les appelants filtrer par URL. SerpAPI reçoit la query originale (avec site:).
    """
    serper_key = os.getenv("SERPER_API_KEY", "")
    serpapi_key = os.getenv("SERPAPI_KEY", api_key)

    # 1. Serper (primaire) — site: retiré car mal géré, les callers filtrent par URL
    if serper_key:
        has_site_op = bool(re.search(r'\bsite:\S+', query))
        serper_query = re.sub(r'\bsite:\S+\s*', '', query).strip() if has_site_op else query
        # Demander plus de résultats pour compenser l'absence du filtre site:
        serper_num = max(num, 10) if has_site_op else num
        results = await _serper_search(client, serper_key, serper_query, serper_num)
        if results:
            logger.debug(f"Serper OK ({len(results)} results): {query[:60]}")
            return results
        logger.info(f"Serper 0 results (query: {serper_query[:60]}), trying SerpAPI")

    # 2. SerpAPI (fallback) — query originale avec site: intacte
    if serpapi_key:
        return await _serpapi_search(client, serpapi_key, query, num)

    return []


_LEGAL_SUFFIXES = {
    "inc", "llc", "ltd", "sarl", "sas", "sa", "group", "groups",
    "international", "global", "holding", "holdings",
}

_NOISE_WORDS = {
    "and", "et", "the", "le", "la", "les", "de", "du", "des", "of",
    "for", "or", "in", "data", "ai", "tech", "digital",
}


def linkedin_person_name_variants(full_name: str) -> List[str]:
    """
    Generate LinkedIn-specific search variants for a person's name.

    Handles:
    - Compound first names: "Ny Avo Sarobidy ANDRIATSILAVO"
      → drops middle part → "Ny Avo ANDRIATSILAVO"
    - ALL-CAPS surname detection (malgasy / French convention)
    - Reversed order: "ANDRIATSILAVO Ny Avo Sarobidy"
    - Fallback: first name only, surname only

    Returns a deduplicated ordered list (most specific first).
    """
    variants: List[str] = []
    seen: set = set()

    def _add(v: str) -> None:
        v = v.strip()
        if v and v not in seen:
            seen.add(v)
            variants.append(f'"{v}"')

    words = full_name.strip().split()
    if not words:
        return []

    # Detect ALL-CAPS tokens as surnames (malgasy / African naming convention)
    surnames = [w for w in words if w.isupper() and len(w) > 2]
    first_words = [w for w in words if not (w.isupper() and len(w) > 2)]

    # Fallback: if no ALL-CAPS detected, assume last word is surname (Western)
    if not surnames:
        surnames = [words[-1]]
        first_words = words[:-1]

    surname_str = " ".join(surnames)
    first_str = " ".join(first_words)

    # 1. Full name exact
    _add(full_name)

    # 2. First 2 first-name tokens + surname (drop extra middle names)
    if len(first_words) >= 3:
        _add(" ".join(first_words[:2]) + " " + surname_str)

    # 3. First first-name token + surname (shortest useful variant)
    if len(first_words) >= 2 and surnames:
        _add(first_words[0] + " " + surname_str)

    # 4. Reversed: surname first (French/malgasy administrative convention)
    if surnames and first_words:
        _add(surname_str + " " + first_str)

    # 5. First name only (useful when LinkedIn shows first name without full surname)
    if len(first_words) >= 2:
        _add(first_str)

    return variants


async def serpapi_linkedin_search(
    client: httpx.AsyncClient,
    api_key: str,
    full_name: str,
    location: str = "",
    company: str = "",
) -> List[Dict[str, Any]]:
    """
    Search LinkedIn profiles via SerpAPI, trying name variants until results are found.

    Tries each variant of the person's name with site:linkedin.com/in/.
    Optionally appends location or company to narrow results.

    Returns first non-empty results list, or [] if all variants fail.
    """
    location_suffix = f" {location}" if location else ""
    company_suffix = f" {company}" if company else ""

    for variant in linkedin_person_name_variants(full_name):
        # Try without extra context first
        query = f"site:linkedin.com/in/ {variant}{location_suffix}"
        results = await serpapi_search(client, api_key, query, num=5)
        if results:
            logger.info(f"LinkedIn variant matched: {query}")
            return results
        logger.info(f"LinkedIn variant 0 results: {query}")

        # Retry with company if provided and different from the base variant
        if company_suffix:
            query_c = f"site:linkedin.com/in/ {variant}{company_suffix}"
            results = await serpapi_search(client, api_key, query_c, num=5)
            if results:
                logger.info(f"LinkedIn variant+company matched: {query_c}")
                return results

    return []


def name_variants(name: str) -> List[str]:
    """
    Generate ordered search query variants for a company or person name.

    Addresses typos, hyphens, and over-long names that cause SerpAPI
    quoted-string searches to return zero results.

    Returns a deduplicated list starting with the most specific variant.

    Example:
        "ClearMind-Anlytics Mada Data and AI"
        → ['"ClearMind-Anlytics Mada Data and AI"',
           '"ClearMind Anlytics Mada Data and AI"',
           '"ClearMind Anlytics"',
           '"ClearMind"']
    """
    variants: List[str] = []
    seen: set = set()

    def _add(v: str) -> None:
        v = v.strip()
        if v and v not in seen:
            seen.add(v)
            variants.append(f'"{v}"')

    # 1. Original name (exact)
    _add(name)

    # 2. Hyphens replaced by spaces
    no_hyphen = name.replace("-", " ")
    _add(no_hyphen)

    # 3. Remove legal suffixes and noise from the end
    words = no_hyphen.split()
    meaningful = [w for w in words if w.lower() not in _LEGAL_SUFFIXES | _NOISE_WORDS and len(w) > 1]

    # 4. First 3 meaningful words
    if len(meaningful) >= 3:
        _add(" ".join(meaningful[:3]))

    # 5. First 2 meaningful words
    if len(meaningful) >= 2:
        _add(" ".join(meaningful[:2]))

    # 6. First meaningful word alone (broadest fallback)
    if meaningful:
        _add(meaningful[0])

    return variants


async def serpapi_search_with_variants(
    client: httpx.AsyncClient,
    api_key: str,
    query_prefix: str,
    name: str,
    num: int = 5,
) -> List[Dict[str, Any]]:
    """
    Try SerpAPI queries for each name variant until one returns results.

    Args:
        client:       An active httpx.AsyncClient.
        api_key:      SerpAPI key.
        query_prefix: Text prepended to the quoted name (e.g. "site:linkedin.com/company/").
        name:         Company or person name — variants are generated automatically.
        num:          Results per query.

    Returns:
        First non-empty results list, or empty list if all variants fail.
    """
    for variant in name_variants(name):
        query = f"{query_prefix} {variant}".strip()
        results = await serpapi_search(client, api_key, query, num=num)
        if results:
            logger.info(f"SerpAPI matched with variant: {query}")
            return results
        logger.info(f"SerpAPI no results for: {query}")
    return []


def extract_public_mentions(
    results: List[Dict[str, Any]],
    full_name: str,
    skip_domains: Optional[Set[str]] = None,
    max_mentions: int = 5,
    snippet_max_chars: int = 300,
) -> List[Dict[str, str]]:
    """
    Extract relevant public web mentions from SerpAPI organic results.

    Args:
        results:          Organic results list from serpapi_search.
        full_name:        Person's full name — only keep results that mention it.
        skip_domains:     Domains to exclude (default: facebook, instagram, twitter, x).
        max_mentions:     Maximum number of mentions to return (default 5).
        snippet_max_chars: Truncate snippets to this length (default 300).

    Returns:
        List of dicts with keys: source, title, snippet, url.
    """
    if skip_domains is None:
        skip_domains = _DEFAULT_SKIP_DOMAINS

    mentions: List[Dict[str, str]] = []
    for r in results:
        link = r.get("link", "")
        domain = re.sub(r'^(?:https?://)?(?:www\.)?', '', link).split('/')[0]

        if domain in skip_domains:
            continue
        if full_name.lower() not in (r.get("title", "") + r.get("snippet", "")).lower():
            continue

        mentions.append({
            "source":  domain,
            "title":   r.get("title", ""),
            "snippet": r.get("snippet", "")[:snippet_max_chars],
            "url":     link,
        })

        if len(mentions) >= max_mentions:
            break

    return mentions
