"""
Web Content Extractor — Jina.ai + Gemini

Stratégie :
- LinkedIn / réseaux sociaux → skip (bloqués côté serveur)
- GitHub, sites perso, mentions publiques → Jina.ai Reader
- Extraction → Gemini 2.5 Flash avec prompt strict et validé

Flow :
1. Reçoit une liste d'URLs depuis content_extract_tool
2. Filtre les domaines bloqués (LinkedIn, Facebook, etc.)
3. Fetch via Jina.ai en parallèle
4. Combine le contenu (max 6000 chars)
5. Gemini extrait les données structurées
6. Retourne un dict compatible Firestore
"""

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List

import httpx

logger = logging.getLogger(__name__)

_genai_client = None


def _get_genai_client():
    global _genai_client
    if _genai_client is None:
        import google.genai as google_genai
        if os.getenv("GOOGLE_GENAI_USE_VERTEXAI") == "1":
            _genai_client = google_genai.Client(
                vertexai=True,
                project=os.getenv("GOOGLE_CLOUD_PROJECT"),
                location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
            )
        else:
            _genai_client = google_genai.Client(api_key=os.getenv("GOOGLE_API_KEY", ""))
    return _genai_client


# Domaines inaccessibles (auth requise ou blocage IP)
_SKIP_DOMAINS = {
    "linkedin.com",
    "facebook.com", "instagram.com",
    "twitter.com", "x.com", "tiktok.com", "youtube.com",
    "google.com", "bing.com",
}

_JINA_BASE   = "https://r.jina.ai/"
_MAX_CONTENT = 6000
_MAX_URLS    = 4


def _is_extractable(url: str) -> bool:
    if not url:
        return False
    return not any(d in url for d in _SKIP_DOMAINS)


# ─────────────────────────────────────────────
# Jina.ai Reader
# ─────────────────────────────────────────────

async def _fetch_via_jina(client: httpx.AsyncClient, url: str) -> str:
    """
    Retourne le contenu markdown de l'URL via Jina.ai Reader.

    Sans JINA_API_KEY : 20 req/min (gratuit)
    Avec JINA_API_KEY : 500 req/min (gratuit sur https://jina.ai/?sui=apikey)
    """
    headers = {"X-Return-Format": "markdown", "X-No-Cache": "true"}
    jina_key = os.getenv("JINA_API_KEY", "")
    if jina_key:
        headers["Authorization"] = f"Bearer {jina_key}"

    try:
        resp = await client.get(
            f"{_JINA_BASE}{url}",
            headers=headers,
            timeout=25,
            follow_redirects=True,
        )
        if resp.status_code == 200:
            logger.info(f"Jina.ai OK: {url[:70]}")
            return resp.text
        logger.warning(f"Jina.ai HTTP {resp.status_code} for {url[:70]}")
    except Exception as e:
        logger.warning(f"Jina.ai fetch error ({url[:60]}): {type(e).__name__}: {e}")
    return ""


# ─────────────────────────────────────────────
# Gemini extraction
# ─────────────────────────────────────────────

def _gemini_extract_sync(content: str, full_name: str) -> dict:
    """
    Extrait les données professionnelles depuis le contenu web via Gemini.
    Prompt strict pour éviter les confusions position/company et les faux skills.
    """
    try:
        client = _get_genai_client()

        prompt = f"""You are a professional data extractor. Extract information about "{full_name}" from the web content below.

Return ONLY a valid JSON object. Use null for any missing field. Use [] for empty lists.

JSON format to return:
{{
  "current_position": "exact job title (NOT company name) — e.g. 'Software Engineer', 'Data Analyst', 'CEO'. null if unknown.",
  "current_company": "company or organization name — e.g. 'Google', 'ClearMind Analytics'. null if unknown.",
  "description": "professional bio or summary in 1-2 sentences, max 200 chars. null if none.",
  "skills": ["technical skill or tool", "..."],
  "education": [{{"school": "university or school name", "degree": "degree type e.g. Bachelor, Master, PhD", "year": "graduation year or null"}}],
  "experience": [{{"title": "job title", "company": "company name", "duration": "e.g. 2020-2023 or null"}}]
}}

STRICT RULES — follow exactly:
1. current_position = job title only. NEVER put a company name as current_position.
   BAD: "current_position": "ClearMind-Analytics"
   GOOD: "current_position": "Software Engineer"

2. skills = only real technical skills (languages, tools, frameworks, methodologies).
   NEVER include: locations, names, labels like "Location: ...", company names, generic words.
   BAD: ["Location: Analamanga", "ClearMind"]
   GOOD: ["Python", "React", "Docker", "Machine Learning"]

3. If current_position and current_company are identical → set current_position to null.

4. Only extract information that clearly refers to {full_name}.
   Do NOT invent or guess — null is always better than wrong data.

5. Return ONLY the JSON. No markdown, no explanation, no code block.

Web content to analyze:
{content}"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        text = response.text.strip()
        match = re.search(r'\{[\s\S]*\}', text)
        if not match:
            logger.warning("Gemini returned no JSON block")
            return {}

        parsed = json.loads(match.group(0))

        # Post-validation Python — filet de sécurité après le JSON parsé
        cleaned = {}
        for k, v in parsed.items():
            if v in (None, [], ""):
                continue
            if k == "current_position" and isinstance(v, str):
                company = parsed.get("current_company", "") or ""
                if v.strip().lower() == company.strip().lower():
                    continue
            if k == "skills" and isinstance(v, list):
                v = [
                    s for s in v
                    if isinstance(s, str)
                    and len(s.strip()) > 1
                    and not re.match(
                        r'^(location|experience|education|followers?|connections?)\s*:',
                        s, re.IGNORECASE
                    )
                    and len(s.split()) <= 5
                ]
                if not v:
                    continue
            cleaned[k] = v

        return cleaned

    except Exception as e:
        logger.warning(f"Gemini extraction error: {type(e).__name__}: {e}")
        return {}


# ─────────────────────────────────────────────
# Point d'entrée principal
# ─────────────────────────────────────────────

async def extract_profile_from_urls(
    full_name: str,
    urls: List[str],
) -> dict:
    """
    Fetche les URLs publiques via Jina.ai et extrait les données avec Gemini.

    Args:
        full_name : Nom complet (utilisé dans le prompt Gemini)
        urls      : Liste d'URLs candidates (LinkedIn et réseaux sociaux filtrés)

    Returns:
        Dict avec : current_position, current_company, description, skills,
                    education, experience, source_urls
        Retourne {} si aucun contenu récupérable.
    """
    seen: set = set()
    unique_urls = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            unique_urls.append(u)

    extractable = [u for u in unique_urls if _is_extractable(u)][:_MAX_URLS]

    if not extractable:
        logger.info(f"No extractable URLs for {full_name}")
        return {}

    logger.info(f"Fetching {len(extractable)} URL(s) via Jina.ai for {full_name}")

    async with httpx.AsyncClient(timeout=30) as client:
        results = await asyncio.gather(
            *[_fetch_via_jina(client, u) for u in extractable],
            return_exceptions=True,
        )

    combined   = ""
    valid_urls = []
    for url, content in zip(extractable, results):
        if isinstance(content, str) and len(content.strip()) > 100:
            combined  += f"\n\n--- Source: {url} ---\n{content[:2000]}"
            valid_urls.append(url)

    if not combined.strip():
        logger.info(f"No usable content fetched for {full_name}")
        return {}

    combined = combined[:_MAX_CONTENT]

    extracted = await asyncio.to_thread(_gemini_extract_sync, combined, full_name)

    if extracted:
        extracted["source_urls"] = valid_urls
        logger.info(
            f"Extraction done for {full_name} — "
            f"position={extracted.get('current_position')} | "
            f"company={extracted.get('current_company')} | "
            f"skills={len(extracted.get('skills') or [])} | "
            f"experience={len(extracted.get('experience') or [])} | "
            f"education={len(extracted.get('education') or [])}"
        )
    else:
        logger.info(f"Gemini found nothing extractable for {full_name}")

    return extracted
