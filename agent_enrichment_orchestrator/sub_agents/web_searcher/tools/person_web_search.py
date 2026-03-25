"""
Person Web Presence Search Tool
Cherche la présence en ligne d'un individu :
  - GitHub par email via l'API officielle (précis)
  - LinkedIn via SerpAPI
  - Site personnel / portfolio via Jina Search (s.jina.ai) — contenu complet
  - Mentions publiques (articles, conférences)
"""

import os
import json
import logging
import re
import unicodedata
import httpx
from google.adk.tools import FunctionTool, ToolContext
from ....tools.serpapi_utils import serpapi_search, serpapi_linkedin_search, extract_public_mentions

logger = logging.getLogger(__name__)

_JINA_SEARCH_BASE = "https://s.jina.ai/"


async def _search_via_jina(query: str) -> list:
    """
    Jina Search API (s.jina.ai) — retourne des résultats avec contenu complet.

    Avantage vs SerpAPI : chaque résultat inclut le contenu markdown de la page,
    pas seulement le snippet de 150 chars.

    Sans JINA_API_KEY : fonctionne mais limité (20 req/min)
    Avec JINA_API_KEY : 500 req/min (gratuit sur jina.ai)

    Timeout dédié (45s) — indépendant du client partagé (20s) pour éviter
    les ReadTimeout sur s.jina.ai qui est plus lent que SerpAPI.

    Returns:
        Liste de dicts {url, title, content, description}
    """
    headers = {"Accept": "application/json", "X-Timeout": "40"}
    jina_key = os.getenv("JINA_API_KEY", "")
    if jina_key:
        headers["Authorization"] = f"Bearer {jina_key}"

    try:
        async with httpx.AsyncClient(timeout=45) as jina_client:
            resp = await jina_client.get(
                _JINA_SEARCH_BASE,
                params={"q": query, "num": 5},
                headers=headers,
            )
        if resp.status_code != 200:
            logger.warning(f"Jina Search HTTP {resp.status_code} for query: {query[:60]}")
            return []

        data = resp.json()
        results = data.get("data") or []
        logger.info(f"Jina Search: {len(results)} résultats pour '{query[:60]}'")
        return results

    except Exception as e:
        logger.warning(f"Jina Search error ({query[:50]}): {type(e).__name__}: {e}")
        return []


_GENERIC_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "icloud.com", "live.com", "ymail.com", "aol.com", "protonmail.com",
}

_SOCIAL_DOMAINS = {
    "linkedin.com", "github.com", "facebook.com",
    "instagram.com", "twitter.com", "x.com", "tiktok.com",
}

# Common skill/tech keywords to extract from snippets
_SKILL_PATTERNS = re.compile(
    r'\b(python|java|javascript|typescript|react|node\.?js|angular|vue|'
    r'sql|nosql|mongodb|postgresql|mysql|redis|'
    r'docker|kubernetes|terraform|ansible|'
    r'machine\s*learning|deep\s*learning|ml|mlops|nlp|ai|'
    r'data\s*science|data\s*engineer|data\s*analyst|'
    r'aws|gcp|azure|google\s*cloud|'
    r'django|flask|fastapi|spring|laravel|'
    r'git|linux|bash|devops|ci/?cd|'
    r'r\b|scala|kotlin|swift|go|rust|c\+\+|c#)\b',
    re.IGNORECASE,
)


def _normalize_tokens(s: str) -> set:
    """Lowercase + remove accents → set of tokens."""
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return set(s.split())


def _name_in_text(full_name: str, text: str) -> bool:
    """
    Return True if the text likely refers to this person.

    Strategy: require the 2 longest tokens of the name (= most distinctive).
    This handles snippets that show "Sarobidy ANDRIATSILAVO" instead of
    "Ny Avo Sarobidy ANDRIATSILAVO", while still avoiding false positives.
    """
    if not full_name or not text:
        return False
    all_tokens = sorted(_normalize_tokens(full_name), key=len, reverse=True)
    # Keep the 2 most distinctive tokens (longest); min 1 if name has 1 token
    required = set(all_tokens[:2]) if len(all_tokens) >= 2 else set(all_tokens[:1])
    fnd = _normalize_tokens(text)
    return required.issubset(fnd)


async def _search_github_by_email(client: httpx.AsyncClient, email: str) -> dict:
    """
    Cherche un profil GitHub par email via l'API officielle.
    Retourne un dict avec les infos du profil ou {} si non trouvé.
    """
    token = os.getenv("GITHUB_TOKEN", "")
    if not token or not email:
        return {}

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    try:
        # Étape 1 : chercher l'utilisateur par email
        resp = await client.get(
            "https://api.github.com/search/users",
            params={"q": f"{email}+in:email", "per_page": 1},
            headers=headers,
        )
        if resp.status_code != 200 or not resp.json().get("items"):
            logger.info(f"GitHub: aucun profil trouvé pour {email}")
            return {}

        login = resp.json()["items"][0]["login"]
        profile_url = resp.json()["items"][0]["html_url"]

        # Étape 2 : récupérer les détails du profil
        resp2 = await client.get(
            f"https://api.github.com/users/{login}",
            headers=headers,
        )
        if resp2.status_code != 200:
            return {"github_url": profile_url, "github_login": login}

        data = resp2.json()
        logger.info(f"✅ GitHub trouvé : {profile_url}")
        return {
            "github_url":       profile_url,
            "github_login":     login,
            "github_name":      data.get("name"),
            "github_bio":       data.get("bio"),
            "github_company":   data.get("company", "").lstrip("@") if data.get("company") else None,
            "github_blog":      data.get("blog") or None,
            "github_location":  data.get("location"),
            "github_public_repos": data.get("public_repos", 0),
            "github_followers": data.get("followers", 0),
        }

    except Exception as e:
        logger.warning(f"GitHub API error: {e}")
        return {}


def get_person_web_search_tool() -> FunctionTool:

    async def search_person_web_presence(
        full_name: str,
        email: str = "",
        location: str = "",
        company: str = "",
        event_title: str = "",
        tool_context: ToolContext = None,
    ) -> str:
        """
        Search for a person's public web presence.

        Uses GitHub API (by email) for accurate GitHub profile lookup,
        and SerpAPI for LinkedIn, personal site, and public mentions.
        Result is written directly to session.state['web_profile_data'].

        Args:
            full_name:   Full name of the person
            email:       Email address — used for GitHub API lookup + company inference
            location:    City / country to narrow search results
            company:     Current company if known
            event_title: Event they attended (context for search)

        Returns:
            Brief status string (data is in session.state['web_profile_data'])
        """
        serpapi_key = os.getenv("SERPAPI_KEY", "")
        logger.info(f"🌐 Web presence search for: {full_name} ({email})")

        # Inférer la company depuis le domaine email
        email_company = ""
        if email and "@" in email:
            domain = email.split("@")[-1].lower()
            if domain not in _GENERIC_DOMAINS:
                email_company = domain.split(".")[0]

        context = company or email_company or location or event_title or ""

        found = {
            "github_url":       None,
            "github_login":     None,
            "github_bio":       None,
            "github_company":   None,
            "github_blog":      None,
            "github_location":  None,
            "github_public_repos": 0,
            "github_followers": 0,
            "personal_site":    None,
            "linkedin_url":     None,
            "linkedin_snippet": None,
            "linkedin_title":   None,
            "facebook_url":     None,
            "professional_description": None,
            "skills":           [],
            "mentions":         [],
            "inferred_company": email_company or None,
        }

        jina_prefetched: list = []

        async with httpx.AsyncClient(timeout=20) as client:

            # ── 1. GitHub par email (API officielle) ─────────────────────────
            if email:
                gh = await _search_github_by_email(client, email)
                if gh:
                    found.update(gh)
                    # Le blog GitHub est souvent le site perso
                    if gh.get("github_blog") and not found["personal_site"]:
                        found["personal_site"] = gh["github_blog"]
                    # La company GitHub enrichit l'inférence
                    if gh.get("github_company") and not found["inferred_company"]:
                        found["inferred_company"] = gh["github_company"]

            # ── 2. SerpAPI : LinkedIn + site perso + mentions ────────────────
            if serpapi_key:
                # ── LinkedIn : essai avec variants de nom (prénom composé, ordre inversé)
                linkedin_results = await serpapi_linkedin_search(
                    client, serpapi_key, full_name,
                    location=location or "",
                    company=company or "",
                )
                for r in linkedin_results:
                    link = r.get("link", "")
                    if not found["linkedin_url"] and "linkedin.com/in/" in link:
                        found["linkedin_url"]     = link
                        found["linkedin_snippet"] = r.get("snippet", "")
                        found["linkedin_title"]   = r.get("title", "")
                        break

                # ── Jina Search : portfolio / site perso / GitHub / mentions ──
                # s.jina.ai retourne contenu complet (vs snippet 150 chars SerpAPI)
                # Langage naturel uniquement — s.jina.ai ne supporte pas les opérateurs booléens OR
                jina_query = f"{full_name} portfolio github website"
                if context:
                    jina_query = f"{full_name} {context} portfolio github website"
                jina_results = await _search_via_jina(jina_query)

                # Stocker le contenu pré-fetché pour content_extractor
                jina_prefetched = []

                for r in jina_results:
                    link    = r.get("url", "")
                    title   = r.get("title", "").lower()
                    content = r.get("content", "") or r.get("description", "")

                    if not link:
                        continue

                    # LinkedIn (si pas encore trouvé)
                    if not found["linkedin_url"] and "linkedin.com/in/" in link:
                        found["linkedin_url"]     = link
                        found["linkedin_snippet"] = r.get("description", "")[:300]
                        found["linkedin_title"]   = r.get("title", "")

                    # GitHub
                    if not found["github_url"] and "github.com/" in link:
                        if not any(x in link for x in ["/issues/", "/pull/", "/commit/", "/blob/"]):
                            found["github_url"] = link
                            logger.info(f"Jina Search found GitHub: {link}")

                    # Site perso
                    if not found["personal_site"]:
                        if any(kw in title for kw in ["portfolio", "personal", "blog", "about", full_name.lower()]):
                            if not any(d in link for d in _SOCIAL_DOMAINS):
                                found["personal_site"] = link

                    # Stocker le contenu pour content_extractor (évite r.jina.ai en double)
                    if content and len(content.strip()) > 100:
                        if not any(d in link for d in {"linkedin.com", "facebook.com", "instagram.com", "twitter.com", "x.com"}):
                            jina_prefetched.append({"url": link, "content": content[:2000]})

                    # Skills depuis le contenu
                    matches = _SKILL_PATTERNS.findall(f"{title} {content[:500]}")
                    for m in matches:
                        found.setdefault("_skill_set", set()).add(m.lower().replace(" ", "_"))

                # Fallback SerpAPI si Jina Search n'a rien retourné
                if not jina_results:
                    queries = [
                        f'"{full_name}" {context}'.strip() if context else f'"{full_name}"',
                        f'"{full_name}" portfolio OR website OR blog',
                    ]
                    for query in queries:
                        results = await serpapi_search(client, serpapi_key, query, num=5)
                        if not results:
                            continue
                        for r in results:
                            link  = r.get("link", "")
                            title = r.get("title", "").lower()
                            if not found["linkedin_url"] and "linkedin.com/in/" in link:
                                found["linkedin_url"]     = link
                                found["linkedin_snippet"] = r.get("snippet", "")
                                found["linkedin_title"]   = r.get("title", "")
                            if not found["personal_site"]:
                                if any(kw in title for kw in ["portfolio", "personal", "blog", "about me", full_name.lower()]):
                                    if not any(d in link for d in _SOCIAL_DOMAINS):
                                        found["personal_site"] = link
                        mentions = extract_public_mentions(
                            results, full_name=full_name,
                            skip_domains={"linkedin.com", "facebook.com", "instagram.com", "twitter.com", "x.com", "github.com"},
                        )
                        for m in mentions:
                            if m not in found["mentions"]:
                                found["mentions"].append(m)
                else:
                    # Mentions depuis les résultats Jina
                    # Pas besoin de _name_in_text : Jina a déjà cherché par nom,
                    # les résultats sont pertinents par construction.
                    for r in jina_results:
                        link    = r.get("url", "")
                        content = r.get("content", "") or r.get("description", "")
                        if not link or any(d in link for d in {"linkedin.com", "facebook.com", "instagram.com", "twitter.com", "x.com", "github.com"}):
                            continue
                        mention = {"url": link, "title": r.get("title", ""), "snippet": content[:300]}
                        if mention not in found["mentions"]:
                            found["mentions"].append(mention)

            # ── 3. Recherche générale pure — extrait Facebook URL + description pro ─
            # Sans context : le curl test montre que "Nom" seul retourne 10 résultats
            # incluant Facebook, codeur.com, skills — alors que "Nom" + company filtre tout
            if serpapi_key:
                gen_results = await serpapi_search(client, serpapi_key, f'"{full_name}"', num=10)

                skill_set: set = set()
                for r in gen_results:
                    link    = r.get("link", "")
                    snippet = r.get("snippet", "")
                    title   = r.get("title", "")
                    combined = f"{title} {snippet}"

                    # Valider que le résultat mentionne vraiment la personne
                    if not _name_in_text(full_name, combined):
                        continue

                    # Facebook URL — si pas encore trouvée via facebook_searcher
                    if not found["facebook_url"] and "facebook.com" in link:
                        # Accepter uniquement les URLs de profil (pas photos/videos/posts/etc.)
                        if re.search(r'facebook\.com/(?:people/[^/]+/?|profile\.php\?id=\d+|[a-zA-Z0-9][a-zA-Z0-9.]+/?$)', link):
                            found["facebook_url"] = link
                            logger.info(f"General search found Facebook URL: {link}")

                    # Description professionnelle — snippet pertinent hors réseaux sociaux
                    if not found["professional_description"]:
                        if not any(d in link for d in _SOCIAL_DOMAINS):
                            if len(snippet.strip()) > 40:
                                found["professional_description"] = snippet.strip()[:400]

                    # Extraction de skills depuis les snippets
                    matches = _SKILL_PATTERNS.findall(combined)
                    for m in matches:
                        skill_set.add(m.lower().replace(" ", "_"))

                if skill_set:
                    found["skills"] = sorted(skill_set)

                # Mentions depuis la recherche générale (non-social, non-github)
                gen_mentions = extract_public_mentions(
                    gen_results, full_name=full_name,
                    skip_domains={"linkedin.com", "facebook.com", "instagram.com",
                                  "twitter.com", "x.com", "github.com"},
                )
                for m in gen_mentions:
                    if m not in found["mentions"]:
                        found["mentions"].append(m)

                # Fusionner skills depuis Jina Search
                jina_skill_set = found.pop("_skill_set", set())
                if jina_skill_set:
                    existing = set(found.get("skills") or [])
                    found["skills"] = sorted(existing | jina_skill_set)

        # ── 4. Stocker les URLs découvertes pour le content_extractor ────────
        # Le content_extractor lira discovered_urls + jina_prefetched_content.
        # jina_prefetched_content contient le contenu déjà fetché par s.jina.ai —
        # content_extractor l'utilisera directement sans re-fetcher via r.jina.ai.
        discovered_urls = {
            "linkedin":      found["linkedin_url"],
            "github":        found["github_url"],
            "personal_site": found["personal_site"],
            "facebook":      found["facebook_url"],
            "mentions":      found["mentions"],
        }
        if tool_context is not None:
            tool_context.state["discovered_urls"] = discovered_urls
            if jina_prefetched:
                tool_context.state["jina_prefetched_content"] = jina_prefetched
                logger.info(f"Jina Search: {len(jina_prefetched)} page(s) pré-fetchées pour content_extractor")

        # Score qualité
        filled = sum(1 for v in [found["github_url"], found["personal_site"], found["linkedin_url"], found["facebook_url"], found["professional_description"]] if v)
        filled += min(len(found["mentions"]), 3)
        quality = round(filled / 8, 2)

        result = {
            "status":                   "found" if filled > 0 else "not_found",
            "full_name":                full_name,
            "personal_site":            found["personal_site"],
            "linkedin_url":             found["linkedin_url"],
            "linkedin_snippet":         found["linkedin_snippet"],
            "linkedin_title":           found["linkedin_title"],
            "facebook_url":             found["facebook_url"],
            "professional_description": found["professional_description"],
            "skills":                   found["skills"] if found["skills"] else None,
            "inferred_company":         found["inferred_company"],
            "mentions":                 found["mentions"][:5],
            "data_quality_score":       quality,
            "provider":                 "github_api+jina_search+serpapi",
        }

        # Champs GitHub uniquement si profil trouvé — pas de null parasites
        if found["github_url"]:
            result["github_url"]          = found["github_url"]
            result["github_login"]        = found["github_login"]
            result["github_bio"]          = found["github_bio"]
            result["github_company"]      = found["github_company"]
            result["github_blog"]         = found["github_blog"]
            result["github_location"]     = found["github_location"]
            result["github_public_repos"] = found["github_public_repos"]
            result["github_followers"]    = found["github_followers"]

        logger.info(
            f"Web presence: "
            f"github={'OK' if found['github_url'] else '-'} "
            f"| linkedin={'OK' if found['linkedin_url'] else '-'} "
            f"| facebook={'OK' if found['facebook_url'] else '-'} "
            f"| site={'OK' if found['personal_site'] else '-'} "
            f"| skills={len(found['skills'])} "
            f"| mentions={len(found['mentions'])}"
        )

        # Écriture directe dans session.state — bypasse le reformatage LLM
        if tool_context is not None:
            tool_context.state["web_profile_data"] = result

        status = result.get("status", "not_found")
        return f"web_profile_data stored: status={status}, linkedin={'found' if found['linkedin_url'] else 'not_found'}, github={'found' if found['github_url'] else 'not_found'}"

    return FunctionTool(func=search_person_web_presence)
