"""
LinkedIn Search Tool - Waterfall: Apify (by name) → Gemini URL → SerpAPI → Gemini full.

Provider priority:
1. Apify (by name) (primary)    - harvestapi/linkedin-profile-search-by-name, no URL needed
2. Gemini URL-only (URL search) - Google Search grounding to discover URL if Apify fails
3. SerpAPI         (fallback)   - Google Search snippets for skills/mentions
4. GeminiSearch    (last resort)- Full extraction with Gemini (if no URL found at all)

Key principle:
- Apify searches LinkedIn directly by firstName + lastName — solves the URL discovery problem
- Gemini is used ONLY for URL discovery (step 2) if Apify also fails
- GeminiSearch full extraction (step 4) is an absolute last resort

URL hint flow:
- web_searcher stores LinkedIn URL in session.state["discovered_urls"]["linkedin"]
- linkedin_searcher reads this hint to verify the best match among Apify results

Environment variables required:
- APIFY_API_KEY      (primary — recommended)
- SERPAPI_KEY        (fallback)
- GOOGLE_GENAI_USE_VERTEXAI / GOOGLE_CLOUD_PROJECT (Gemini — already set for the pipeline)
"""

import asyncio
import httpx
import os
import logging
import re
from typing import Optional, Dict, Any, List
from google.adk.tools import FunctionTool, ToolContext
import json
from datetime import datetime

from ....tools.serpapi_utils import serpapi_search, serpapi_search_with_variants, serpapi_linkedin_search, extract_public_mentions
from ...web_searcher.tools.web_content_extractor import _get_genai_client

logger = logging.getLogger(__name__)


class LinkedInSearchProvider:
    """Base class for LinkedIn search providers."""

    def __init__(self):
        self.timeout = int(os.getenv("TIMEOUT_SECONDS", "30"))

    async def search(
        self,
        full_name: str,
        email: Optional[str] = None,
        company: Optional[str] = None,
        location: Optional[str] = None,
        search_type: str = "individual",
        linkedin_url: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Search for LinkedIn profile or company page. Must be implemented by subclass."""
        raise NotImplementedError


class ApifyLinkedInProvider(LinkedInSearchProvider):
    """
    Apify LinkedIn scraper — recherche par nom, pas besoin d'URL en entrée.

    Actor : harvestapi/linkedin-profile-search-by-name
    Input : {"firstName": "...", "lastName": "...", "scrapeType": "Full", "maxItems": 3}
    Output : linkedinUrl, experience (position/companyName), education (schoolName),
             skills (list de {name}), certifications, location (objet), connections, followers

    Avantage clé : fonctionne sans URL LinkedIn — résout le problème de découverte.
    Requiert APIFY_API_KEY dans l'environnement.
    """

    DEFAULT_ACTOR = "harvestapi~linkedin-profile-search-by-name"
    POLL_INTERVAL = 5
    POLL_TIMEOUT  = 120  # cet actor peut prendre plus de temps (ouvre chaque profil)

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("APIFY_API_KEY")
        if not self.api_key:
            raise ValueError("APIFY_API_KEY not found in environment")
        self.actor_id = os.getenv("APIFY_LINKEDIN_ACTOR_ID", self.DEFAULT_ACTOR)

    def _split_name(self, full_name: str):
        """
        Divise full_name en (firstName, lastName).
        Détecte les tokens ALL-CAPS comme nom de famille (convention malgasy/africaine).
        Fallback : dernier mot = nom de famille (convention occidentale).
        """
        words = full_name.strip().split()
        surnames = [w for w in words if w.isupper() and len(w) > 2]
        first_words = [w for w in words if not (w.isupper() and len(w) > 2)]
        if not surnames:
            surnames = [words[-1]] if words else []
            first_words = words[:-1] if len(words) > 1 else words
        return " ".join(first_words), " ".join(surnames)

    def _best_match(self, items: List[Dict], full_name: str, linkedin_url: Optional[str]) -> Optional[Dict]:
        """
        Choisit le profil le plus pertinent parmi les résultats Apify.
        Priorité : URL exacte → nom complet → premier résultat.
        """
        name_lower = full_name.lower()

        # 1. URL exacte si on en a une
        if linkedin_url:
            slug = re.search(r'linkedin\.com/in/([^/?]+)', linkedin_url or "")
            if slug:
                for item in items:
                    item_url = item.get("linkedinUrl") or ""
                    if slug.group(1) in item_url:
                        return item

        # 2. Correspondance nom complet
        for item in items:
            candidate = (
                f"{item.get('firstName', '')} {item.get('lastName', '')}".lower()
            )
            if name_lower in candidate or candidate in name_lower:
                return item

        # 3. Premier résultat par défaut
        return items[0] if items else None

    async def search(
        self,
        full_name: str,
        email: Optional[str] = None,
        company: Optional[str] = None,
        location: Optional[str] = None,
        search_type: str = "individual",
        linkedin_url: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:

        first_name, last_name = self._split_name(full_name)
        if not first_name or not last_name:
            logger.warning(f"Apify: impossible de diviser le nom '{full_name}'")
            return None

        # Normaliser en title case pour LinkedIn (qui stocke les noms en title case)
        # La détection ALL-CAPS a déjà servi pour le split ci-dessus
        first_name_norm = first_name.title()
        last_name_norm  = last_name.title()

        actor_input: Dict[str, Any] = {
            "firstName":          first_name_norm,
            "lastName":           last_name_norm,
            "profileScraperMode": "Full",
            "maxItems":           3,
        }
        if location:
            actor_input["locations"] = [location]

        logger.info(f"Apify LinkedIn search by name: firstName='{first_name_norm}' lastName='{last_name_norm}'")

        try:
            async with httpx.AsyncClient(timeout=self.POLL_TIMEOUT + 10) as client:
                # ── Démarrer le run ──────────────────────────────────────────
                run_resp = await client.post(
                    f"https://api.apify.com/v2/acts/{self.actor_id}/runs",
                    params={"token": self.api_key},
                    json=actor_input,
                )
                run_resp.raise_for_status()
                run_data   = run_resp.json()["data"]
                run_id     = run_data["id"]
                dataset_id = run_data["defaultDatasetId"]
                logger.info(f"Apify run started: {run_id}")

                # ── Polling ──────────────────────────────────────────────────
                waited = 0
                status = "RUNNING"
                while waited < self.POLL_TIMEOUT and status not in ("SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"):
                    await asyncio.sleep(self.POLL_INTERVAL)
                    waited += self.POLL_INTERVAL
                    st_resp = await client.get(
                        f"https://api.apify.com/v2/actor-runs/{run_id}",
                        params={"token": self.api_key},
                    )
                    status = st_resp.json()["data"]["status"]
                    logger.info(f"Apify run {run_id}: {status} ({waited}s)")

                if status != "SUCCEEDED":
                    logger.warning(f"Apify run ended: {status}")
                    return None

                # ── Récupérer le dataset ─────────────────────────────────────
                items_resp = await client.get(
                    f"https://api.apify.com/v2/datasets/{dataset_id}/items",
                    params={"token": self.api_key, "format": "json"},
                )
                items_resp.raise_for_status()
                items: List[Dict[str, Any]] = items_resp.json()

            if not items:
                logger.warning(f"Apify: 0 résultats pour {full_name}")
                return None

            best = self._best_match(items, full_name, linkedin_url)
            if not best:
                return None

            return self._parse_profile(best, full_name)

        except Exception as e:
            logger.error(f"Apify error for {full_name}: {e}")
            return None

    def _parse_profile(self, data: Dict[str, Any], full_name: str) -> Dict[str, Any]:
        """Parse la réponse harvestapi vers notre format standard."""
        profile_url = data.get("linkedinUrl") or ""

        # ── Experience ────────────────────────────────────────────────────────
        raw_exp = data.get("experience") or []
        experiences = []
        for exp in raw_exp[:6]:
            start = (exp.get("startDate") or {})
            end   = (exp.get("endDate") or {})
            experiences.append({
                "title":       exp.get("position") or exp.get("title"),
                "company":     exp.get("companyName") or exp.get("company"),
                "starts_at":   start.get("text") or start.get("year"),
                "ends_at":     end.get("text") or end.get("year"),
                "description": exp.get("description"),
            })

        # ── Education ─────────────────────────────────────────────────────────
        raw_edu = data.get("education") or []
        education = []
        for edu in raw_edu[:4]:
            start = (edu.get("startDate") or {})
            end   = (edu.get("endDate") or {})
            education.append({
                "school":         edu.get("schoolName") or edu.get("school"),
                "degree":         edu.get("degree"),
                "field_of_study": edu.get("fieldOfStudy"),
                "starts_at":      start.get("text") or start.get("year"),
                "ends_at":        end.get("text") or end.get("year"),
            })

        # ── Skills ────────────────────────────────────────────────────────────
        skills = []
        for s in (data.get("skills") or [])[:15]:
            name = s.get("name") if isinstance(s, dict) else s
            if name and isinstance(name, str):
                skills.append(name)

        # ── Certifications ────────────────────────────────────────────────────
        certifications = []
        for c in (data.get("certifications") or [])[:6]:
            title = c.get("title") or c.get("name")
            if title:
                certifications.append({
                    "title":     title,
                    "authority": c.get("issuedBy") or c.get("authority"),
                })

        # ── Poste actuel ──────────────────────────────────────────────────────
        current_pos_list = data.get("currentPosition") or []
        current_company = (
            current_pos_list[0].get("companyName") if current_pos_list
            else None
        )
        current_exp = next((e for e in experiences if not e.get("ends_at") or "present" in str(e.get("ends_at", "")).lower()), experiences[0] if experiences else {})
        current_title   = current_exp.get("title")
        current_company = current_company or current_exp.get("company")

        # ── Location ──────────────────────────────────────────────────────────
        loc = data.get("location") or {}
        location_str = (
            loc.get("linkedinText") if isinstance(loc, dict)
            else str(loc) if loc else None
        )

        logger.info(f"✅ Apify scraped {full_name}: {len(experiences)} exp, {len(skills)} skills")
        return {
            "profile_url":         profile_url,
            "headline":            data.get("headline"),
            "current_position":    current_title,
            "current_company":     current_company,
            "location":            location_str,
            "connections_count":   data.get("connectionsCount"),
            "followers":           data.get("followerCount"),
            "experience":          experiences,
            "education":           education,
            "skills":              skills,
            "certifications":      certifications,
            "summary":             data.get("about"),
            "profile_picture_url": data.get("photo"),
            "last_updated":        datetime.utcnow().isoformat(),
            "data_quality_score":  0.88,
            "provider":            "apify_linkedin",
        }


class ApifyLinkedInUrlProvider(LinkedInSearchProvider):
    """
    Apify LinkedIn scraper par URL directe.

    Actor : harvestapi~linkedin-profile-scraper
    Input : {"profileUrls": ["https://www.linkedin.com/in/..."]}
    Utilisé quand l'URL LinkedIn est déjà connue (via web_searcher).
    Plus précis que la recherche par nom — 0 risque de mauvais match.
    """

    ACTOR_ID     = "harvestapi~linkedin-profile-scraper"
    POLL_INTERVAL = 5
    POLL_TIMEOUT  = 120

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("APIFY_API_KEY")
        if not self.api_key:
            raise ValueError("APIFY_API_KEY not found in environment")

    async def search(
        self,
        full_name: str,
        linkedin_url: str,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        if not linkedin_url:
            return None

        logger.info(f"Apify URL scraper: {linkedin_url} for {full_name}")

        try:
            async with httpx.AsyncClient(timeout=self.POLL_TIMEOUT + 10) as client:
                run_resp = await client.post(
                    f"https://api.apify.com/v2/acts/{self.ACTOR_ID}/runs",
                    params={"token": self.api_key},
                    json={"profileUrls": [linkedin_url]},
                )
                run_resp.raise_for_status()
                run_data   = run_resp.json()["data"]
                run_id     = run_data["id"]
                dataset_id = run_data["defaultDatasetId"]
                logger.info(f"Apify URL run started: {run_id}")

                waited = 0
                status = "RUNNING"
                while waited < self.POLL_TIMEOUT and status not in ("SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"):
                    await asyncio.sleep(self.POLL_INTERVAL)
                    waited += self.POLL_INTERVAL
                    st_resp = await client.get(
                        f"https://api.apify.com/v2/actor-runs/{run_id}",
                        params={"token": self.api_key},
                    )
                    status = st_resp.json()["data"]["status"]

                if status != "SUCCEEDED":
                    logger.warning(f"Apify URL run ended: {status}")
                    return None

                items_resp = await client.get(
                    f"https://api.apify.com/v2/datasets/{dataset_id}/items",
                    params={"token": self.api_key, "format": "json"},
                )
                items_resp.raise_for_status()
                items: List[Dict[str, Any]] = items_resp.json()

            if not items:
                logger.warning(f"Apify URL: 0 résultats pour {linkedin_url}")
                return None

            # Un seul profil attendu puisqu'on donne une URL exacte
            return ApifyLinkedInProvider._parse_profile(self, items[0], full_name)

        except Exception as e:
            logger.error(f"Apify URL error for {linkedin_url}: {e}")
            return None


class SerpAPIProvider(LinkedInSearchProvider):
    """
    SerpAPI provider for LinkedIn search via Google Search.

    Runs two queries:
    1. site:linkedin.com/in/ — direct profile page
    2. General Google search — public mentions, press, company pages
    """

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("SERPAPI_KEY")
        if not self.api_key:
            raise ValueError("SERPAPI_KEY not found in environment")

    async def search(
        self,
        full_name: str,
        email: Optional[str] = None,
        company: Optional[str] = None,
        location: Optional[str] = None,
        search_type: str = "individual",
        linkedin_url: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Search using SerpAPI — LinkedIn profile/company page + public web mentions."""

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:

                # ── Query 1: LinkedIn profile or company page ──────────────
                if search_type == "company":
                    # Try name variants to handle typos / long names
                    li_data = await serpapi_search_with_variants(
                        client, self.api_key,
                        query_prefix="site:linkedin.com/company/",
                        name=full_name,
                        num=5,
                    )
                else:
                    # Priorité : si on a déjà le slug, chercher directement par slug
                    # (plus fiable que par nom — évite les ambiguïtés)
                    slug_m = re.search(r'linkedin\.com/in/([^/?]+)', linkedin_url or "")
                    if slug_m:
                        slug = slug_m.group(1)
                        li_query = f'site:linkedin.com/in/{slug}'
                        logger.info(f"SerpAPI LinkedIn query (by slug): {li_query}")
                        li_data = await serpapi_search(client, self.api_key, li_query, num=5)
                    else:
                        # Pas de slug connu → essayer toutes les variantes du nom
                        # (prénom composé, ordre inversé, nom court…)
                        logger.info(f"SerpAPI LinkedIn: trying name variants for {full_name}")
                        li_data = await serpapi_linkedin_search(
                            client, self.api_key, full_name,
                            location=location or "",
                            company=company or "",
                        )

                # ── Query 2: Public web mentions ───────────────────────────
                web_query = f'"{full_name}"'
                if company:
                    web_query += f' "{company}"'
                if location:
                    web_query += f" {location}"

                logger.info(f"SerpAPI web query: {web_query}")
                web_data = await serpapi_search(client, self.api_key, web_query, num=10)

            profile    = self._parse_linkedin_results(li_data, full_name)
            web_extras = extract_public_mentions(
                web_data, full_name,
                skip_domains={"linkedin.com", "facebook.com", "instagram.com", "twitter.com", "x.com"},
            )

            if not profile.get("profile_url") and not web_extras:
                logger.warning(f"No results found via SerpAPI for {full_name}")
                return None

            return {
                "profile_url":      profile.get("profile_url"),
                "headline":         profile.get("headline"),
                "current_position": profile.get("current_position"),
                "current_company":  profile.get("current_company") or company,
                "location":         profile.get("location") or location,
                "industry":         profile.get("industry"),
                "connections_count": None,
                "experience":       profile.get("experience", []),
                "education":        profile.get("education", []),
                "skills":           profile.get("skills", []),
                "summary":          profile.get("summary"),
                "web_mentions":     web_extras,
                "profile_picture_url": None,
                "last_updated":     datetime.utcnow().isoformat(),
                "data_quality_score": self._score(profile, web_extras),
                "provider":         "serpapi",
            }

        except Exception as e:
            logger.error(f"SerpAPI error: {e}")
            return None

    def _parse_linkedin_results(self, results: list, full_name: str) -> Dict[str, Any]:
        """Extract structured info from LinkedIn search result snippets (individual or company)."""
        for r in results:
            link = r.get("link", "")
            is_company_page = "linkedin.com/company/" in link
            if "linkedin.com/in/" not in link and not is_company_page:
                continue

            # ── Company page ──────────────────────────────────────────────
            if is_company_page:
                snippet = r.get("snippet", "")
                title = r.get("title", "")
                company_name = re.sub(r'\s*\|.*LinkedIn.*$', '', title, flags=re.IGNORECASE).strip()
                logger.info(f"✅ LinkedIn company page found via SerpAPI: {link}")
                return {
                    "type": "company_page",
                    "profile_url": link,
                    "company_name": company_name or full_name,
                    "description": snippet[:500],
                    "data_quality_score": self._score({}, []),
                    "provider": "serpapi",
                }

            snippet = r.get("snippet", "")
            title   = r.get("title", "")

            # ── Headline depuis le titre ─────────────────────────────────────
            # Format Google : "Name - Position | Company | LinkedIn"
            headline = re.sub(r'\s*\|[^|]*LinkedIn[^|]*$', '', title, flags=re.IGNORECASE).strip()
            headline = re.sub(rf'^{re.escape(full_name)}\s*[-–]\s*', '', headline).strip()
            # Nettoyer le suffixe "| Company" résiduel si le headline contient company
            headline_clean = re.sub(r'\s*\|.*$', '', headline).strip()

            # ── current_position / current_company ──────────────────────────
            # Priorité 1 : "Position at/chez Company" dans le headline
            pos_m = re.search(
                r'^(.+?)\s+(?:at|@|chez|à)\s+(.+?)(?:\s*[|·]|$)',
                headline, re.IGNORECASE
            )
            # Priorité 2 : "Position | Company" dans le headline
            pipe_m = re.search(r'^(.+?)\s*\|\s*(.+)$', headline) if not pos_m else None

            if pos_m:
                current_position = pos_m.group(1).strip()
                current_company  = pos_m.group(2).strip()
            elif pipe_m:
                current_position = pipe_m.group(1).strip()
                current_company  = pipe_m.group(2).strip()
            else:
                current_position = headline_clean or None
                current_company  = None

            # Priorité 3 : extraire company depuis le snippet Google
            # Format Google : "Name. Employer School. Location. followers..."
            # → La 1ère phrase après le nom contient souvent l'employeur
            if not current_company:
                first_sentence_m = re.match(
                    rf'{re.escape(full_name)}[.\s]+([^.]+)',
                    snippet, re.IGNORECASE
                )
                if first_sentence_m:
                    first_sent = first_sentence_m.group(1).strip()
                    # Séparer employer / school dans la 1ère phrase
                    school_kw = r'(?:Université|University|Institut|Institute|Ecole|École|School|College|Collège|Academy|ENI|IUT|BTS|MBA|HEC|ESSEC|Polytechnique|Sciences\s*Po)'
                    school_split = re.split(school_kw, first_sent, maxsplit=1, flags=re.IGNORECASE)
                    if school_split:
                        employer_part = school_split[0].strip().rstrip(',')
                        if employer_part and len(employer_part) < 60:
                            current_company = employer_part

            # ── Education depuis le snippet ──────────────────────────────────
            # Google snippet format : "Name. Employer School. Location."
            # → chercher les patterns "École/University/Institut..." dans le snippet
            education = []
            school_pattern = re.compile(
                r'([A-ZÀÂÉÈÊËÎÏÔÙÛÜÇ][^.]+?'
                r'(?:Université|University|Institut(?:e)?|Ecole|École|School|College|Collège|Academy|ENI|IUT|BTS|MBA|HEC|Polytechnique|Sciences\s*Po)'
                r'[^.]{0,60})',
                re.IGNORECASE
            )
            for m in school_pattern.finditer(snippet):
                school_name = m.group(1).strip().rstrip(',; ')
                # Supprimer le préfixe company si présent (Google colle employer + school)
                if current_company and school_name.lower().startswith(current_company.lower()):
                    school_name = school_name[len(current_company):].strip()
                if len(school_name) > 5 and school_name not in [e.get("school") for e in education]:
                    education.append({"school": school_name, "degree": None, "field_of_study": None})

            # Expérience depuis snippet (format "2020 - Present · Role at Company")
            experience = []
            for exp in re.finditer(
                r'(\d{4})\s*[-–]\s*(?:\d{4}|Present|Présent)\s*·?\s*([^·\n]{5,80})',
                snippet
            ):
                experience.append({"period": exp.group(1), "role": exp.group(2).strip()})

            # ── Location ─────────────────────────────────────────────────────
            # \.\s retiré — trop large, capture "Name. Company School..." comme location
            loc_m = re.search(
                r'(?:Location|Based in|·)'
                r'\s*([A-ZÀÂÉÈÊËÎÏÔÙÛÜÇ][^·\n,]{2,}(?:,\s*[A-Z][^·\n,]{2,})?)',
                snippet
            )
            if not loc_m:
                # Fallback: 3ème phrase du snippet souvent = location
                sentences = [s.strip() for s in snippet.split('.') if s.strip()]
                if len(sentences) >= 3:
                    candidate = sentences[2]
                    if len(candidate) < 60 and re.search(r'[A-Z]', candidate):
                        loc_m = None
                        location = candidate
                    else:
                        location = None
                else:
                    location = None
            else:
                location = loc_m.group(1).strip()

            # ── Skills depuis les items séparés par | ────────────────────────
            # Extraire uniquement depuis les parties pipe-séparées (pas toute la phrase)
            skills_raw = []
            for seg in re.split(r'\s*[|·]\s*', snippet):
                seg = seg.strip()
                if 1 < len(seg) <= 30 and not re.search(r'[,\.\?\!]', seg):
                    if re.search(r'[A-Za-z]', seg) and len(seg.split()) <= 4:
                        skills_raw.append(seg)
            # Ajouter les skills du headline si format "Pos | Skill1 | Skill2"
            for seg in re.split(r'\s*\|\s*', headline):
                seg = seg.strip()
                if 1 < len(seg) <= 25 and not re.search(r'[,\.\?\!]', seg):
                    if re.search(r'[A-Za-z]', seg) and len(seg.split()) <= 3:
                        skills_raw.append(seg)
            # Déduplication + filtre noms/mots parasites
            stop = {full_name.lower(), "linkedin", "profil", "profile",
                    (current_company or "").lower(), (current_position or "").lower()}
            seen: set = set()
            skills = []
            for s in skills_raw:
                key = s.lower()
                if key in seen or key in stop or len(key) <= 1:
                    continue
                # Filtre labels parasites : "Location: ...", "Experience: ...", etc.
                if re.match(r'^(location|experience|education|followers?|connections?)\s*:', s, re.IGNORECASE):
                    continue
                seen.add(key)
                skills.append(s)
            skills = skills[:12]

            # Si position == company (headline était juste le nom de l'entreprise),
            # c'est un étudiant/stagiaire sans titre explicite — garder seulement company
            if current_position and current_company:
                if current_position.lower().strip() == current_company.lower().strip():
                    current_position = None

            logger.info(f"LinkedIn profile found via SerpAPI: {link}")
            return {
                "profile_url":      link,
                "headline":         headline_clean or headline or None,
                "current_position": current_position,
                "current_company":  current_company,
                "location":         location,
                "industry":         None,
                "experience":       experience,
                "education":        education,
                "skills":           skills,
                "summary":          snippet[:500],
                "data_quality_score": self._score(
                    {"profile_url": link, "headline": headline_clean,
                     "current_position": current_position, "current_company": current_company,
                     "location": location, "experience": experience, "skills": skills},
                    []
                ),
                "provider": "serpapi",
            }

        return {}

    def _score(self, profile: Dict, web_mentions: list) -> float:
        score = 0.1
        if profile.get("profile_url"):      score += 0.30
        if profile.get("headline"):         score += 0.10
        if profile.get("current_position"): score += 0.10
        if profile.get("current_company"):  score += 0.05
        if profile.get("location"):         score += 0.05
        if profile.get("experience"):       score += 0.10
        if profile.get("skills"):           score += 0.05
        score += min(len(web_mentions) * 0.03, 0.15)
        return round(min(score, 1.0), 2)


def _extract_experience_from_web_mentions(mentions: list) -> list:
    """
    Extrait les entrées d'expérience depuis les snippets web_mentions.
    Cherche uniquement le format structuré "2020 - Present · Role at Company".
    Le format "Experience: Company" est ignoré car Google affiche souvent
    le nom de la company seul (ex: "ClearMind-Analytics") au lieu du poste réel.
    """
    experience = []
    for mention in mentions:
        snippet = mention.get("snippet", "")
        # Format "2020 - Present · Role at Company" — uniquement si contient " at " ou " · "
        for m in re.finditer(
            r'(\d{4})\s*[-–]\s*(?:\d{4}|Present|Présent)\s*·?\s*([^·\n]{5,80})',
            snippet
        ):
            role = m.group(2).strip()
            # Exclure si le role ressemble à une company seule (pas d'espace = 1 mot)
            if " " in role or " at " in role.lower():
                experience.append({"period": m.group(1), "role": role})
    # Dédupliquer par rôle
    seen = set()
    deduped = []
    for e in experience:
        key = e["role"].lower()
        if key not in seen:
            seen.add(key)
            deduped.append(e)
    return deduped[:5]


def _extract_skills_from_web_mentions(mentions: list, full_name: str) -> list:
    """
    Extrait les skills depuis les snippets des web_mentions (format pipe-séparé).
    Utilisé comme dernier recours quand aucun provider n'a retourné de skills.
    Ex: "Développeur Full stack | Angular | React | C# | Node js | Java Antananarivo..."
    """
    skills_raw = []
    for mention in mentions:
        snippet = mention.get("snippet", "")
        if "|" not in snippet:
            continue
        for seg in re.split(r'\s*\|\s*', snippet):
            seg = seg.strip()
            if 1 < len(seg) <= 30 and not re.search(r'[,\.\?\!\d]', seg):
                if re.search(r'[A-Za-z]', seg) and len(seg.split()) <= 4:
                    skills_raw.append(seg)

    name_parts = {p.lower() for p in full_name.split()}
    stop = name_parts | {"linkedin", "profil", "profile", "more", "and", "in", "view", "followers", "connections"}
    seen: set = set()
    skills = []
    for s in skills_raw:
        key = s.lower()
        # Filtrer les segments qui contiennent des mots du nom ou trop longs
        if key not in seen and key not in stop and len(key) > 1:
            # Exclure "Location: ...", "Experience: ...", etc. (labels Google)
            if re.match(r'^(location|experience|education|followers?|connections?)\s*:', key, re.IGNORECASE):
                continue
            # Exclure les segments qui ressemblent à des lieux (Antananarivo, Fianarantsoa...)
            if not any(p in key for p in name_parts):
                seen.add(key)
                skills.append(s)
    return skills[:12]


def _is_incomplete(result: Optional[Dict]) -> bool:
    """
    Retourne True si le résultat manque de skills OU d'experience.
    Education seule ne suffit pas — on veut toujours enrichir skills + experience.
    """
    if not result or result.get("status") in ("not_found", "error"):
        return True
    return not result.get("skills") or not result.get("experience")


def _merge_results(primary: Dict, fallback: Dict) -> Dict:
    """
    Fusionne deux résultats LinkedIn (primary gagne sur les champs scalaires).
    Le fallback complète les tableaux vides (experience, education, skills).
    """
    merged = dict(primary)
    for key in ("experience", "education", "skills", "web_mentions"):
        if not merged.get(key) and fallback.get(key):
            merged[key] = fallback[key]
    # Compléter headline si absent
    if not merged.get("headline") and fallback.get("headline"):
        merged["headline"] = fallback["headline"]
    if not merged.get("summary") and fallback.get("summary"):
        merged["summary"] = fallback["summary"]
    if not merged.get("location") and fallback.get("location"):
        merged["location"] = fallback["location"]
    # Recalculer le score
    merged["data_quality_score"] = max(
        merged.get("data_quality_score", 0.0),
        fallback.get("data_quality_score", 0.0),
    )
    merged["provider"] = f"{primary.get('provider', 'unknown')}+{fallback.get('provider', 'fallback')}"
    return merged


class GoogleCSEProvider(LinkedInSearchProvider):
    """
    Google Custom Search Engine — LinkedIn enrichment via pagemap.

    Retourne title + snippet + pagemap (og:description, metatags, person schema)
    indexés par Google, puis Gemini extrait les données structurées.
    Même approche que n8n Google Search → Gemini.
    """

    BASE_URL = "https://www.googleapis.com/customsearch/v1"

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("GOOGLE_CSE_KEY", "")
        self.cx      = os.getenv("GOOGLE_CSE_CX", "")
        if not self.api_key or not self.cx:
            raise ValueError("GOOGLE_CSE_KEY or GOOGLE_CSE_CX not set")

    async def search(
        self,
        full_name: str,
        email: Optional[str] = None,
        company: Optional[str] = None,
        location: Optional[str] = None,
        search_type: str = "individual",
        linkedin_url: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            slug_m = re.search(r'linkedin\.com/in/([^/?]+)', linkedin_url or "")
            query  = f'site:linkedin.com/in/{slug_m.group(1)}' if slug_m else f'"{full_name}" site:linkedin.com/in'
            logger.info(f"Google CSE query: {query}")

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(
                    self.BASE_URL,
                    params={"key": self.api_key, "cx": self.cx, "q": query, "num": 3},
                )

            if resp.status_code != 200:
                logger.warning(f"Google CSE HTTP {resp.status_code}: {resp.text[:200]}")
                return None

            items = resp.json().get("items", [])
            if not items:
                return None

            # Combiner title + snippet + pagemap pour chaque résultat LinkedIn
            content_parts = []
            profile_url_found = None

            for item in items:
                link = item.get("link", "")
                if "linkedin.com/in/" not in link:
                    continue
                if not profile_url_found:
                    profile_url_found = link

                title    = item.get("title", "")
                snippet  = item.get("snippet", "")
                pagemap  = item.get("pagemap", {})
                metatags = (pagemap.get("metatags") or [{}])[0]
                person   = (pagemap.get("person") or [{}])[0]

                part = f"Profile URL: {link}\nTitle: {title}\nSnippet: {snippet}"
                og_title = metatags.get("og:title") or metatags.get("twitter:title", "")
                og_desc  = metatags.get("og:description") or metatags.get("twitter:description", "")
                if og_title and og_title != title:
                    part += f"\nOG Title: {og_title}"
                if og_desc:
                    part += f"\nDescription: {og_desc}"
                if person.get("jobtitle"):
                    part += f"\nJob Title: {person['jobtitle']}"
                if person.get("name"):
                    part += f"\nName: {person['name']}"
                content_parts.append(part)

            if not profile_url_found or not content_parts:
                return None

            combined = "\n\n".join(content_parts)
            logger.info(f"Google CSE: {len(content_parts)} result(s), {len(combined)} chars")

            extracted = await asyncio.to_thread(
                _gemini_extract_sync, combined, full_name
            )

            if not extracted:
                return None

            extracted["profile_url"]        = profile_url_found
            extracted["provider"]           = "google_cse"
            extracted["data_quality_score"] = 0.60
            return extracted

        except Exception as e:
            logger.error(f"Google CSE error for {full_name}: {e}")
            return None


class GeminiSearchProvider(LinkedInSearchProvider):
    """
    Gemini + Google Search grounding pour LinkedIn.

    Deux modes :
    - find_url()   → recherche uniquement l'URL du profil LinkedIn (rapide, fiable)
    - search()     → extraction complète (dernier recours si aucune URL trouvée)

    Utilise le SDK google.genai directement (pas ADK) pour contourner
    la limitation "single tool per agent" de l'ADK google_search tool.
    """

    def _find_url_sync(self, full_name: str, location: Optional[str]) -> Optional[str]:
        """
        Appel Gemini + Google Search grounding — retourne UNIQUEMENT l'URL LinkedIn.
        Plus rapide et moins sujet aux hallucinations qu'une extraction complète.
        """
        try:
            import google.genai as google_genai
            from google.genai import types as genai_types

            client = _get_genai_client()
            loc_hint = f" in {location}" if location else ""

            prompt = f"""Search Google for the LinkedIn profile of "{full_name}"{loc_hint}.

Return ONLY a valid JSON object with one field:
{{"profile_url": "https://www.linkedin.com/in/..."}}

Use null if not found. Return ONLY the JSON. No explanation."""

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
                ),
            )

            text = (response.text or "").strip()
            match = re.search(r'\{[\s\S]*\}', text)
            if not match:
                logger.info(f"Gemini URL search: no JSON for {full_name}")
                return None

            parsed = json.loads(match.group(0))
            url = parsed.get("profile_url")
            if url and "linkedin.com/in/" in str(url):
                logger.info(f"Gemini URL search found: {url}")
                return str(url)

            logger.info(f"Gemini URL search: no valid LinkedIn URL for {full_name}")
            return None

        except Exception as e:
            logger.error(f"Gemini URL search error for {full_name}: {e}")
            return None

    async def find_url(self, full_name: str, location: Optional[str] = None) -> Optional[str]:
        """Recherche uniquement l'URL LinkedIn (async wrapper)."""
        return await asyncio.to_thread(self._find_url_sync, full_name, location)

    def _extract_sync(self, full_name: str, linkedin_url: Optional[str], location: Optional[str]) -> Optional[Dict]:
        """Appel synchrone Gemini + Google Search grounding."""
        try:
            import google.genai as google_genai
            from google.genai import types as genai_types

            client = _get_genai_client()

            slug_m = re.search(r'linkedin\.com/in/([^/?]+)', linkedin_url or "")
            url_hint = f"\nLinkedIn URL: {linkedin_url}" if linkedin_url else ""
            loc_hint = f"\nLocation: {location}" if location else ""

            prompt = f"""Search Google for the LinkedIn profile of "{full_name}" and extract their professional information.{url_hint}{loc_hint}

Return ONLY a valid JSON object. Use null for missing fields. Use [] for empty lists.

{{
  "profile_url": "LinkedIn profile URL found (linkedin.com/in/...)",
  "current_position": "job title only — NOT company name",
  "current_company": "company or organization name",
  "location": "city, country",
  "summary": "professional summary in 1-2 sentences",
  "skills": ["technical skill or tool"],
  "experience": [{{"title": "job title", "company": "company name", "duration": "2020-2023"}}],
  "education": [{{"school": "school name", "degree": "degree type", "year": "graduation year"}}]
}}

STRICT RULES:
1. current_position = job title ONLY. NEVER put a company name as current_position.
2. skills = technical skills only (languages, tools, frameworks). No location labels.
3. Only extract data clearly referring to {full_name}.
4. Return ONLY the JSON. No markdown, no explanation."""

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
                ),
            )

            text = (response.text or "").strip()
            match = re.search(r'\{[\s\S]*\}', text)
            if not match:
                logger.info(f"Gemini Search: no JSON found for {full_name}")
                return None

            parsed = json.loads(match.group(0))

            # Validation et nettoyage
            result = {}
            for k, v in parsed.items():
                if v in (None, [], ""):
                    continue
                if k == "skills" and isinstance(v, list):
                    v = [s for s in v
                         if isinstance(s, str) and len(s.strip()) > 1
                         and not re.match(r'^(location|experience|education|followers?|connections?)\s*:', s, re.IGNORECASE)
                         and len(s.split()) <= 5]
                    if not v:
                        continue
                if k == "current_position" and isinstance(v, str):
                    company = parsed.get("current_company", "") or ""
                    if v.strip().lower() == company.strip().lower():
                        continue
                result[k] = v

            profile_url = result.get("profile_url") or linkedin_url
            if not profile_url:
                logger.info(f"Gemini Search: no profile URL for {full_name}")
                return None

            result["profile_url"]        = profile_url
            result["provider"]           = "gemini_search"
            result["data_quality_score"] = 0.65
            logger.info(
                f"Gemini Search found profile for {full_name}: "
                f"{result.get('current_position')} @ {result.get('current_company')}"
            )
            return result

        except Exception as e:
            logger.error(f"Gemini Search extraction error for {full_name}: {e}")
            return None

    async def search(
        self,
        full_name: str,
        email: Optional[str] = None,
        company: Optional[str] = None,
        location: Optional[str] = None,
        search_type: str = "individual",
        linkedin_url: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        logger.info(f"Gemini Search grounding for {full_name}")
        return await asyncio.to_thread(self._extract_sync, full_name, linkedin_url, location)


class LinkedInSearchTool:
    """
    LinkedIn search tool — waterfall Python pur.

    Stratégie :
    1. Apify (by name)    → recherche LinkedIn par nom, retourne profil complet + URL (score 0.88)
    2. Gemini URL-only    → Google Search grounding pour trouver l'URL si Apify échoue
    3. SerpAPI            → complément snippet/mentions (score 0.4-0.7)
    4. GeminiSearch full  → dernier recours absolu (extraction + URL, score 0.65)

    Apify (harvestapi/linkedin-profile-search-by-name) ne nécessite pas d'URL en entrée —
    il résout le problème de découverte d'URL pour les noms exotiques/composés.
    """

    def __init__(self):
        self.apify         = self._get_apify()
        self.serpapi       = self._get_serpapi()
        self.gemini_search = GeminiSearchProvider()

    def _get_apify(self) -> Optional[ApifyLinkedInProvider]:
        """Apify LinkedIn scraper par nom — actif seulement si APIFY_API_KEY est défini."""
        if os.getenv("APIFY_API_KEY"):
            try:
                return ApifyLinkedInProvider()
            except ValueError:
                pass
        return None

    def _get_apify_url(self) -> Optional[ApifyLinkedInUrlProvider]:
        """Apify LinkedIn scraper par URL — actif seulement si APIFY_API_KEY est défini."""
        if os.getenv("APIFY_API_KEY"):
            try:
                return ApifyLinkedInUrlProvider()
            except ValueError:
                pass
        return None

    def _get_serpapi(self) -> Optional[SerpAPIProvider]:
        """SerpAPI fallback de complétion (snippet)."""
        if os.getenv("SERPAPI_KEY"):
            try:
                return SerpAPIProvider()
            except ValueError:
                pass
        return None

    async def search(
        self,
        full_name: str,
        email: Optional[str] = None,
        company: Optional[str] = None,
        location: Optional[str] = None,
        search_type: str = "individual",
        linkedin_url: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:

        if not self.apify and not self.serpapi:
            return {
                "status": "error",
                "reason": "No LinkedIn provider configured. Set APIFY_API_KEY or SERPAPI_KEY.",
                "data_quality_score": 0.0,
            }

        kwargs = dict(
            full_name=full_name, email=email, company=company,
            location=location, search_type=search_type, linkedin_url=linkedin_url,
        )

        try:
            result = None

            # ── Company search : SerpAPI directement ─────────────────────────
            if search_type == "company":
                if self.serpapi:
                    result = await self.serpapi.search(**kwargs)
                if result and result.get("status") not in ("not_found", "error", None):
                    logger.info(f"LinkedIn company found for {full_name} via SerpAPI")
                    return result
                logger.warning(f"No LinkedIn company page found for {full_name}")
                return {
                    "status": "not_found",
                    "reason": f"No LinkedIn company page found for {full_name}",
                    "data_quality_score": 0.0,
                }

            if search_type == "individual":
                profile_url = linkedin_url or ""

                # ── Apify : provider principal (recherche par nom) ────────────
                # harvestapi~linkedin-profile-search-by-name cherche directement
                # par firstName + lastName et retourne le profil complet + l'URL.
                if self.apify:
                    logger.info(f"LinkedIn: Apify search by name for {full_name}")
                    apify_result = await self.apify.search(
                        full_name=full_name, linkedin_url=profile_url,
                        company=company, location=location,
                    )
                    if apify_result and apify_result.get("profile_url"):
                        result = apify_result
                        profile_url = result.get("profile_url") or profile_url
                        logger.info(f"Apify found profile for {full_name}: {profile_url}")

                # ── GeminiSearch : URL discovery si Apify aussi échoue ────────
                if _is_incomplete(result) and not profile_url:
                    logger.info(f"LinkedIn: Gemini URL discovery for {full_name}")
                    found_url = await self.gemini_search.find_url(full_name, location)
                    if found_url:
                        profile_url = found_url
                        logger.info(f"Gemini found URL: {profile_url}")

                # ── SerpAPI (complète les données manquantes + mentions) ───────
                if _is_incomplete(result) and self.serpapi:
                    logger.info(f"Still incomplete — trying SerpAPI for {full_name}")
                    serp_kwargs = dict(kwargs)
                    if profile_url and not serp_kwargs.get("linkedin_url"):
                        serp_kwargs["linkedin_url"] = profile_url
                    serp = await self.serpapi.search(**serp_kwargs)
                    if serp and serp.get("status") not in ("not_found", "error"):
                        result = _merge_results(result, serp) if result and result.get("profile_url") else serp
                        logger.info(f"Merged with SerpAPI for {full_name}")

                # ── GeminiSearch full (dernier recours absolu) ────────────────
                if _is_incomplete(result) and not (result or {}).get("profile_url"):
                    logger.info(f"GeminiSearch full extraction (last resort) for {full_name}")
                    gemini_result = await self.gemini_search.search(
                        full_name=full_name, linkedin_url=profile_url, location=location,
                    )
                    if gemini_result and gemini_result.get("profile_url"):
                        result = _merge_results(result, gemini_result) if result and result.get("profile_url") else gemini_result
                        logger.info(f"Gemini full enrichment done for {full_name}")

            if result and result.get("status") not in ("not_found", "error", None):
                logger.info(f"LinkedIn found for {full_name} via {result.get('provider', '?')}")
                return result

            if result and result.get("profile_url"):
                # Profile partiel (URL trouvée mais peu de données)
                logger.info(f"LinkedIn partial match for {full_name}")
                return result

            logger.warning(f"No LinkedIn profile found for {full_name}")
            return {
                "status": "not_found",
                "reason": f"No LinkedIn profile found for {full_name}",
                "data_quality_score": 0.0,
            }

        except Exception as e:
            logger.error(f"LinkedIn search error for {full_name}: {e}")
            return {
                "status": "error",
                "reason": str(e),
                "data_quality_score": 0.0,
            }


def _extract_linkedin_url_from_web_data(web_raw: Any) -> Optional[str]:
    """
    Extrait un URL LinkedIn depuis web_profile_data (session.state).
    Cherche d'abord linkedin_url direct, puis dans les mentions.
    """
    mention = _extract_linkedin_mention_from_web_data(web_raw)
    return mention.get("url") if mention else None


def _extract_linkedin_mention_from_web_data(web_raw: Any) -> Optional[Dict[str, Any]]:
    """
    Retourne le dict de la mention LinkedIn depuis web_profile_data.
    Inclut url, snippet, title — utilisé pour extraire skills sans appel API.
    """
    try:
        data = json.loads(web_raw) if isinstance(web_raw, str) else web_raw
        if not isinstance(data, dict):
            return None

        # Priorité 1 : linkedin_url + snippet/title sauvegardés par web_searcher
        direct_url = data.get("linkedin_url")
        if direct_url and "linkedin.com/in/" in direct_url:
            return {
                "url":     direct_url,
                "snippet": data.get("linkedin_snippet", "") or "",
                "title":   data.get("linkedin_title", "") or "",
            }

        # Priorité 2 : scan des mentions (cas edge où linkedin.com/in/ serait dans mentions)
        for mention in data.get("mentions", []):
            url = mention.get("url", "")
            if "linkedin.com/in/" in url:
                return {
                    "url":     url,
                    "snippet": mention.get("snippet", ""),
                    "title":   mention.get("title", ""),
                }
    except Exception as e:
        logger.debug(f"Could not parse web_profile_data for LinkedIn mention: {e}")

    return None


def get_linkedin_search_tool() -> FunctionTool:
    """
    Create ADK function tool for LinkedIn profile search.

    Le fallback LinkedIn (Gemini URL → SerpAPI → GeminiSearch) est géré en Python
    via ToolContext — plus fiable que de le déléguer au LLM.
    """

    async def search_linkedin_profile(
        full_name: str,
        email: str = "",
        company: str = "",
        location: str = "",
        search_type: str = "individual",
        linkedin_url: str = "",
        tool_context: ToolContext = None,  # ADK injecte automatiquement, invisible pour le LLM
    ) -> str:
        """
        Search for a LinkedIn profile (individual) or company page and return enriched information.

        Uses Apify (harvestapi) as primary provider. For individuals, if incomplete,
        the tool automatically falls through Gemini URL → SerpAPI → GeminiSearch full
        — no need to handle this fallback in the prompt.

        Args:
            full_name:    Full name of the person OR company name (required)
            email:        Email address for more accurate matching (optional)
            company:      Company name or website domain (optional)
            location:     Geographic location (city, country) (optional)
            search_type:  "individual" or "company"
            linkedin_url: Known LinkedIn profile URL (optional, skips name/email matching)

        Returns:
            JSON string with LinkedIn profile/company data or error message
        """
        # ── Upfront URL hint depuis session.state ────────────────────────
        # Si le LLM n'a pas passé de linkedin_url, on récupère celle trouvée
        # par web_searcher — directement dans discovered_urls ou web_profile_data.
        # Cela aide Apify _best_match() à choisir le bon profil parmi les résultats.
        if not linkedin_url and search_type == "individual" and tool_context is not None:
            # Priorité 1 : discovered_urls (écrit directement par person_web_search.py)
            discovered = tool_context.state.get("discovered_urls") or {}
            hint = discovered.get("linkedin") or ""
            # Priorité 2 : web_profile_data (fallback)
            if not hint:
                web_raw = tool_context.state.get("web_profile_data")
                hint = _extract_linkedin_url_from_web_data(web_raw) if web_raw else ""
            if hint:
                linkedin_url = hint
                logger.info(f"LinkedIn URL hint from session.state: {linkedin_url}")

        logger.info(
            f"Searching LinkedIn for: {full_name}"
            + (f" (url hint: {linkedin_url})" if linkedin_url else "")
        )

        tool = LinkedInSearchTool()

        async def _run(li_url: str = "") -> Optional[Dict]:
            return await tool.search(
                full_name=full_name,
                email=email or None,
                company=company or None,
                location=location or None,
                search_type=search_type,
                linkedin_url=li_url or None,
            )

        try:
            result = await _run(linkedin_url)

            # ── Fallback automatique pour les individus ───────────────────────
            # Si Apify ne trouve pas, on cherche dans web_profile_data
            # l'URL LinkedIn trouvée par le web_searcher et on réessaie.
            if (
                search_type == "individual"
                and (not result or result.get("status") in ("not_found", "error"))
                and tool_context is not None
            ):
                web_raw = tool_context.state.get("web_profile_data")
                fallback_url = _extract_linkedin_url_from_web_data(web_raw) if web_raw else None

                if fallback_url and fallback_url != linkedin_url:
                    logger.info(f"LinkedIn fallback: retrying with URL from web_profile_data: {fallback_url}")
                    retry = await _run(fallback_url)
                    if retry and retry.get("status") not in ("not_found", "error"):
                        result = retry
                        logger.info(f"LinkedIn fallback successful for {full_name}")
                    else:
                        logger.info(f"LinkedIn fallback also returned not_found for {full_name}")

            # ── Complément depuis le snippet de la mention web ────────────────
            # Si le résultat a un profile_url mais manque de skills/education/experience,
            # on parse le snippet LinkedIn trouvé par web_searcher (sans appel API).
            if (
                search_type == "individual"
                and result and result.get("profile_url")
                and _is_incomplete(result)
                and tool_context is not None
            ):
                web_raw = tool_context.state.get("web_profile_data")
                li_mention = _extract_linkedin_mention_from_web_data(web_raw) if web_raw else None
                if li_mention and (li_mention.get("snippet") or li_mention.get("title")):
                    logger.info(f"Extracting skills from web mention snippet for {full_name}")
                    fake_result = [{
                        "link": li_mention["url"],
                        "snippet": li_mention.get("snippet", ""),
                        "title": li_mention.get("title", ""),
                    }]
                    serp_provider = tool.serpapi or SerpAPIProvider.__new__(SerpAPIProvider)
                    parsed = serp_provider._parse_linkedin_results(fake_result, full_name)
                    if parsed:
                        result = _merge_results(result, parsed)
                        logger.info(f"Skills extracted from web mention: {result.get('skills')}")

            if not result:
                result = {
                    "status": "not_found",
                    "reason": f"No LinkedIn profile found for {full_name}",
                    "data_quality_score": 0.0,
                }

            # ── Dernier recours : skills + experience depuis les web_mentions ──
            if result and result.get("web_mentions"):
                if not result.get("skills"):
                    extracted_skills = _extract_skills_from_web_mentions(result["web_mentions"], full_name)
                    if extracted_skills:
                        result["skills"] = extracted_skills
                        logger.info(f"Skills from web_mentions: {extracted_skills}")
                if not result.get("experience"):
                    extracted_exp = _extract_experience_from_web_mentions(result["web_mentions"])
                    if extracted_exp:
                        result["experience"] = extracted_exp
                        logger.info(f"Experience from web_mentions: {extracted_exp}")

            # Écriture directe dans session.state — bypasse le reformatage LLM
            if tool_context is not None:
                tool_context.state["linkedin_profile_data"] = result

            found = bool(result.get("profile_url"))
            return f"linkedin_profile_data stored: {'found' if found else 'not_found'} for {full_name}"

        except Exception as e:
            logger.error(f"LinkedIn search error for {full_name}: {e}")
            err = {"status": "error", "reason": str(e), "data_quality_score": 0.0}
            if tool_context is not None:
                tool_context.state["linkedin_profile_data"] = err
            return f"linkedin_profile_data error: {e}"

    return FunctionTool(search_linkedin_profile)