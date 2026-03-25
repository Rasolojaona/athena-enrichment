"""
Web Scraper Tool - Extracts company information from their website.

Approach (no external scraping API required):
- Fetches homepage + priority sub-pages (/about, /contact, /services, /team)
- Extracts structured data (JSON-LD / Schema.org) — richest source
- Extracts Open Graph & meta tags
- Falls back to HTML section parsing
- Retry logic with exponential backoff
- HTML entity decoding
"""

import httpx
import os
import logging
import json
import re
import asyncio
import html as html_lib
from typing import Optional, Dict, Any, List
from urllib.parse import urljoin, urlparse, quote
from google.adk.tools import FunctionTool

logger = logging.getLogger(__name__)

# Stop words to strip when deriving domain candidates from a company name
_STOP_WORDS = {
    "and", "et", "the", "le", "la", "les", "de", "du", "des", "of", "for",
    "or", "in", "inc", "llc", "ltd", "sarl", "sas", "sa", "group", "groups",
    "solutions", "technologies", "technology", "tech", "services", "service",
    "consulting", "digital", "global", "international", "ai", "data",
}

# Sub-pages to try after the homepage
_SUBPAGE_SLUGS = [
    "/about", "/about-us", "/a-propos", "/qui-sommes-nous",
    "/services", "/nos-services", "/solutions",
    "/contact", "/contact-us", "/nous-contacter",
    "/team", "/notre-equipe", "/equipe",
    "/products", "/produits",
]

_USER_AGENTS = [
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/17.4 Safari/605.1.15"
    ),
]


# ─────────────────────────────────────────────
# Public tool
# ─────────────────────────────────────────────

def get_web_scraper_tool() -> FunctionTool:
    """Create ADK tool to extract company info from website."""

    async def scrape_company_website(company_name: str, website_url: str) -> str:
        """
        Extract company information from their website.

        Fetches the homepage and key sub-pages, extracts structured data
        (JSON-LD / Schema.org), Open Graph tags, and falls back to HTML parsing.

        Args:
            company_name: Name of the company
            website_url:  Company website URL

        Returns:
            JSON string with scraped company data
        """
        logger.info(f"🌐 Scraping website for {company_name}: {website_url or '(none — will guess from name)'}")

        # Normalise empty/placeholder values to empty string so the
        # name-candidate fallback below can take over.
        if not website_url or website_url.lower() in ["unknown", "not provided", "none"]:
            website_url = ""

        if website_url and not website_url.startswith(("http://", "https://")):
            website_url = f"https://{website_url}"

        timeout = int(os.getenv("TIMEOUT_SECONDS", "30"))
        base_url = _get_base_url(website_url)

        # Build ordered list of URLs to try:
        # 1. The provided URL (canonical)
        # 2. www. variant (SSL hostname mismatch is common)
        # 3. Candidates derived from company name (fallback if URL unknown / unreachable)
        urls_to_try: List[str] = []
        if website_url:
            urls_to_try.append(website_url)
            parsed_url = urlparse(website_url)
            if not parsed_url.netloc.startswith("www."):
                www_variant = website_url.replace(parsed_url.netloc, f"www.{parsed_url.netloc}", 1)
                urls_to_try.append(www_variant)

        # Only fall back to name-derived candidates if no URL was given or if provided URL fails
        name_candidates = _url_candidates_from_name(company_name)
        # Add candidates that aren't already in the list
        for c in name_candidates:
            if c not in urls_to_try:
                urls_to_try.append(c)

        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=True,
                verify=False,  # SSL certs on company sites are often misconfigured
                headers={"User-Agent": _USER_AGENTS[0], "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8"},
            ) as client:
                # 1. Try each URL in order until one succeeds
                homepage_html = None
                for attempt_url in urls_to_try:
                    homepage_html = await _fetch_with_retry(client, attempt_url)
                    if homepage_html:
                        if attempt_url != website_url:
                            logger.info(f"✅ Found working URL: {attempt_url}")
                        website_url = attempt_url
                        base_url = _get_base_url(website_url)
                        break
                    logger.info(f"⚠️  Unreachable: {attempt_url}")

                if not homepage_html:
                    return json.dumps({"status": "unreachable", "company_name": company_name, "data_quality_score": 0.0})

                all_html = homepage_html

                # 2. Discover and fetch priority sub-pages
                discovered = _discover_subpage_urls(homepage_html, base_url)
                subpage_htmls = await _fetch_subpages(client, discovered, base_url)
                for sp_html in subpage_htmls:
                    all_html += "\n" + sp_html

            # 3. Parse everything
            parsed = _parse_website_content(all_html, base_url)
            quality_score = _compute_quality_score(parsed)

            result = {
                "website_url": website_url,
                "company_name": company_name,
                "status": "found",
                "pages_scraped": 1 + len(subpage_htmls),
                "data": parsed,
                "data_quality_score": quality_score,
                "provider": "web_scraper",
            }

            logger.info(f"✅ Scraped {company_name} — {result['pages_scraped']} pages, quality: {quality_score:.2f}")
            return json.dumps(result)

        except httpx.TimeoutException:
            return json.dumps({"status": "timeout", "company_name": company_name, "data_quality_score": 0.0, "error": "Timed out"})
        except httpx.HTTPStatusError as e:
            return json.dumps({"status": "http_error", "company_name": company_name, "http_status": e.response.status_code, "data_quality_score": 0.0})
        except Exception as e:
            logger.error(f"Web scraping error for {company_name}: {e}")
            return json.dumps({"status": "error", "company_name": company_name, "error": str(e), "data_quality_score": 0.0})

    return FunctionTool(scrape_company_website)


# ─────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────

def _get_base_url(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}"


def _url_candidates_from_name(company_name: str) -> List[str]:
    """
    Generate plausible website URLs from a company name.
    Tries full name and a short version (first meaningful word(s)) across .com / .io / .ai.
    Never returns more than 6 candidates to keep latency low.
    """
    # Normalise: lowercase, keep only alphanumeric + spaces
    raw = company_name.lower()
    raw = re.sub(r"[^a-z0-9\s]", " ", raw)
    words = raw.split()

    # Full slug (all words joined with hyphens)
    full_slug = "-".join(words)
    # Short slug: strip stop-words, keep up to 2 remaining meaningful words
    meaningful = [w for w in words if w not in _STOP_WORDS and len(w) > 2]
    short_slug = "-".join(meaningful[:2]) if meaningful else full_slug
    # Single first meaningful word
    first_slug = meaningful[0] if meaningful else words[0]

    slugs = dict.fromkeys([full_slug, short_slug, first_slug])  # ordered, no dups
    tlds = [".com", ".io", ".ai"]

    candidates = []
    for slug in slugs:
        for tld in tlds:
            candidates.append(f"https://www.{slug}{tld}")
            if len(candidates) >= 6:
                return candidates
    return candidates


async def _fetch_with_retry(client: httpx.AsyncClient, url: str, retries: int = 2) -> Optional[str]:
    """Fetch URL with exponential backoff on failure."""
    for attempt in range(retries + 1):
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.text
        except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
            if attempt < retries:
                await asyncio.sleep(2 ** attempt)
            else:
                logger.warning(f"Failed to fetch {url} after {retries + 1} attempts: {e}")
                return None
        except Exception as e:
            logger.warning(f"Error fetching {url}: {e}")
            return None
    return None


def _discover_subpage_urls(html: str, base_url: str) -> List[str]:
    """Find internal links that match priority sub-page slugs."""
    found = set()
    for href in re.findall(r'href=["\']([^"\'#?]+)["\']', html, re.IGNORECASE):
        for slug in _SUBPAGE_SLUGS:
            if href.rstrip("/").endswith(slug) or href == slug:
                full = urljoin(base_url, href)
                if full.startswith(base_url):
                    found.add(full)
    # Also try canonical slugs directly
    for slug in _SUBPAGE_SLUGS:
        found.add(f"{base_url}{slug}")
    return list(found)[:8]  # cap at 8 sub-pages


async def _fetch_subpages(client: httpx.AsyncClient, urls: List[str], base_url: str) -> List[str]:
    """Fetch sub-pages concurrently, ignore failures."""
    tasks = [_fetch_with_retry(client, url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if isinstance(r, str) and r]


# ─────────────────────────────────────────────
# Main parser
# ─────────────────────────────────────────────

def _parse_website_content(html: str, base_url: str = "") -> Dict[str, Any]:
    clean = _clean_html(html)
    return {
        "title":               _extract_title(html),
        "description":         _extract_meta_description(html),
        "schema_org":          _extract_schema_org(html),
        "og_tags":             _extract_og_tags(html),
        "about":               _extract_section(clean, ["About", "About Us", "À propos", "Qui sommes-nous", "Who We Are"]),
        "mission_vision":      _extract_section(clean, ["Mission", "Vision", "Purpose", "Why We Exist", "Notre mission"]),
        "values":              _extract_section(clean, ["Values", "Our Values", "Culture", "Valeurs", "Principles"]),
        "services":            _extract_section(clean, ["Services", "Our Services", "What We Do", "Solutions", "Nos services", "Prestations"], max_chars=250),
        "products":            _extract_section(clean, ["Products", "Our Products", "Produits", "Offerings", "Platform", "Features"], max_chars=250),
        "team":                _extract_team(html),
        "testimonials":        _extract_testimonials(html),
        "news_blog":           _extract_news_blog(html),
        "job_openings":        _extract_job_openings(html),
        "geographic_presence": _extract_geography(clean),
        "partnerships":        _extract_section(clean, ["Partners", "Partnerships", "Certifications", "Clients", "Partenaires", "Trusted by", "Ils nous font confiance"]),
        "contact_info":        _extract_contact_info(html),
        "social_links":        _extract_social_links(html),
        "technology":          _extract_technology_hints(html),
    }


# ─────────────────────────────────────────────
# Quality score
# ─────────────────────────────────────────────

def _compute_quality_score(data: Dict[str, Any]) -> float:
    score = 0.2
    if data.get("description") or data.get("og_tags", {}).get("description"): score += 0.08
    if data.get("schema_org"):                                                  score += 0.10
    if data.get("about"):                                                       score += 0.08
    if data.get("mission_vision"):                                              score += 0.05
    if data.get("values"):                                                      score += 0.04
    if data.get("services") or data.get("products"):                            score += 0.12
    if data.get("team"):                                                        score += 0.08
    if data.get("testimonials"):                                                score += 0.05
    if data.get("news_blog"):                                                   score += 0.04
    if data.get("job_openings"):                                                score += 0.04
    if data.get("contact_info", {}).get("email") or \
       data.get("contact_info", {}).get("phone"):                              score += 0.06
    if data.get("social_links"):                                                score += 0.04
    if data.get("geographic_presence"):                                         score += 0.02
    return round(min(score, 1.0), 2)


# ─────────────────────────────────────────────
# HTML utilities
# ─────────────────────────────────────────────

def _clean_html(html: str) -> str:
    """Strip scripts/styles, decode HTML entities, collapse whitespace."""
    text = re.sub(r'<(script|style|noscript)[^>]*>[\s\S]*?</\1>', ' ', html, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = html_lib.unescape(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ─────────────────────────────────────────────
# Structured data extractors
# ─────────────────────────────────────────────

def _extract_schema_org(html: str) -> Dict[str, Any]:
    """Extract JSON-LD / Schema.org data — the richest structured source."""
    result: Dict[str, Any] = {}
    for raw in re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>([\s\S]*?)</script>',
        html, re.IGNORECASE
    ):
        try:
            data = json.loads(raw.strip())
            items = data if isinstance(data, list) else [data]
            for item in items:
                t = item.get("@type", "")
                if t in ("Organization", "Corporation", "LocalBusiness", "Company"):
                    result["name"]        = item.get("name") or result.get("name")
                    result["description"] = item.get("description") or result.get("description")
                    result["url"]         = item.get("url") or result.get("url")
                    result["logo"]        = item.get("logo", {}).get("url") if isinstance(item.get("logo"), dict) else item.get("logo")
                    result["email"]       = item.get("email") or result.get("email")
                    result["telephone"]   = item.get("telephone") or result.get("telephone")
                    result["founded"]     = item.get("foundingDate") or result.get("founded")
                    result["employees"]   = item.get("numberOfEmployees", {}).get("value") if isinstance(item.get("numberOfEmployees"), dict) else item.get("numberOfEmployees")
                    addr = item.get("address", {})
                    if isinstance(addr, dict):
                        result["address"] = ", ".join(filter(None, [
                            addr.get("streetAddress"),
                            addr.get("addressLocality"),
                            addr.get("addressCountry"),
                        ]))
                    sameAs = item.get("sameAs", [])
                    if sameAs:
                        result["same_as"] = sameAs if isinstance(sameAs, list) else [sameAs]
        except (json.JSONDecodeError, AttributeError):
            continue
    return {k: v for k, v in result.items() if v}


def _extract_og_tags(html: str) -> Dict[str, str]:
    """Extract Open Graph meta tags."""
    og: Dict[str, str] = {}
    for m in re.finditer(
        r'<meta\s+property=["\']og:(\w+)["\'][^>]*content=["\']([^"\']+)["\']',
        html, re.IGNORECASE
    ):
        og[m.group(1)] = html_lib.unescape(m.group(2).strip())
    for m in re.finditer(
        r'<meta\s+content=["\']([^"\']+)["\'][^>]*property=["\']og:(\w+)["\']',
        html, re.IGNORECASE
    ):
        og[m.group(2)] = html_lib.unescape(m.group(1).strip())
    return og


# ─────────────────────────────────────────────
# Meta extractors
# ─────────────────────────────────────────────

def _extract_title(html: str) -> Optional[str]:
    m = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
    return html_lib.unescape(m.group(1).strip()) if m else None


def _extract_meta_description(html: str) -> Optional[str]:
    patterns = [
        r'<meta\s+name=["\']description["\'][^>]*content=["\']([^"\']+)["\']',
        r'<meta\s+content=["\']([^"\']+)["\'][^>]*name=["\']description["\']',
    ]
    for p in patterns:
        m = re.search(p, html, re.IGNORECASE)
        if m:
            return html_lib.unescape(m.group(1).strip())
    return None


# ─────────────────────────────────────────────
# Section extractors (work on cleaned text)
# ─────────────────────────────────────────────

def _extract_section(clean_text: str, keywords: List[str], max_chars: int = 300) -> List[str]:
    """
    Find text passages near section keywords.
    Works on clean (tag-stripped) text for better recall.
    """
    results = []
    kw_pattern = '|'.join(re.escape(k) for k in keywords)

    # Strategy 1: keyword followed by content within ~400 chars
    for m in re.finditer(
        rf'\b(?:{kw_pattern})\b[^\n]{{0,80}}\n([\s\S]{{20,{max_chars}}}?)(?:\n\n|\Z)',
        clean_text, re.IGNORECASE
    ):
        text = re.sub(r'\s+', ' ', m.group(1)).strip()
        if len(text) >= 20:
            results.append(text[:max_chars])
        if len(results) >= 4:
            break

    # Strategy 2: sentences containing keywords
    if not results:
        for sent in re.split(r'(?<=[.!?])\s+', clean_text):
            if re.search(kw_pattern, sent, re.IGNORECASE) and len(sent) > 30:
                results.append(sent.strip()[:max_chars])
            if len(results) >= 3:
                break

    return results


def _extract_team(html: str) -> List[str]:
    members = []
    # Pattern: Name in heading/strong + nearby title/role
    for m in re.finditer(
        r'<(?:h[2-4]|strong)[^>]*>([A-ZÀ-Ü][a-zà-ü\-]+ [A-ZÀ-Ü][a-zà-ü\-]+(?:\s[A-ZÀ-Ü][a-zà-ü\-]+)?)</(?:h[2-4]|strong)>'
        r'[\s\S]{0,200}?<(?:p|span|div)[^>]*>([^<]{5,80})</(?:p|span|div)>',
        html, re.IGNORECASE
    ):
        name = m.group(1).strip()
        role = re.sub(r'\s+', ' ', m.group(2)).strip()
        entry = f"{name} — {role}"
        if entry not in members:
            members.append(entry)
        if len(members) >= 8:
            break
    return members


def _extract_testimonials(html: str) -> List[str]:
    quotes = []
    for m in re.finditer(r'<blockquote[^>]*>([\s\S]{20,400}?)</blockquote>', html, re.IGNORECASE):
        text = re.sub(r'<[^>]+>', '', m.group(1))
        text = html_lib.unescape(re.sub(r'\s+', ' ', text)).strip()
        if len(text) > 20 and text not in quotes:
            quotes.append(text[:300])
        if len(quotes) >= 4:
            break
    # Fallback: elements with testimonial-related class
    if not quotes:
        for m in re.finditer(
            r'<[^>]+class=["\'][^"\']*(?:testimonial|review|quote|temoignage)[^"\']*["\'][^>]*>([\s\S]{20,300}?)</[^>]+>',
            html, re.IGNORECASE
        ):
            text = re.sub(r'<[^>]+>', '', m.group(1))
            text = html_lib.unescape(re.sub(r'\s+', ' ', text)).strip()
            if len(text) > 20:
                quotes.append(text[:300])
            if len(quotes) >= 4:
                break
    return quotes


def _extract_news_blog(html: str) -> List[str]:
    titles = []
    # Articles with blog/news/post class
    for m in re.finditer(
        r'<(?:h[2-4]|a)[^>]+class=["\'][^"\']*(?:post|article|blog|news|actualit)[^"\']*["\'][^>]*>'
        r'([^<]{10,150})</(?:h[2-4]|a)>',
        html, re.IGNORECASE
    ):
        t = html_lib.unescape(re.sub(r'\s+', ' ', m.group(1))).strip()
        if t and t not in titles:
            titles.append(t)
        if len(titles) >= 5:
            break
    # Fallback: <article> headings
    if not titles:
        for m in re.finditer(
            r'<article[^>]*>[\s\S]{0,500}?<h[2-4][^>]*>([^<]{10,150})</h[2-4]>',
            html, re.IGNORECASE
        ):
            t = html_lib.unescape(re.sub(r'\s+', ' ', m.group(1))).strip()
            if t and t not in titles:
                titles.append(t)
            if len(titles) >= 5:
                break
    return titles


def _extract_job_openings(html: str) -> List[str]:
    jobs = []
    job_kw = (
        r'Engineer|Developer|Manager|Designer|Analyst|Consultant|Lead|Intern|Director|Officer|'
        r'Ingénieur|Développeur|Responsable|Chef de projet|Coordinateur|Commercial|Chargé'
    )
    for m in re.finditer(
        rf'<(?:h[2-5]|a|li|td)[^>]*>([^<]{{5,100}}(?:{job_kw})[^<]{{0,60}})</(?:h[2-5]|a|li|td)>',
        html, re.IGNORECASE
    ):
        t = html_lib.unescape(re.sub(r'\s+', ' ', m.group(1))).strip()
        if t not in jobs:
            jobs.append(t)
        if len(jobs) >= 6:
            break
    return jobs


def _extract_geography(clean_text: str) -> List[str]:
    locations = []
    patterns = [
        r'(?:offices?\s+in|located\s+in|headquartered\s+in|based\s+in|operat(?:ing|e)\s+in)[^.]{5,150}\.',
        r'(?:présent[^.]{0,15}dans|basé[^.]{0,10}à|siège[^.]{0,10}à|bureaux[^.]{0,10}à)[^.]{5,150}\.',
    ]
    for pat in patterns:
        for m in re.finditer(pat, clean_text, re.IGNORECASE):
            text = re.sub(r'\s+', ' ', m.group(0)).strip()
            if text not in locations:
                locations.append(text[:200])
            if len(locations) >= 3:
                break
    return locations


def _extract_contact_info(html: str) -> Dict[str, Optional[str]]:
    clean = re.sub(r'<(script|style)[^>]*>[\s\S]*?</\1>', '', html, flags=re.IGNORECASE)
    clean = html_lib.unescape(clean)

    contact: Dict[str, Optional[str]] = {"email": None, "phone": None, "address": None}

    # Emails — prefer contact/info addresses over noreply
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', clean)
    preferred = [e for e in emails if not re.search(r'noreply|no-reply|donotreply|example', e, re.I)]
    contact["email"] = (preferred or emails)[0] if (preferred or emails) else None

    # Phone — mailto: href as fallback
    phone_m = re.search(r'tel:([+\d\s\-\(\)]{7,})', clean)
    if phone_m:
        contact["phone"] = phone_m.group(1).strip()
    else:
        phone_m = re.search(r'(?<!\d)(\+?[\d][\d\s\-\.\(\)]{8,}\d)(?!\d)', clean)
        if phone_m:
            contact["phone"] = phone_m.group(1).strip()

    # Address
    addr_m = re.search(
        r'\d{1,5}\s+\w[\w\s,\.]{10,80}(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Rue|BP|Lot)\b',
        clean, re.IGNORECASE
    )
    if addr_m:
        contact["address"] = addr_m.group(0).strip()

    return contact


def _extract_social_links(html: str) -> Dict[str, str]:
    patterns = {
        "linkedin":  r'https?://(?:www\.)?linkedin\.com/(?:company|in)/[\w\-]+',
        "twitter":   r'https?://(?:www\.)?(?:twitter|x)\.com/[\w\-]+',
        "facebook":  r'https?://(?:www\.)?facebook\.com/(?!sharer|share)[\w\.\-]+',
        "instagram": r'https?://(?:www\.)?instagram\.com/[\w\.\-]+',
        "youtube":   r'https?://(?:www\.)?youtube\.com/(?:c/|channel/|@)[\w\-]+',
        "github":    r'https?://(?:www\.)?github\.com/[\w\-]+',
        "tiktok":    r'https?://(?:www\.)?tiktok\.com/@[\w\.\-]+',
    }
    links = {}
    for platform, pat in patterns.items():
        m = re.search(pat, html, re.IGNORECASE)
        if m:
            links[platform] = m.group(0)
    return links


def _extract_technology_hints(html: str) -> List[str]:
    tech_map = {
        "React":      r'react(?:\.js|dom|-dom)',
        "Vue.js":     r'vue(?:\.js)?',
        "Angular":    r'angular(?:\.js)?',
        "Next.js":    r'_next/static|next(?:\.js)?',
        "Nuxt.js":    r'nuxt(?:\.js)?',
        "Svelte":     r'svelte',
        "Node.js":    r'node(?:\.js)?',
        "Python":     r'python',
        "Django":     r'django',
        "FastAPI":    r'fastapi',
        "Laravel":    r'laravel',
        "WordPress":  r'wp-content|wordpress',
        "Shopify":    r'shopify',
        "Webflow":    r'webflow\.com',
        "Wix":        r'wix\.com',
        "Salesforce": r'salesforce',
        "HubSpot":    r'hubspot',
        "Stripe":     r'stripe\.com',
        "Google Analytics": r'gtag\(|google-analytics|UA-\d+',
    }
    found = []
    for name, pat in tech_map.items():
        if re.search(pat, html, re.IGNORECASE):
            found.append(name)
    return found
