"""
Athena Enrichment - Cloud Run Entry Point
Reçoit les messages Pub/Sub (push) et lance l'enrichissement via l'agent ADK.
Compatible avec le Pub/Sub emulator local (port 8085) et GCP Cloud Run.
"""
import base64
import httpx
import os
import uuid
import asyncio
import json
import re
import time
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv("agent_enrichment_orchestrator/.env")

from fastapi import FastAPI, Request, HTTPException
from google.adk.runners import Runner
from google.genai import types
import google.genai as google_genai
from agent_enrichment_orchestrator.agent import root_agent_batch
from agent_enrichment_orchestrator.session_service import get_session_service
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Athena Enrichment — Cloud Run")

# ── Runner ADK ────────────────────────────────────────────────────────────────
# InMemory (dev local) ou DatabaseSessionService (Cloud SQL PostgreSQL en prod).
# Contrôlé par la variable DATABASE_URL dans .env.
batch_runner = Runner(
    agent=root_agent_batch,
    app_name="athena-enrichment",
    session_service=get_session_service(),
)

# ── Semaphore de concurrence ──────────────────────────────────────────────────
# Limite le nombre d'enrichissements parallèles pour ne pas saturer les APIs.
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_ENRICHMENTS", "5"))
_sem = asyncio.Semaphore(MAX_CONCURRENT)

# Verrou en mémoire — évite d'enrichir le même ID en parallèle (duplicates batch)
_in_progress: set = set()


# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "athena-enrichment-cloudrun"}


# ─────────────────────────────────────────────
# Batch Endpoints
# ─────────────────────────────────────────────

@app.post("/batch/all")
async def batch_all():
    asyncio.create_task(_batch_enrich_companies())
    asyncio.create_task(_batch_enrich_registrations())
    return {"status": "started", "tasks": ["companies", "registrations"]}


@app.post("/batch/companies")
async def batch_companies():
    asyncio.create_task(_batch_enrich_companies())
    return {"status": "started", "tasks": ["companies"]}


@app.post("/batch/registrations")
async def batch_registrations():
    asyncio.create_task(_batch_enrich_registrations())
    return {"status": "started", "tasks": ["registrations"]}


# ─────────────────────────────────────────────
# Pub/Sub Push Endpoint
# ─────────────────────────────────────────────

@app.post("/pubsub")
async def pubsub_push(request: Request):
    """
    Reçoit les messages Pub/Sub en push.
    Pub/Sub envoie un POST avec ce format :
    {
      "message": {
        "data": "<base64-encoded JSON>",
        "messageId": "...",
        "publishTime": "..."
      },
      "subscription": "..."
    }
    """
    body = await request.json()
    message = body.get("message", {})
    data_b64 = message.get("data", "")

    if not data_b64:
        raise HTTPException(status_code=400, detail="No data field in Pub/Sub message")

    try:
        payload = json.loads(base64.b64decode(data_b64).decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid message data: {e}")

    record_type = payload.get("type", "individual")
    doc_id = payload.get("visiteur_id") or payload.get("id") or str(uuid.uuid4())

    logger.info(f"📨 Pub/Sub reçu : {record_type} | id={doc_id}")

    if record_type == "company":
        name = payload.get("name") or payload.get("full_name", "")
        initial_state = {
            "visiteur_id": doc_id,
            "firestore_collection": "companies",
            "visiteur_data": {
                "visiteur_id": doc_id,
                "full_name": name,
                "nom": name,
                "prenom": "",
                "email": payload.get("email", ""),
                "company": name,
                "website": payload.get("website", ""),
                "industry": payload.get("industry", ""),
                "members_count": payload.get("members_count", 0),
                "type": "company",
            },
        }
        message_text = (
            f"Enrich company profile for {name}, "
            f"email: {payload.get('email', '')}, "
            f"industry: {payload.get('industry', 'unknown')}"
        )
    else:
        first_name = payload.get("prenom") or payload.get("first_name", "")
        last_name = payload.get("nom") or payload.get("last_name", "")
        full_name = payload.get("full_name") or f"{first_name} {last_name}".strip()
        initial_state = {
            "visiteur_id": doc_id,
            "firestore_collection": "event_registrations",
            "visiteur_data": {
                "visiteur_id": doc_id,
                "type": "individual",
                "full_name": full_name,
                "prenom": first_name,
                "nom": last_name,
                "email": payload.get("email", ""),
                "company": payload.get("company", ""),
                "event_title": payload.get("event_title", ""),
                "event_location": payload.get("event_location", ""),
            },
        }
        message_text = (
            f"Enrich profile for {full_name}, "
            f"email: {payload.get('email', '')}, "
            f"company: {payload.get('company', '')}"
        )

    # Traitement synchrone — Pub/Sub reçoit le 200 seulement après succès.
    # Si on retourne 200 avant la fin (create_task), Pub/Sub ne peut pas relivrer
    # en cas d'échec silencieux.
    await _run_enrichment(
        session_id=f"{record_type}_{doc_id}_{uuid.uuid4().hex[:6]}",
        message=message_text,
        initial_state=initial_state,
        user_id=f"pubsub_{record_type}",
    )

    return {"status": "done", "doc_id": doc_id, "type": record_type}


# ─────────────────────────────────────────────
# Auth emulator
# ─────────────────────────────────────────────

_auth_token: str = ""
_token_expiry: float = 0.0


async def _get_auth_token() -> str:
    global _auth_token, _token_expiry
    if _auth_token and time.time() < _token_expiry - 60:
        return _auth_token

    try:
        import google.auth
        import google.auth.transport.requests as google_requests
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        await asyncio.to_thread(credentials.refresh, google_requests.Request())
        _auth_token = credentials.token
        _token_expiry = (
            credentials.expiry.timestamp()
            if credentials.expiry
            else time.time() + 3600
        )
        logger.info("Token Google ADC obtenu")
        return _auth_token
    except Exception as e:
        logger.warning(f"Google auth failed: {e}")
        return ""


# ─────────────────────────────────────────────
# Firestore REST helpers
# ─────────────────────────────────────────────

def _firestore_url(path: str) -> str:
    project = os.getenv("FIREBASE_PROJECT_ID", "gen-lang-client-0685149390")
    return f"https://firestore.googleapis.com/v1/projects/{project}/databases/(default)/documents/{path}"


def _parse_fs_value(value: dict):
    if "stringValue"    in value: return value["stringValue"]
    if "integerValue"   in value: return int(value["integerValue"])
    if "doubleValue"    in value: return float(value["doubleValue"])
    if "booleanValue"   in value: return value["booleanValue"]
    if "nullValue"      in value: return None
    if "timestampValue" in value: return value["timestampValue"]
    if "arrayValue"     in value:
        return [_parse_fs_value(v) for v in value["arrayValue"].get("values", [])]
    if "mapValue"       in value:
        return {k: _parse_fs_value(v) for k, v in value["mapValue"].get("fields", {}).items()}
    return value


def _to_firestore_value(v) -> dict:
    if v is None:            return {"nullValue": None}
    if isinstance(v, bool):  return {"booleanValue": v}
    if isinstance(v, int):   return {"integerValue": str(v)}
    if isinstance(v, float): return {"doubleValue": v}
    if isinstance(v, list):  return {"arrayValue": {"values": [_to_firestore_value(i) for i in v]}}
    if isinstance(v, dict):  return {"mapValue": {"fields": {k: _to_firestore_value(vv) for k, vv in v.items()}}}
    return {"stringValue": str(v)}


# ─────────────────────────────────────────────
# Enrichment schema builders
# ─────────────────────────────────────────────

def _parse_json_safe(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw or not isinstance(raw, str):
        return {}
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if match:
        raw = match.group(1).strip()
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _completeness(fields: list) -> float:
    filled = sum(1 for f in fields if f not in (None, "", [], {}))
    return round(filled / len(fields), 2) if fields else 0.0


_GENERIC_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "icloud.com", "live.com", "ymail.com", "aol.com", "protonmail.com",
}

_SENIOR_TITLES = {"senior", "lead", "manager", "principal", "consultant"}
_EXECUTIVE_TITLES = {"ceo", "cto", "cfo", "coo", "founder", "co-founder", "director",
                     "vp", "head", "partner", "president"}


def _compute_scores(linkedin: dict, facebook: dict, web: dict, visiteur_data: dict = None) -> dict:
    li_url    = (linkedin or {}).get("profile_url") or ""
    fb_url    = ((facebook or {}).get("facebook_profile") or {}).get("profile_url") or ""
    web_found = (web or {}).get("status") == "found"

    # Overall Quality Score
    oq = 0.0
    if li_url:    oq += 0.30
    if fb_url:    oq += 0.20
    if web_found: oq += 0.20
    if visiteur_data and all([visiteur_data.get("email"), visiteur_data.get("nom"), visiteur_data.get("prenom")]):
        oq += 0.15
    if sum([bool(li_url), bool(fb_url), web_found]) >= 2:
        oq += 0.15

    # Networking Potential Score
    np_ = 0.0
    connections = (linkedin or {}).get("connections_count") or 0
    if connections >= 500:   np_ += 0.30
    elif connections >= 100: np_ += 0.15
    headline_words = set(((linkedin or {}).get("headline") or "").lower().replace("-", " ").split())
    if headline_words & _EXECUTIVE_TITLES: np_ += 0.25
    elif headline_words & _SENIOR_TITLES:  np_ += 0.15
    if li_url:    np_ += 0.20
    if fb_url:    np_ += 0.10
    if web_found: np_ += 0.15

    # Engagement Likelihood Score
    el = 0.0
    if li_url:    el += 0.25
    if fb_url:    el += 0.20
    if web_found: el += 0.20
    if ((linkedin or {}).get("data_quality_score") or 0.0) >= 0.5: el += 0.15
    if (facebook or {}).get("public_mentions"): el += 0.10

    return {
        "overall_quality":       round(min(oq,  1.0) * 100),
        "networking_potential":  round(min(np_, 1.0) * 100),
        "engagement_likelihood": round(min(el,  1.0) * 100),
    }


# ─────────────────────────────────────────────
# Gemini summary generator
# ─────────────────────────────────────────────

_genai_client = None

def _get_genai_client() -> google_genai.Client:
    global _genai_client
    if _genai_client is None:
        if os.getenv("GOOGLE_GENAI_USE_VERTEXAI") == "1":
            _genai_client = google_genai.Client(
                vertexai=True,
                project=os.getenv("GOOGLE_CLOUD_PROJECT"),
                location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
            )
        else:
            _genai_client = google_genai.Client(api_key=os.getenv("GOOGLE_API_KEY", ""))
    return _genai_client


async def _generate_summary(enrichment: dict) -> str:
    """
    Génère un résumé professionnel via Gemini à partir des données enrichies.
    Retourne une chaîne vide en cas d'échec — n'est jamais bloquant.
    Lit depuis le nouveau format : profile, experience, education à la racine.
    """
    try:
        record_type = enrichment.get("record_type", "individual")

        if record_type == "individual":
            profile = enrichment.get("profile", {})
            name    = profile.get("full_name", "")
            parts   = []
            pos  = profile.get("current_title")
            comp = profile.get("current_company")
            if pos and comp:
                parts.append(f"Poste actuel : {pos} chez {comp}")
            if profile.get("skills"):
                parts.append(f"Compétences : {', '.join(profile['skills'][:8])}")
            exp_list = enrichment.get("experience", [])[:3]
            if exp_list:
                exp_strs = [
                    f"{e.get('title','')} chez {e.get('company','')}"
                    for e in exp_list if e.get("title")
                ]
                if exp_strs:
                    parts.append(f"Expériences : {' ; '.join(exp_strs)}")
            edu_list = enrichment.get("education", [])[:2]
            if edu_list:
                edu_strs = [e.get("school", "") for e in edu_list if e.get("school")]
                if edu_strs:
                    parts.append(f"Formation : {', '.join(edu_strs)}")
            if profile.get("location"):
                parts.append(f"Localisation : {profile['location']}")
            if not parts:
                return ""
            prompt = (
                f"Voici les données d'un profil professionnel :\n"
                f"Nom : {name}\n"
                + "\n".join(parts)
                + "\n\nRédige un résumé professionnel de 2 à 3 phrases, factuel et neutre, "
                "dans la même langue que les données. Écris uniquement le résumé, sans titre ni introduction."
            )

        else:  # company
            profile  = enrichment.get("profile", {})
            offerings = enrichment.get("offerings", {})
            name     = profile.get("name", "")
            parts = []
            if profile.get("industry"):
                parts.append(f"Secteur : {profile['industry']}")
            if profile.get("description"):
                parts.append(f"Description : {profile['description'][:300]}")
            if offerings.get("services"):
                parts.append(f"Services : {', '.join(offerings['services'][:3])}")
            if offerings.get("specialties"):
                parts.append(f"Spécialités : {', '.join(offerings['specialties'][:5])}")
            if profile.get("headquarters"):
                parts.append(f"Siège : {profile['headquarters']}")
            if not parts:
                return ""
            prompt = (
                f"Voici les données d'une entreprise :\n"
                f"Nom : {name}\n"
                + "\n".join(parts)
                + "\n\nRédige une description de l'entreprise en 2 à 3 phrases, factuelle et neutre, "
                "dans la même langue que les données. Écris uniquement la description, sans titre ni introduction."
            )

        client   = _get_genai_client()
        response = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=200, temperature=0.3),
        )
        summary = (response.text or "").strip()
        logger.info(f"Summary generated ({len(summary)} chars)")
        return summary

    except Exception as e:
        logger.warning(f"Summary generation skipped: {e}")
        return ""


def _extract_web_links(mentions: list) -> list:
    """Transforme les web mentions brutes en liste de liens propres {url, title, source}."""
    from urllib.parse import urlparse
    links = []
    seen  = set()
    for m in (mentions or []):
        if not isinstance(m, dict):
            continue
        url = m.get("link") or m.get("url") or ""
        if not url or url in seen:
            continue
        seen.add(url)
        try:
            source = urlparse(url).netloc.replace("www.", "")
        except Exception:
            source = ""
        links.append({
            "url":    url,
            "title":  (m.get("title") or "").strip(),
            "source": source,
        })
    return links[:10]


def _filter_skills(skills: list, first_name: str, last_name: str) -> list:
    """Supprime les skills invalides : noms propres, phrases, mots de la personne."""
    name_parts = {p.lower() for p in (first_name + " " + last_name).split() if len(p) > 2}
    clean = []
    for skill in skills:
        if not isinstance(skill, str):
            continue
        s = skill.strip()
        if not s or len(s) > 50:
            continue
        # Contient le nom ou prénom de la personne
        if any(part in s.lower() for part in name_parts):
            continue
        # Plusieurs mots → probablement une phrase extraite du texte, pas un skill
        if len(s.split()) > 4:
            continue
        clean.append(s)
    return clean


def _build_individual_enrichment(source_id: str, visiteur_data: dict,
                                  linkedin: dict, facebook: dict, web: dict,
                                  web_extracted: dict = None) -> dict:
    fb_p = facebook.get("facebook_profile") or {}

    first_name = visiteur_data.get("prenom", "")
    last_name  = visiteur_data.get("nom", "")
    email      = visiteur_data.get("email", "")

    inferred_company = None
    if email and "@" in email:
        domain = email.split("@")[-1].lower()
        if domain not in _GENERIC_EMAIL_DOMAINS:
            inferred_company = domain.split(".")[0]

    # URL personnelle extraite du snippet Facebook si non trouvée ailleurs
    fb_snippet = fb_p.get("snippet", "") or ""
    fb_bio_url = None
    _m = re.search(r'https?://[^\s"\'<>]+', fb_snippet)
    if _m:
        _candidate = _m.group(0).rstrip(".,)")
        if "facebook.com" not in _candidate:
            fb_bio_url = _candidate

    # Données extraites depuis les pages web via Jina.ai + Gemini (content_extractor agent)
    web_ext = web_extracted or {}

    # Facebook education est une string, LinkedIn une liste — normaliser
    fb_education = fb_p.get("education")
    education = (
        linkedin.get("education", [])
        or web_ext.get("education", [])
        or ([{"school": fb_education}] if fb_education else [])
    )

    # Merge skills: LinkedIn (structured) + Jina extraction + web snippets
    li_skills  = _filter_skills(linkedin.get("skills", []), first_name, last_name)
    ext_skills = _filter_skills(web_ext.get("skills") or [], first_name, last_name)
    snp_skills = [s.replace("_", " ") for s in (web.get("skills") or [])]
    seen_skills: set = set()
    skills: list = []
    for s in li_skills + ext_skills + snp_skills:
        key = s.lower()
        if key not in seen_skills:
            seen_skills.add(key)
            skills.append(s)

    location = linkedin.get("location") or fb_p.get("location")
    li_url   = linkedin.get("profile_url") or web.get("linkedin_url")
    # Facebook URL: prefer structured data from facebook_searcher, fallback to web general search
    fb_url   = fb_p.get("profile_url") or web.get("facebook_url")

    sources = {
        "linkedin": bool(li_url),
        "facebook": bool(fb_p.get("profile_url")),  # only count structured FB data
        "web":      web.get("status") == "found",
    }

    current_title   = linkedin.get("current_position") or web_ext.get("current_position")
    current_company = (
        linkedin.get("current_company")
        or web_ext.get("current_company")
        or web.get("github_company")
        or inferred_company
    )
    headline = (
        linkedin.get("headline")
        or web_ext.get("description")
        or web.get("professional_description")
    )
    experience     = linkedin.get("experience", []) or web_ext.get("experience", [])
    certifications = linkedin.get("certifications", [])

    completeness_fields = [
        visiteur_data.get("full_name"), email, location,
        current_title, current_company,
        linkedin.get("industry"),
        li_url, fb_url, skills, experience,
    ]

    return {
        "enrichment_id": str(uuid.uuid4()),
        "source_id":     source_id,
        "record_type":   "individual",
        "status":        "completed",
        "enriched_at":   datetime.now(timezone.utc).isoformat(),
        "completeness":  round(_completeness(completeness_fields) * 100),
        "sources":       sources,
        "profile": {
            "full_name":       visiteur_data.get("full_name", ""),
            "first_name":      first_name,
            "last_name":       last_name,
            "initials":        "".join(p[0].upper() for p in [first_name, last_name] if p)[:2],
            "email":           email,
            "location":        location,
            "current_title":   current_title,
            "current_company": current_company,
            "industry":        linkedin.get("industry"),
            "headline":        headline,
            "skills":          skills,
            "summary":         "",  # rempli par _generate_summary après construction
        },
        "links": {
            "linkedin": li_url,
            "facebook": fb_url,
            "website":  web.get("personal_site") or fb_bio_url,
            "github":   web.get("github_url"),
        },
        "experience":     experience,
        "certifications": certifications,
        "education":      education,
        "web_links":      _extract_web_links(web.get("mentions", [])),
        "scores":      _compute_scores(linkedin, facebook, web, visiteur_data),
        "event": {
            "title":    visiteur_data.get("event_title", ""),
            "location": visiteur_data.get("event_location", ""),
        },
    }


def _build_company_enrichment(source_id: str, visiteur_data: dict,
                               linkedin: dict, facebook: dict, web: dict) -> dict:
    web_data    = web.get("data", {}) if isinstance(web.get("data"), dict) else {}
    schema_org  = web_data.get("schema_org", {}) if isinstance(web_data.get("schema_org"), dict) else {}
    og_tags     = web_data.get("og_tags", {}) if isinstance(web_data.get("og_tags"), dict) else {}
    web_socials = web_data.get("social_links", {}) if isinstance(web_data.get("social_links"), dict) else {}
    web_contact = web_data.get("contact_info", {}) if isinstance(web_data.get("contact_info"), dict) else {}

    fb_p = facebook.get("facebook_profile") or {}

    li_url  = linkedin.get("profile_url") or web_socials.get("linkedin")
    fb_url  = fb_p.get("profile_url") or web_socials.get("facebook")
    slug    = visiteur_data.get("slug", "")
    website = visiteur_data.get("website") or web.get("website_url") or (f"https://www.{slug}.com" if slug else None)

    sources = {
        "linkedin": bool(li_url),
        "facebook": bool(fb_url),
        "web":      web.get("status") == "found",
    }

    description = (
        linkedin.get("description")
        or (web_data.get("about") or [None])[0]
        or web_data.get("description")
        or og_tags.get("description")
        or schema_org.get("description")
    )

    completeness_fields = [
        visiteur_data.get("full_name"), website,
        linkedin.get("headquarters") or fb_p.get("location"),
        visiteur_data.get("industry") or linkedin.get("industry"),
        description, linkedin.get("company_size"),
        li_url, fb_url,
        visiteur_data.get("email"),
        web_data.get("services") or web_data.get("products"),
    ]

    return {
        "enrichment_id": str(uuid.uuid4()),
        "source_id":     source_id,
        "record_type":   "company",
        "status":        "completed",
        "enriched_at":   datetime.now(timezone.utc).isoformat(),
        "completeness":  round(_completeness(completeness_fields) * 100),
        "sources":       sources,
        "profile": {
            "name":           visiteur_data.get("full_name", ""),
            "industry":       visiteur_data.get("industry") or linkedin.get("industry"),
            "description":    description,
            "tagline":        linkedin.get("tagline") or og_tags.get("title"),
            "employee_count": linkedin.get("company_size") or schema_org.get("employees"),
            "founded_year":   linkedin.get("founded") or schema_org.get("founded"),
            "headquarters":   linkedin.get("headquarters") or fb_p.get("location") or schema_org.get("address"),
            "logo_url":       schema_org.get("logo"),
            "summary":        "",  # rempli par _generate_summary après construction
        },
        "links": {
            "linkedin": li_url,
            "facebook": fb_url,
            "website":  website,
        },
        "contacts": {
            "email":   visiteur_data.get("email") or web_contact.get("email"),
            "phone":   visiteur_data.get("phone") or web_contact.get("phone"),
            "address": visiteur_data.get("address") or web_contact.get("address"),
        },
        "offerings": {
            "services":    web_data.get("services", []),
            "products":    web_data.get("products", []),
            "specialties": linkedin.get("specialties", []),
        },
        "intelligence": {
            "tech_stack":   web_data.get("technology", []),
            "job_openings": web_data.get("job_openings", []),
            "recent_news":  web_data.get("news_blog", []),
            "partnerships": web_data.get("partnerships", []),
            "team":         web_data.get("team", []),
        },
        "web_links": _extract_web_links(web.get("mentions", []) or web_data.get("mentions", [])),
        "scores": {
            "overall_quality": _compute_scores(linkedin, facebook, web)["overall_quality"],
        },
    }


# ─────────────────────────────────────────────
# ADK Runner
# ─────────────────────────────────────────────

async def _save_enrichment_from_state(initial_state: dict, tool_results: dict):
    visiteur_data = initial_state.get("visiteur_data", {})
    doc_id = initial_state.get("visiteur_id") or visiteur_data.get("visiteur_id")
    if not doc_id:
        logger.error("❌ Cannot save: no visiteur_id")
        return

    linkedin      = _parse_json_safe(tool_results.get("linkedin_profile_data", ""))
    facebook      = _parse_json_safe(tool_results.get("facebook_profile_data", ""))
    web           = _parse_json_safe(tool_results.get("web_profile_data", ""))
    web_extracted = tool_results.get("web_extracted_data") or {}
    if isinstance(web_extracted, str):
        web_extracted = _parse_json_safe(web_extracted)

    record_type = visiteur_data.get("type", "individual")
    if record_type == "company":
        enrichment = _build_company_enrichment(doc_id, visiteur_data, linkedin, facebook, web)
    else:
        enrichment = _build_individual_enrichment(doc_id, visiteur_data, linkedin, facebook, web, web_extracted)

    # Générer le résumé professionnel via Gemini (non bloquant si échec)
    summary = await _generate_summary(enrichment)
    if summary:
        enrichment["profile"]["summary"] = summary

    logger.info(
        f"Completeness: {enrichment['completeness']}% | "
        f"Sources: {[k for k,v in enrichment['sources'].items() if v]} | "
        f"Summary: {'OK' if summary else '-'}"
    )

    token = await _get_auth_token()
    payload = json.dumps({"fields": {k: _to_firestore_value(v) for k, v in enrichment.items()}})

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.patch(
            _firestore_url(f"enrichments/{doc_id}"),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            content=payload,
        )
        if resp.status_code in (200, 201):
            logger.info(f"✅ Enrichment sauvegardé : enrichments/{doc_id}")
        else:
            logger.error(f"❌ Firestore write failed ({resp.status_code}): {resp.text[:300]}")


async def _run_enrichment(session_id: str, message: str, initial_state: dict, user_id: str = "pubsub"):
    try:
        await batch_runner.session_service.create_session(
            app_name=batch_runner.app_name,
            user_id=user_id,
            session_id=session_id,
            state=initial_state,
        )
        user_message = types.Content(role="user", parts=[types.Part(text=message)])

        async for _ in batch_runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=user_message,
        ):
            pass

        session = await batch_runner.session_service.get_session(
            app_name=batch_runner.app_name,
            user_id=user_id,
            session_id=session_id,
        )
        tool_results = {}
        if session:
            tool_results["linkedin_profile_data"] = session.state.get("linkedin_profile_data", "")
            tool_results["facebook_profile_data"] = session.state.get("facebook_profile_data", "")
            tool_results["web_profile_data"]       = session.state.get("web_profile_data", "")
            tool_results["web_extracted_data"]     = session.state.get("web_extracted_data", {})
            li  = _parse_json_safe(tool_results.get("linkedin_profile_data", ""))
            fb  = _parse_json_safe(tool_results.get("facebook_profile_data", ""))
            web = _parse_json_safe(tool_results.get("web_profile_data", ""))
            logger.info(
                f"LinkedIn: {'found' if li.get('profile_url') else 'not found'} | "
                f"Facebook: {'found' if (fb.get('facebook_profile') or {}).get('profile_url') else 'not found'} | "
                f"Web: {'found' if web.get('status') == 'found' else 'not found'}"
            )

        await _save_enrichment_from_state(initial_state, tool_results)

    except Exception as e:
        logger.error(f"❌ Enrichment failed for session {session_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())


# ─────────────────────────────────────────────
# Firestore helpers (lecture)
# ─────────────────────────────────────────────

def _parse_fs_doc(doc: dict) -> dict:
    doc_id = doc.get("name", "").split("/")[-1]
    parsed = {k: _parse_fs_value(v) for k, v in doc.get("fields", {}).items()}
    parsed["_id"] = doc_id
    return parsed


async def _fetch_all_firestore(collection: str) -> list:
    token = await _get_auth_token()
    docs = []
    page_token = None
    async with httpx.AsyncClient(timeout=30) as client:
        while True:
            params = {"pageSize": 100}
            if page_token:
                params["pageToken"] = page_token
            resp = await client.get(
                _firestore_url(collection),
                headers={"Authorization": f"Bearer {token}"},
                params=params,
            )
            if not resp.is_success:
                logger.error(f"❌ Firestore {collection}: {resp.status_code} — {resp.text[:200]}")
                break
            data = resp.json()
            for doc in data.get("documents", []):
                docs.append(_parse_fs_doc(doc))
            page_token = data.get("nextPageToken")
            if not page_token:
                break
    logger.info(f"📂 {len(docs)} docs récupérés depuis '{collection}'")
    return docs


async def _is_already_enriched(doc_id: str) -> bool:
    token = await _get_auth_token()
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            _firestore_url(f"enrichments/{doc_id}"),
            headers={"Authorization": f"Bearer {token}"},
        )
        return resp.status_code == 200


# ─────────────────────────────────────────────
# Batch enrichment
# ─────────────────────────────────────────────

async def _enrich_registration(reg: dict, counters: dict):
    """Enrichit une inscription — appelé en parallèle via semaphore."""
    doc_id     = reg.get("_id")
    user_id    = reg.get("userId") or doc_id
    first_name = reg.get("userFirstName", "")
    last_name  = reg.get("userLastName", "")
    email      = reg.get("userEmail", "")

    if not first_name and not last_name:
        logger.warning(f"⏭  Skipping {doc_id}: pas de nom")
        counters["skipped"] += 1
        return

    if await _is_already_enriched(user_id):
        logger.info(f"⏭  Déjà enrichi : {first_name} {last_name}")
        counters["skipped"] += 1
        return

    full_name    = f"{first_name} {last_name}".strip()
    session_id   = f"reg_{user_id}_{uuid.uuid4().hex[:6]}"
    company_hint = ""
    if email and "@" in email:
        domain = email.split("@")[-1].lower()
        if domain not in _GENERIC_EMAIL_DOMAINS:
            company_hint = domain.split(".")[0]

    initial_state = {
        "visiteur_id": user_id,
        "firestore_collection": "event_registrations",
        "visiteur_data": {
            "visiteur_id":    user_id,
            "source_doc_id":  doc_id,
            "type":           "individual",
            "full_name":      full_name,
            "prenom":         first_name,
            "nom":            last_name,
            "email":          email,
            "company":        company_hint,
            "event_title":    reg.get("eventTitle", ""),
            "event_location": reg.get("eventLocation", ""),
        },
    }
    message = f"Enrich profile for {full_name}, email: {email}"
    if company_hint:
        message += f", company: {company_hint}"

    if user_id in _in_progress:
        logger.info(f"⏭  Déjà en cours : {full_name} ({user_id})")
        counters["skipped"] += 1
        return

    _in_progress.add(user_id)
    async with _sem:
        try:
            await _run_enrichment(session_id, message, initial_state, "batch_registrations")
            counters["processed"] += 1
            logger.info(f"✅ Done : {full_name} → enrichments/{user_id}")
        except Exception as e:
            counters["failed"] += 1
            logger.error(f"❌ Échec : {full_name} : {e}")
        finally:
            _in_progress.discard(user_id)


async def _batch_enrich_registrations():
    logger.info("▶ Batch event_registrations : démarré")
    try:
        registrations = await _fetch_all_firestore("event_registrations")
        counters = {"processed": 0, "skipped": 0, "failed": 0}
        await asyncio.gather(*[_enrich_registration(r, counters) for r in registrations])
        logger.info(
            f"📊 Batch registrations — "
            f"Traités: {counters['processed']} | "
            f"Ignorés: {counters['skipped']} | "
            f"Échecs: {counters['failed']}"
        )
    except Exception as e:
        logger.error(f"❌ Batch registrations error: {e}")
        import traceback
        logger.error(traceback.format_exc())


async def _enrich_company(company: dict, counters: dict):
    """Enrichit une company — appelé en parallèle via semaphore."""
    doc_id   = company.get("_id")
    name     = company.get("name", "")
    members  = company.get("members", [])

    if not name:
        logger.warning(f"⏭  Skipping {doc_id}: pas de nom")
        counters["skipped"] += 1
        return

    if await _is_already_enriched(doc_id):
        logger.info(f"⏭  Déjà enrichi : {name}")
        counters["skipped"] += 1
        return

    organizer = next(
        (m for m in members if isinstance(m, dict) and m.get("role") == "ORGANIZER"),
        members[0] if members and isinstance(members[0], dict) else {}
    )
    organizer_name = (
        f"{organizer.get('firstName', '')} {organizer.get('lastName', '')}".strip()
        if organizer else ""
    )

    initial_state = {
        "visiteur_id": doc_id,
        "firestore_collection": "companies",
        "visiteur_data": {
            "visiteur_id":    doc_id,
            "full_name":      name,
            "nom":            name,
            "prenom":         "",
            "email":          company.get("email", ""),
            "company":        name,
            "website":        company.get("website") or "",
            "industry":       company.get("industry") or "",
            "phone":          company.get("phone") or "",
            "address":        company.get("address") or "",
            "slug":           company.get("slug", ""),
            "organizer_name": organizer_name,
            "members_count":  len(members),
            "type":           "company",
        },
    }
    message = f"Enrich company profile for {name}, email: {company.get('email', '')}, industry: {company.get('industry') or 'unknown'}"
    if organizer_name:
        message += f", organizer: {organizer_name}"

    if doc_id in _in_progress:
        logger.info(f"⏭  Déjà en cours : {name} ({doc_id})")
        counters["skipped"] += 1
        return

    _in_progress.add(doc_id)
    async with _sem:
        try:
            session_id = f"company_{doc_id}_{uuid.uuid4().hex[:6]}"
            await _run_enrichment(session_id, message, initial_state, "batch_companies")
            counters["processed"] += 1
            logger.info(f"✅ Done : {name} → enrichments/{doc_id}")
        except Exception as e:
            counters["failed"] += 1
            logger.error(f"❌ Échec : {name} : {e}")
        finally:
            _in_progress.discard(doc_id)


async def _batch_enrich_companies():
    logger.info("▶ Batch companies : démarré")
    try:
        companies = await _fetch_all_firestore("companies")
        counters = {"processed": 0, "skipped": 0, "failed": 0}
        await asyncio.gather(*[_enrich_company(c, counters) for c in companies])
        logger.info(
            f"📊 Batch companies — "
            f"Traités: {counters['processed']} | "
            f"Ignorés: {counters['skipped']} | "
            f"Échecs: {counters['failed']}"
        )
    except Exception as e:
        logger.error(f"❌ Batch companies error: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
