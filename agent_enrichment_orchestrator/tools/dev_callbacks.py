"""
Dev Callbacks — Cache fichier JSON + Logging structuré.

Activer avec DEV_CACHE_ENABLED=true dans le .env.
Le cache persiste entre les runs dans .dev_cache.json à la racine du projet.

Usage :
    from ...tools.dev_callbacks import get_dev_callbacks

    _dev = get_dev_callbacks()  # None si DEV_CACHE_ENABLED != true

    LlmAgent(
        ...
        before_tool_callback=_dev.before_tool if _dev else None,
        after_tool_callback=_dev.after_tool  if _dev else None,
    )
"""

import os
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

logger = logging.getLogger(__name__)

# ── Outils mis en cache ───────────────────────────────────────────────────────
CACHEABLE_TOOLS = {
    "search_linkedin_profile",
    "search_facebook_profile",
    "search_person_web_presence",
    "scrape_company_website",
    "extract_web_content",
}

# Clé session.state écrite par chaque outil
TOOL_STATE_KEYS = {
    "search_linkedin_profile":    "linkedin_profile_data",
    "search_facebook_profile":    "facebook_profile_data",
    "search_person_web_presence": "web_profile_data",
    "scrape_company_website":     "web_profile_data",
    "extract_web_content":        "web_extracted_data",
}

# TTL en secondes par outil
CACHE_TTLS = {
    "search_linkedin_profile":    7 * 24 * 3600,  # 7 jours  — profil stable
    "search_facebook_profile":    7 * 24 * 3600,  # 7 jours
    "search_person_web_presence": 1 * 24 * 3600,  # 1 jour   — mentions changent plus vite
    "scrape_company_website":     1 * 24 * 3600,  # 1 jour
    "extract_web_content":        1 * 24 * 3600,  # 1 jour
}

# Coût estimé en USD par appel (pour le logging)
TOOL_COSTS = {
    "search_linkedin_profile":    0.08,  # Apify Full profile
    "search_facebook_profile":    0.02,  # Apify Facebook
    "search_person_web_presence": 0.01,  # Serper queries
    "scrape_company_website":     0.00,  # HTTP direct
    "extract_web_content":        0.01,  # Jina.ai + Gemini
}


# ── Cache fichier JSON ────────────────────────────────────────────────────────

class DevCache:
    """
    Cache fichier JSON persistant entre les runs de test.
    Stocké dans .dev_cache.json à la racine du projet.
    """

    def __init__(self, cache_file: Path):
        self.path = cache_file
        self._data: Dict[str, Any] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
                valid = self._count_valid()
                logger.info(f"🗄️  Dev cache chargé : {valid} entrées valides / {len(self._data)} total ({self.path})")
            except Exception as e:
                logger.warning(f"Dev cache load failed: {e}")
                self._data = {}

    def _save(self):
        try:
            self.path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Dev cache save failed: {e}")

    def _make_key(self, tool_name: str, full_name: str, email: str = "") -> str:
        raw = f"{tool_name}:{full_name.lower().strip()}:{email.lower().strip()}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _count_valid(self) -> int:
        now = time.time()
        return sum(
            1 for e in self._data.values()
            if now - e.get("ts", 0) < max(CACHE_TTLS.values())
        )

    def get(self, tool_name: str, full_name: str, email: str = "") -> Optional[Any]:
        """Retourne la donnée cachée ou None si absent/expiré."""
        key = self._make_key(tool_name, full_name, email)
        entry = self._data.get(key)
        if not entry:
            return None
        ttl = CACHE_TTLS.get(tool_name, 86400)
        age = time.time() - entry.get("ts", 0)
        if age > ttl:
            del self._data[key]
            self._save()
            return None
        return entry["data"]

    def set(self, tool_name: str, full_name: str, data: Any, email: str = ""):
        """Sauvegarde la donnée dans le cache."""
        key = self._make_key(tool_name, full_name, email)
        self._data[key] = {"ts": time.time(), "data": data, "tool": tool_name, "name": full_name}
        self._save()

    def stats(self) -> Dict:
        return {"total": len(self._data), "valid": self._count_valid()}

    def clear(self):
        """Vide le cache (utile pour forcer un re-enrichissement)."""
        self._data = {}
        self._save()
        logger.info("🗑️  Dev cache vidé")


# ── Callbacks combinés (cache + logging) ─────────────────────────────────────

class DevCallbacks:
    """
    Callbacks ADK combinant cache fichier + logging structuré.

    before_tool : vérifie le cache → si HIT, restaure session.state et skip l'outil
    after_tool  : sauvegarde session.state dans le cache + log timing/coût
    """

    def __init__(self, cache_file: Path):
        self.cache = DevCache(cache_file)
        # timing_key = "{tool_name}:{id(tool_context)}" → start_time
        self._timings: Dict[str, float] = {}

    def _get_identity(self, args: Dict, tool_context: ToolContext):
        """Extrait (full_name, email) depuis args ou session.state."""
        # 1. Depuis les args du tool (cas search_linkedin_profile, etc.)
        full_name = args.get("full_name") or args.get("company_name") or ""
        email     = args.get("email") or ""

        # 2. Fallback : visiteur_data en session.state
        if not full_name:
            vd = tool_context.state.get("visiteur_data") or {}
            if isinstance(vd, str):
                try:
                    vd = json.loads(vd)
                except Exception:
                    vd = {}
            full_name = vd.get("full_name") or ""
            if not email:
                email = vd.get("email") or ""

        return full_name.strip(), email.strip()

    def before_tool(
        self,
        tool: BaseTool,
        args: Dict,
        tool_context: ToolContext,
    ) -> Optional[Dict]:
        """
        1. Démarre le chrono pour le logging.
        2. Vérifie le cache → si HIT, restaure session.state et retourne
           un résultat fictif pour que ADK skip l'appel réel.
        """
        timing_key = f"{tool.name}:{id(tool_context)}"
        self._timings[timing_key] = time.time()

        if tool.name not in CACHEABLE_TOOLS:
            return None

        full_name, email = self._get_identity(args, tool_context)
        if not full_name:
            return None

        cached = self.cache.get(tool.name, full_name, email)
        if cached is None:
            logger.info(f"🔍 MISS [{tool.name}] {full_name}")
            return None

        # ── Cache HIT ────────────────────────────────────────────────────────
        state_key = TOOL_STATE_KEYS.get(tool.name)
        if state_key:
            tool_context.state[state_key] = cached

        age_min = int((time.time() - self.cache._data.get(
            self.cache._make_key(tool.name, full_name, email), {}
        ).get("ts", time.time())) / 60)

        saved_cost = TOOL_COSTS.get(tool.name, 0)
        logger.info(
            f"💾 HIT  [{tool.name}] {full_name}"
            f" | âge: {age_min}min | économie: ~${saved_cost:.2f}"
        )
        # Retourner un faux résultat — ADK skip l'appel réel
        return {"result": f"{state_key} stored: [CACHED] for {full_name}"}

    def after_tool(
        self,
        tool: BaseTool,
        args: Dict,
        tool_context: ToolContext,
        tool_response: Dict,
    ) -> Optional[Dict]:
        """
        1. Calcule le temps écoulé et log le résultat structuré.
        2. Sauvegarde session.state dans le cache si données valides.
        """
        timing_key = f"{tool.name}:{id(tool_context)}"
        elapsed = time.time() - self._timings.pop(timing_key, time.time())

        full_name, email = self._get_identity(args, tool_context)
        state_key  = TOOL_STATE_KEYS.get(tool.name)
        state_data = tool_context.state.get(state_key) if state_key else None

        # Extraire le message de retour (tool_response peut être str ou dict)
        if isinstance(tool_response, dict):
            response_text = (
                tool_response.get("result")
                or tool_response.get("output")
                or str(tool_response)
            )
        else:
            response_text = str(tool_response) if tool_response else ""
        is_cached = "[CACHED]" in str(response_text)
        cost      = TOOL_COSTS.get(tool.name, 0) if not is_cached else 0.0

        # ── Log structuré ─────────────────────────────────────────────────────
        if is_cached:
            logger.info(f"   ↳ [{tool.name}] {full_name} | depuis cache (0s | $0.00)")
        else:
            # Extraire quelques métriques du state_data pour le log
            extra = ""
            if isinstance(state_data, dict):
                skills_n = len(state_data.get("skills") or [])
                exp_n    = len(state_data.get("experience") or [])
                provider = state_data.get("provider") or state_data.get("status") or ""
                if skills_n or exp_n:
                    extra = f" | skills={skills_n} exp={exp_n}"
                if provider:
                    extra += f" | via {provider}"
            logger.info(
                f"✅ DONE [{tool.name}] {full_name}"
                f" | {elapsed:.1f}s | ~${cost:.2f}{extra}"
            )

        # ── Sauvegarde dans le cache ──────────────────────────────────────────
        if (
            tool.name in CACHEABLE_TOOLS
            and state_data
            and not is_cached
            and isinstance(state_data, dict)
            and state_data.get("status") not in ("not_found", "error")
        ):
            self.cache.set(tool.name, full_name, state_data, email)
            logger.debug(f"   💾 SET [{tool.name}] {full_name}")

        return None  # ne pas modifier la réponse


# ── Singleton ─────────────────────────────────────────────────────────────────

_instance: Optional[DevCallbacks] = None

def get_dev_callbacks() -> Optional[DevCallbacks]:
    """
    Retourne le singleton DevCallbacks si DEV_CACHE_ENABLED=true,
    sinon None (aucun callback attaché → comportement normal).
    """
    global _instance
    if os.getenv("DEV_CACHE_ENABLED", "false").lower() != "true":
        return None
    if _instance is None:
        cache_path = Path(os.getenv("DEV_CACHE_FILE", ".dev_cache.json"))
        _instance = DevCallbacks(cache_path)
    return _instance
