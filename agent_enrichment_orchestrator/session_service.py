"""
Session Service Factory — InMemory (dev) ou Database (prod GCP).

Sélectionne automatiquement le bon service selon l'environnement :
  - LOCAL  (DATABASE_URL absent) → InMemorySessionService  (comportement actuel)
  - PROD   (DATABASE_URL défini) → DatabaseSessionService  (Cloud SQL PostgreSQL)

Cloud Run + Cloud SQL (via Unix socket) :
  DATABASE_URL=postgresql+pg8000://USER:PASSWORD@/DB_NAME?unix_sock=/cloudsql/PROJECT:REGION:INSTANCE/.s.PGSQL.5432

Variables d'environnement :
  DATABASE_URL   URL SQLAlchemy complète (absent = InMemory)
  SESSION_TTL    Durée de vie des sessions en secondes (défaut: 3600)
"""

import os
import logging

logger = logging.getLogger(__name__)


def get_session_service():
    """
    Retourne InMemorySessionService ou DatabaseSessionService selon DATABASE_URL.

    L'API est identique dans les deux cas — aucun changement dans main.py
    sauf l'import et la création du runner.
    """
    db_url = os.getenv("DATABASE_URL", "").strip()

    if not db_url:
        from google.adk.sessions import InMemorySessionService
        logger.info("Session service : InMemorySessionService (dev local)")
        return InMemorySessionService()

    from google.adk.sessions import DatabaseSessionService

    logger.info("Session service : DatabaseSessionService (persistant)")
    logger.info(f"  DB URL : {_mask_url(db_url)}")

    service = DatabaseSessionService(db_url=db_url)
    return service


def _mask_url(url: str) -> str:
    """Masque le mot de passe dans l'URL pour les logs."""
    import re
    return re.sub(r'://([^:]+):([^@]+)@', r'://\1:***@', url)
