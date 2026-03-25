"""
Web Searcher tools package.
"""

from .web_scraper import get_web_scraper_tool
from .person_web_search import get_person_web_search_tool

__all__ = ["get_web_scraper_tool", "get_person_web_search_tool"]
