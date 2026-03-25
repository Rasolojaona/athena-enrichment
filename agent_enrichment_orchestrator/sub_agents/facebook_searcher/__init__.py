"""
Facebook Searcher Sub-Agent Package.

Agent spécialisé dans la recherche de profils Facebook.
Utilise Apify pour le scraping de profils publics.
"""

from .agent import facebook_agent

__all__ = ["facebook_agent"]