"""
Enrichment Orchestrator - Agent ADK Package.

Package principal contenant l'agent orchestrateur et tous les sub-agents
pour l'enrichissement de profils visiteurs.
"""

from .agent import root_agent

__all__ = ["root_agent"]