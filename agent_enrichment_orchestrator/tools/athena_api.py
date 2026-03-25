"""
Athena API Tool - Interfaces with the FastAPI backend.
Implements ADK function tool pattern for API communication.
"""

import httpx
from typing import Optional, List, Dict, Any
from google.adk.tools import FunctionTool
from config.settings import settings
from models.visiteur import Visiteur, VisiteurUpdate
from models.enriched_profile import EnrichedProfile
import logging

logger = logging.getLogger(__name__)


class AthenaAPITool:
    """
    Tool for interacting with Athena Event API.
    Follows ADK best practices for external API integration.
    """
    
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.base_url = base_url or settings.athena_api_base_url
        self.api_key = api_key or settings.athena_api_key
        self.timeout = settings.timeout_seconds
        
        # Setup HTTP client
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers=self._get_headers()
        )
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers with authentication."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    async def get_visiteur_by_id(self, visiteur_id: str) -> Optional[Visiteur]:
        """
        Fetch visitor by ID.
        
        Args:
            visiteur_id: Unique visitor identifier
            
        Returns:
            Visiteur object or None if not found
        """
        try:
            response = await self.client.get(f"/visiteurs/{visiteur_id}")
            response.raise_for_status()
            return Visiteur(**response.json())
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching visiteur {visiteur_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching visiteur {visiteur_id}: {e}")
            return None
    
    async def get_visiteur_by_email(self, email: str) -> Optional[Visiteur]:
        """
        Fetch visitor by email.
        
        Args:
            email: Visitor email address
            
        Returns:
            Visiteur object or None if not found
        """
        try:
            response = await self.client.get(f"/visiteurs/email/{email}")
            response.raise_for_status()
            return Visiteur(**response.json())
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching visiteur by email {email}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching visiteur by email {email}: {e}")
            return None
    
    async def get_visiteurs_by_event(self, event_id: str) -> List[Visiteur]:
        """
        Fetch all visitors for an event.
        
        Args:
            event_id: Event identifier
            
        Returns:
            List of Visiteur objects
        """
        try:
            response = await self.client.get(f"/visiteurs/event/{event_id}")
            response.raise_for_status()
            data = response.json()
            return [Visiteur(**v) for v in data]
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching visiteurs for event {event_id}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching visiteurs for event {event_id}: {e}")
            return []
    
    async def update_visiteur(
        self,
        visiteur_id: str,
        update_data: VisiteurUpdate
    ) -> Optional[Visiteur]:
        """
        Update visitor information.
        
        Args:
            visiteur_id: Visitor ID to update
            update_data: Update data
            
        Returns:
            Updated Visiteur object or None on error
        """
        try:
            response = await self.client.put(
                f"/visiteurs/{visiteur_id}",
                json=update_data.model_dump(exclude_none=True)
            )
            response.raise_for_status()
            return Visiteur(**response.json())
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error updating visiteur {visiteur_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error updating visiteur {visiteur_id}: {e}")
            return None
    
    async def send_enriched_profile(
        self,
        visiteur_id: str,
        enriched_profile: EnrichedProfile
    ) -> bool:
        """
        Send enriched profile data back to Athena API.
        This creates a callback with the enriched data.
        
        Args:
            visiteur_id: Visitor ID
            enriched_profile: Enriched profile data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Update visitor metadata with enriched data
            metadata = {
                "enrichment_status": enriched_profile.enrichment_status,
                "enrichment_timestamp": enriched_profile.enrichment_timestamp.isoformat(),
                "linkedin_profile": enriched_profile.linkedin_profile.model_dump() if enriched_profile.linkedin_profile else None,
                "facebook_profile": enriched_profile.facebook_profile.model_dump() if enriched_profile.facebook_profile else None,
                "professional_summary": enriched_profile.professional_summary,
                "interests_summary": enriched_profile.interests_summary,
                "networking_potential_score": enriched_profile.networking_potential_score,
                "overall_quality_score": enriched_profile.overall_quality_score,
                "data_sources_used": enriched_profile.data_sources_used,
                "errors": enriched_profile.errors,
                "warnings": enriched_profile.warnings
            }
            
            update_data = VisiteurUpdate(metadata=metadata)
            result = await self.update_visiteur(visiteur_id, update_data)
            
            if result:
                logger.info(f"Successfully sent enriched profile for visiteur {visiteur_id}")
                return True
            else:
                logger.error(f"Failed to send enriched profile for visiteur {visiteur_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending enriched profile for {visiteur_id}: {e}")
            return False
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# ADK Function Tools
# These are the tools that will be registered with ADK agents

def get_visiteur_tool() -> FunctionTool:
    """Create ADK function tool for fetching visitor data."""
    
    async def get_visiteur(visiteur_id: str) -> str:
        """
        Fetch visitor information from Athena API.
        
        Args:
            visiteur_id: The unique identifier of the visitor
            
        Returns:
            JSON string with visitor information
        """
        tool = AthenaAPITool()
        try:
            visiteur = await tool.get_visiteur_by_id(visiteur_id)
            if visiteur:
                return visiteur.model_dump_json()
            return f"Visitor with ID {visiteur_id} not found"
        finally:
            await tool.close()
    
    return FunctionTool(get_visiteur)


def get_event_visiteurs_tool() -> FunctionTool:
    """Create ADK function tool for fetching all visitors of an event."""
    
    async def get_event_visiteurs(event_id: str) -> str:
        """
        Fetch all visitors for a specific event.
        
        Args:
            event_id: The unique identifier of the event
            
        Returns:
            JSON string with list of visitors
        """
        tool = AthenaAPITool()
        try:
            visiteurs = await tool.get_visiteurs_by_event(event_id)
            return [v.model_dump() for v in visiteurs]
        finally:
            await tool.close()
    
    return FunctionTool(get_event_visiteurs)


def send_enriched_data_tool() -> FunctionTool:
    """Create ADK function tool for sending enriched profile data."""
    
    async def send_enriched_data(
        visiteur_id: str,
        enriched_data: str
    ) -> str:
        """
        Send enriched profile data back to Athena API.
        
        Args:
            visiteur_id: The unique identifier of the visitor
            enriched_data: JSON string containing enriched profile data
            
        Returns:
            Success or error message
        """
        tool = AthenaAPITool()
        try:
            import json
            data = json.loads(enriched_data)
            enriched_profile = EnrichedProfile(**data)
            
            success = await tool.send_enriched_profile(visiteur_id, enriched_profile)
            if success:
                return f"Successfully sent enriched profile for visitor {visiteur_id}"
            else:
                return f"Failed to send enriched profile for visitor {visiteur_id}"
        except Exception as e:
            return f"Error sending enriched data: {str(e)}"
        finally:
            await tool.close()
    
    return FunctionTool(send_enriched_data)