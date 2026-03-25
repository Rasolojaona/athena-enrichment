"""
Configuration module using Pydantic for type-safe settings management.
Follows ADK best practices for environment-based configuration.
"""

from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings with validation."""
    
    # Athena API
    athena_api_base_url: str = Field(
        default="http://localhost:8000/api/v1",
        description="Base URL for Athena Event API"
    )
    athena_api_key: Optional[str] = Field(
        default=None,
        description="API key for Athena Event API"
    )
    
    # Google Gemini (optional when using Vertex AI: GOOGLE_GENAI_USE_VERTEXAI=1)
    google_api_key: Optional[str] = Field(
        default=None,
        description="Google API key for Gemini models (not needed with Vertex AI)"
    )
    
    # LinkedIn APIs (au moins une requis)
    proxycurl_api_key: Optional[str] = Field(
        default=None,
        description="Proxycurl API key for LinkedIn enrichment"
    )
    serpapi_key: Optional[str] = Field(
        default=None,
        description="SerpAPI key for Google Search"
    )
    
    # Facebook API
    apify_api_key: Optional[str] = Field(
        default=None,
        description="Apify API key for Facebook enrichment"
    )
    rapidapi_key: Optional[str] = Field(
        default=None,
        description="RapidAPI key for social media APIs"
    )
    
    # Agent Configuration
    max_retries: int = Field(default=3, ge=1, le=10)
    timeout_seconds: int = Field(default=30, ge=5, le=300)
    enable_caching: bool = Field(default=True)
    
    # Logging
    log_level: str = Field(default="INFO")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = 'ignore'
    
    # Validation désactivée pour permettre les tests sans toutes les APIs
    # Décommentez ces validators pour forcer la configuration en production
    
    # @validator("proxycurl_api_key", "serpapi_key", always=True)
    # def validate_linkedin_api(cls, v, values):
    #     """Ensure at least one LinkedIn API key is provided."""
    #     if not v and not values.get("proxycurl_api_key") and not values.get("serpapi_key"):
    #         raise ValueError("At least one LinkedIn API key (proxycurl or serpapi) must be provided")
    #     return v
    
    # @validator("apify_api_key", "rapidapi_key", always=True)
    # def validate_facebook_api(cls, v, values):
    #     """Ensure at least one Facebook API key is provided."""
    #     if not v and not values.get("apify_api_key") and not values.get("rapidapi_key"):
    #         raise ValueError("At least one Facebook API key (apify or rapidapi) must be provided")
    #     return v
    
    def get_linkedin_provider(self) -> str:
        """Return the preferred LinkedIn API provider."""
        if self.proxycurl_api_key:
            return "proxycurl"
        elif self.serpapi_key:
            return "serpapi"
        else:
            # Retourner None au lieu de raise pour permettre les tests
            return None
    
    def get_facebook_provider(self) -> str:
        """Return the preferred Facebook API provider."""
        if self.apify_api_key:
            return "apify"
        elif self.rapidapi_key:
            return "rapidapi"
        else:
            # Retourner None au lieu de raise pour permettre les tests
            return None


# Singleton instance
settings = Settings()