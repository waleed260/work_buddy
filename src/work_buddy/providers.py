"""
OpenRouter provider configuration for Remote Work Buddy.
Simplified to only use OpenRouter.
"""

import os
from typing import Optional
from agents.models.openai_provider import OpenAIProvider
from agents.models.interface import Model


def create_model_from_env() -> Optional[Model]:
    """
    Create a model from environment variables for OpenRouter.

    Environment variables:
        OPENAI_API_KEY: OpenRouter API key
        OPENAI_BASE_URL: OpenRouter base URL (default: https://openrouter.ai/api/v1)
        OPENAI_MODEL: Model name (default: openai/gpt-3.5-turbo)
        OPENROUTER_APP_NAME: App name for tracing (optional)
        OPENROUTER_SITE_URL: Site URL for tracing (optional)

    Returns:
        Model instance or None if API key not set
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    model_name = os.getenv("OPENAI_MODEL", "openai/gpt-3.5-turbo")

    # OpenRouter tracing headers
    default_headers = {}
    app_name = os.getenv("OPENROUTER_APP_NAME", "Remote Work Buddy")
    site_url = os.getenv("OPENROUTER_SITE_URL", "https://github.com/waleed260/work_buddy")

    if app_name:
        default_headers["HTTP-Referer"] = site_url
        default_headers["X-Title"] = app_name

    # Create OpenRouter provider
    provider = OpenAIProvider(
        api_key=api_key,
        base_url=base_url,
        use_responses=False
    )

    # Get model
    model = provider.get_model(model_name)

    return model


def create_model(
    api_key: str,
    base_url: Optional[str] = None,
    model_name: Optional[str] = None,
    app_name: Optional[str] = None,
    site_url: Optional[str] = None,
) -> Model:
    """
    Create a model instance for OpenRouter.

    Args:
        api_key: OpenRouter API key
        base_url: OpenRouter base URL (optional, defaults to https://openrouter.ai/api/v1)
        model_name: Model name (optional, defaults to openai/gpt-3.5-turbo)
        app_name: App name for tracing (optional)
        site_url: Site URL for tracing (optional)

    Returns:
        Model instance ready to use
    """
    final_base_url = base_url or "https://openrouter.ai/api/v1"
    final_model = model_name or "openai/gpt-3.5-turbo"

    # OpenRouter tracing headers
    default_headers = {}
    if app_name or site_url:
        default_headers["HTTP-Referer"] = site_url or "https://github.com/waleed260/work_buddy"
        default_headers["X-Title"] = app_name or "Remote Work Buddy"

    # Create provider
    provider = OpenAIProvider(
        api_key=api_key,
        base_url=final_base_url,
        use_responses=False
    )

    # Get model
    model = provider.get_model(final_model)

    return model

