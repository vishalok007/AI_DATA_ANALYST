import os
from dotenv import load_dotenv

from utils.groq_provider import generate as groq_generate
from utils.gemini_provider import generate as gemini_generate


load_dotenv()
AI_PROVIDER = os.getenv("AI_PROVIDER", "groq").lower()
PROVIDER_LABELS = {
    "groq": "Groq",
    "gemini": "Google Gemini",
}


def _provider_order():
    if AI_PROVIDER == "gemini":
        return ["gemini", "groq"]
    return ["groq", "gemini"]


def _call_provider(provider_name, prompt):
    if provider_name == "groq":
        return groq_generate(prompt)
    if provider_name == "gemini":
        return gemini_generate(prompt)
    raise ValueError("Invalid AI_PROVIDER")


def _looks_like_error(response):
    if response is None:
        return True

    text = str(response).strip()
    if text == "":
        return True

    lowered = text.lower()
    error_markers = [
        "ai service unavailable",
        "groq error",
        "gemini error",
        "invalid ai_provider",
        "api key",
        "authentication",
        "unauthorized",
    ]
    return any(marker in lowered for marker in error_markers)


def generate_ai_response_with_source(prompt):
    errors = []

    for provider_name in _provider_order():
        provider_label = PROVIDER_LABELS.get(provider_name, provider_name.title())
        try:
            response = _call_provider(provider_name, prompt)
            if not _looks_like_error(response):
                return response, provider_label
            errors.append(f"{provider_label}: {str(response).strip()[:240]}")
        except Exception as error:
            errors.append(f"{provider_label}: {error}")

    details = "\n".join(f"- {item}" for item in errors) if errors else "- No provider response"
    return f"# AI Service Unavailable\n{details}", "Unavailable"


def generate_ai_response(prompt):
    response, _ = generate_ai_response_with_source(prompt)
    return response
