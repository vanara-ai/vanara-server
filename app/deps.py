"""FastAPI dependencies for BYOK (Bring Your Own Key) authentication.

Users send their Groq API key on every request via the X-Groq-Key header.
The key is used only for that request — never logged, never persisted.
"""

from fastapi import Header, HTTPException

_MISSING_KEY_MESSAGE = (
    "Missing X-Groq-Key header. Add your Groq API key in the app settings. "
    "Get a free key at https://console.groq.com/keys — your key stays in "
    "your browser and is sent directly to Groq on each request."
)


def get_groq_key(x_groq_key: str | None = Header(default=None)) -> str:
    if not x_groq_key or not x_groq_key.strip():
        raise HTTPException(status_code=401, detail=_MISSING_KEY_MESSAGE)
    return x_groq_key.strip()
