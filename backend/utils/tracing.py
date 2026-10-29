from typing import Optional

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from backend.logger import GLOBAL_LOGGER as log
from backend.utils.model_loader import ApiKeyManager

DEFAULT_LANGFUSE_HOST = "https://cloud.langfuse.com"

_client: Optional[Langfuse] = None


def build_langfuse_handler(api_keys: ApiKeyManager) -> Optional[CallbackHandler]:
    """Create the Langfuse client and LangChain callback handler, or None when keys are absent."""
    global _client

    public_key = api_keys.api_keys.get("LANGFUSE_PUBLIC_KEY")
    secret_key = api_keys.api_keys.get("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        log.info("Langfuse keys not set, tracing disabled")
        return None

    host = api_keys.api_keys.get("LANGFUSE_HOST") or DEFAULT_LANGFUSE_HOST
    # Passing keys explicitly works for keys from the ECS secret, which are not in os.environ.
    _client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
    log.info("Langfuse tracing enabled", host=host)
    return CallbackHandler()


def flush_langfuse() -> None:
    """Send any buffered traces before the process exits."""
    if _client is not None:
        _client.flush()
